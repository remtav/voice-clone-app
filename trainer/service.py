"""The trainer daemon: picks queued runs from the database and executes them.

GPU hand-off with the web app (same machine, one GPU, one user at a time):

1. A queued run moves to ``waiting_gpu``.  From then on the app holds new
   generations and unloads its model as soon as no generation is running.
2. The trainer starts once the app's heartbeat says the model is unloaded (or
   the heartbeat is stale: the app is not running) **and** ``nvidia-smi``
   reports enough free memory.
3. When the run ends (done, failed or cancelled) it leaves the active states;
   the app reloads its model on the next generation.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections.abc import Callable

from app.db import Database
from app.training import APP_HEARTBEAT_KEY, RUN_LOG_NAME, TRAINER_HEARTBEAT_KEY
from trainer import runner as runner_module
from trainer.config import TrainerSettings
from trainer.progress import ProgressParser, is_noise, is_progress_bar
from trainer.recipes import PROJECT_ROOT, plan

log = logging.getLogger("trainer")


def gpu_status() -> dict | None:
    """Name and memory of GPU 0 from nvidia-smi, or None when no NVIDIA GPU is visible."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    name, total, free = (part.strip() for part in out[0].split(","))
    return {"name": name, "total_gb": round(float(total) / 1024, 1), "free_gb": round(float(free) / 1024, 1)}


class RunEnded(Exception):
    """A run left the running state (failed or cancelled); message already recorded."""


class Trainer:
    def __init__(
        self,
        db: Database,
        settings: TrainerSettings,
        gpu_probe: Callable[[], dict | None] = gpu_status,
        run_command: Callable[..., runner_module.Result] = runner_module.run,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.db = db
        self.settings = settings
        self.gpu_probe = gpu_probe
        self.run_command = run_command
        self.clock = clock
        self.sleep = sleep
        self.stopping = threading.Event()
        self.current_run: str | None = None

    # Lifecycle -------------------------------------------------------------
    def recover(self) -> None:
        count = self.db.fail_interrupted_runs(
            "The trainer restarted during this run and the toolkit cannot resume it. Start a new run.")
        if count:
            log.warning("Marked %d interrupted run(s) as failed", count)

    def heartbeat(self) -> None:
        self.db.set_kv(TRAINER_HEARTBEAT_KEY, {"t": self.clock(), "gpu": self.gpu_probe(), "busy": self.current_run,
                                       "allow_cpu": self.settings.allow_cpu})

    def run_forever(self) -> None:
        self.recover()
        beat = threading.Thread(target=self._beat_loop, name="trainer-heartbeat", daemon=True)
        beat.start()
        log.info("Trainer ready; data dir %s", self.settings.data_dir.resolve())
        while not self.stopping.is_set():
            self.tick()
            self.stopping.wait(self.settings.poll_seconds)

    def _beat_loop(self) -> None:
        while not self.stopping.is_set():
            try:
                self.heartbeat()
            except Exception:  # noqa: BLE001 - a missed beat must not kill the thread
                log.exception("Heartbeat failed")
            self.stopping.wait(self.settings.heartbeat_seconds)

    def tick(self) -> None:
        run = self.db.active_run()
        if run is None:
            return
        if run["status"] == "queued":
            self.execute(run)
        elif run["status"] == "cancelling" and self.current_run != run["id"]:
            self._finish(run["id"], "cancelled", "Cancelled")  # nobody is running it

    # One run ---------------------------------------------------------------
    def execute(self, run: dict) -> None:
        run_id = run["id"]
        if not self.db.claim_run(run_id, "queued", "waiting_gpu"):
            return
        self.current_run = run_id
        run_dir = self.settings.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_file = (run_dir / RUN_LOG_NAME).open("a", encoding="utf-8")
        try:
            self._check_prerequisites(run)
            device = self._wait_for_gpu(run_id)
            if not self.db.claim_run(run_id, "waiting_gpu", "running"):
                raise RunEnded
            self.db.update_run(run_id, started_at=self.clock(), message=None)
            log.info("Run %s (%s) started on %s", run_id, run["recipe"], device)
            for step in plan(run, self.settings, device):
                self._run_step(run_id, step, log_file)
            output = f"{run['params']['output_name']}.safetensors"
            self.db.update_run(run_id, output_model=output)
            self._finish(run_id, "done", f"Model ready: {output}")
        except RunEnded:
            pass
        except Exception as exc:  # noqa: BLE001 - any failure ends the run with its reason
            log.exception("Run %s failed", run_id)
            self._finish(run_id, "failed", f"{type(exc).__name__}: {exc}")
        finally:
            log_file.close()
            self.current_run = None

    def _check_prerequisites(self, run: dict) -> None:
        if run["recipe"] == "personal":
            base = self.settings.models_dir / run["params"].get("base_model", "")
            if not base.is_file():
                raise FileNotFoundError(f"Base model not found: {base.name}")
            if not (self.settings.own_data / "metadata.csv").is_file():
                raise FileNotFoundError(f"No recordings of your voice in {self.settings.own_data}")

    def _cancel_requested(self, run_id: str) -> bool:
        if self.stopping.is_set():
            return True
        run = self.db.get_run(run_id)
        return run is None or run["status"] == "cancelling"

    def _end_if_cancelled(self, run_id: str) -> None:
        if self.stopping.is_set():
            self._finish(run_id, "failed", "The trainer service stopped during this run.")
            raise RunEnded
        if self._cancel_requested(run_id):
            self._finish(run_id, "cancelled", "Cancelled")
            raise RunEnded

    def _wait_for_gpu(self, run_id: str) -> str:
        """Wait until the app has unloaded its model and the GPU has enough free memory."""
        deadline = self.clock() + self.settings.gpu_wait_seconds
        while True:
            self._end_if_cancelled(run_id)
            gpu = self.gpu_probe()
            if gpu is None and not self.settings.allow_cpu:
                raise RuntimeError("No NVIDIA GPU is visible in the trainer container "
                                   "(check the compose GPU reservation and nvidia-container-toolkit).")
            beat = self.db.get_kv(APP_HEARTBEAT_KEY) or {}
            app_alive = self.clock() - float(beat.get("t", 0)) < self.settings.app_stale_seconds
            app_holds_model = app_alive and bool(beat.get("engine_loaded"))
            enough_memory = gpu is None or gpu["free_gb"] >= self.settings.min_free_vram_gb
            if not app_holds_model and enough_memory:
                return "cpu" if gpu is None else "cuda"
            reason = ("the app still holds its model (it finishes the current generation first)" if app_holds_model
                      else f"{gpu['free_gb']} GB free, {self.settings.min_free_vram_gb:g} GB needed")
            if self.clock() > deadline:
                raise RuntimeError(f"The GPU did not become available: {reason}.")
            self.db.update_run(run_id, message=f"Waiting for the GPU: {reason}")
            self.sleep(self.settings.poll_seconds)

    def _run_step(self, run_id: str, step, log_file) -> None:
        self._end_if_cancelled(run_id)
        if step.skip is not None and step.skip():
            return
        self.db.update_run(run_id, stage=step.label, message=None)
        log_file.write(f"\n=== {step.label} ===\n")
        log_file.flush()
        if step.action is not None:
            step.action()
            return
        parser = ProgressParser()
        state = {"db": 0.0, "log": 0.0}

        def on_line(line: str) -> None:
            now = self.clock()
            if step.training and parser.feed(line) and now - state["db"] >= 3:
                p = parser.progress
                self.db.update_run(run_id, epoch=p.epoch, epochs=p.epochs, step=p.step,
                                   total_steps=p.total_steps, loss=p.loss, eta_seconds=p.eta_seconds)
                state["db"] = now
            if is_noise(line):
                return
            if is_progress_bar(line):
                # tqdm redraws many times a second: keep finished bars and one line every 30 s
                if "100%|" not in line and now - state["log"] < 30:
                    return
                state["log"] = now
            log_file.write(line.rstrip() + "\n")
            log_file.flush()

        result = self.run_command(step.argv, PROJECT_ROOT, on_line, lambda: self._cancel_requested(run_id))
        if step.training:
            p = parser.progress
            self.db.update_run(run_id, epoch=p.epoch, epochs=p.epochs, step=p.step, total_steps=p.total_steps,
                               loss=p.loss, eta_seconds=0 if result.returncode == 0 else p.eta_seconds)
        if result.cancelled:
            self._end_if_cancelled(run_id)
        if result.returncode != 0:
            detail = " | ".join(line.strip() for line in result.tail[-3:] if not is_noise(line))
            raise RuntimeError(f"{step.label} failed (exit {result.returncode}): {detail}")

    def _finish(self, run_id: str, status: str, message: str) -> None:
        self.db.update_run(run_id, status=status, message=message, stage=None, finished_at=self.clock())
        log.info("Run %s %s: %s", run_id, status, message)

