"""Trainer service: progress parsing, subprocess runner, GPU hand-off and run lifecycle."""

from __future__ import annotations

import sys
import textwrap
import time
from pathlib import Path

import pytest

from app.db import Database
from app.training import APP_HEARTBEAT_KEY, RUN_LOG_NAME
from trainer import recipes, runner, service
from trainer.config import TrainerSettings
from trainer.progress import ProgressParser, is_noise
from trainer.recipes import Step

GPU_3090 = {"name": "NVIDIA GeForce RTX 3090", "total_gb": 24.0, "free_gb": 23.1}


# --------------------------------------------------------------------------- progress
def test_parses_the_toolkit_tqdm_line():
    parser = ProgressParser()
    # Captured from a real run of the patched toolkit.
    assert parser.feed("Epoch 1/1:  33%|███▎      | 1/3 [00:27<00:34, 17.05s/it, loss=2.8600, lr=0.000325]")
    p = parser.progress
    assert (p.epoch, p.epochs, p.step, p.total_steps, p.loss) == (1, 1, 1, 3, 2.86)
    assert p.eta_seconds == pytest.approx(34.1)


def test_progress_counts_steps_across_epochs_and_it_per_second():
    parser = ProgressParser()
    assert parser.feed("Epoch 2/4:  10%|█         | 360/3600 [02:00<18:00,  3.00it/s, loss=1.5, lr=2e-05]")
    p = parser.progress
    assert (p.step, p.total_steps) == (3960, 14400)
    assert p.eta_seconds == pytest.approx((14400 - 3960) / 3.0, abs=1)
    assert not parser.feed("Validation: 100%|██████████| 1/1 [00:08<00:00,  8.78s/it]")


def test_every_progress_bar_is_recognized():
    from trainer.progress import is_progress_bar

    assert is_progress_bar("Sampling:  12%|█▏        | 120/1000 [00:10<01:15, 11.5it/s]")
    assert is_progress_bar("Fetching 6 files: 100%|██████████| 6/6 [00:00<00:00, 1148.81it/s]")
    assert not is_progress_bar("Train samples: 3, Validation samples: 1")


def test_noise_filter_keeps_useful_lines():
    assert is_noise("Speech logits shape: torch.Size([2, 374, 8194])") and is_noise("Computed loss: 3.94")
    assert not is_noise("Saved checkpoint to /data/finetune/runs/x/checkpoint_epoch0_step2.pt")


# --------------------------------------------------------------------------- runner
def script(tmp_path: Path, body: str) -> list[str]:
    path = tmp_path / "step.py"
    path.write_text(textwrap.dedent(body))
    return [sys.executable, str(path)]


def test_runner_splits_carriage_returns_and_reports_exit_code(tmp_path: Path):
    argv = script(tmp_path, """
        import sys
        for i in range(3):
            sys.stdout.write(f"Epoch 1/1: {i}%|x| {i}/3 [00:01<00:02, 1.0s/it, loss=0.5]\\r")
            sys.stdout.flush()
        print("\\nboom", file=sys.stderr)
        sys.exit(3)
    """)
    lines: list[str] = []
    result = runner.run(argv, tmp_path, lines.append, lambda: False, poll_seconds=0.05)
    assert result.returncode == 3 and not result.cancelled
    assert [line for line in lines if line.startswith("Epoch")] == [
        f"Epoch 1/1: {i}%|x| {i}/3 [00:01<00:02, 1.0s/it, loss=0.5]" for i in range(3)]
    assert result.tail[-1] == "boom"


def test_runner_cancel_kills_the_whole_process_group(tmp_path: Path):
    pid_file = tmp_path / "child.pid"
    argv = script(tmp_path, f"""
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        open({str(pid_file)!r}, "w").write(str(child.pid))
        print("started", flush=True)
        time.sleep(60)
    """)
    started = time.monotonic()
    result = runner.run(argv, tmp_path, lambda line: None, lambda: pid_file.exists(), poll_seconds=0.05,
                        grace_seconds=5)
    assert result.cancelled and time.monotonic() - started < 10
    child = int(pid_file.read_text())
    time.sleep(0.2)
    assert not alive(child)  # DataLoader workers must not survive a cancel


def alive(pid: int) -> bool:
    """Dead includes zombies: without an init process nobody reaps orphans (hence init: true in compose)."""
    try:
        with open(f"/proc/{pid}/status") as fh:
            return not any(line.startswith("State:") and "Z" in line for line in fh)
    except FileNotFoundError:
        return False


# --------------------------------------------------------------------------- recipes
@pytest.fixture
def settings(tmp_path: Path) -> TrainerSettings:
    return TrainerSettings(data_dir=tmp_path / "data", poll_seconds=0.01, gpu_wait_seconds=5, app_stale_seconds=30)


def test_validation_split_always_keeps_one_clip():
    assert recipes.validation_split(10_000) == 0.05
    assert int(4 * recipes.validation_split(4)) == 1
    assert recipes.validation_split(1) == 0.5


def test_fr_ca_plan_downloads_only_when_the_corpus_is_missing(settings: TrainerSettings):
    run = {"id": "abc123abc123", "recipe": "fr_ca", "params": {"output_name": "t3_qc_x"}}
    steps = recipes.plan(run, settings, "cuda")
    assert [s.key for s in steps] == ["download", "prepare", "toolkit", "train", "export", "validate", "publish"]
    assert steps[0].skip() is False
    (settings.qc_data).mkdir(parents=True)
    (settings.qc_data / "metadata.csv").write_text("file_name,transcription\n")
    assert steps[0].skip() is True and steps[1].skip() is True
    validate = next(s for s in steps if s.key == "validate").argv
    assert validate[validate.index("--base") + 1] == "v3" and "--strict-load" in validate
    assert next(s for s in steps if s.key == "train").training


def test_personal_plan_starts_from_the_chosen_model_and_smoke_tests_your_voice(settings: TrainerSettings):
    (settings.own_data / "audio").mkdir(parents=True)
    (settings.own_data / "audio" / "me_0001.wav").write_bytes(b"x")
    (settings.own_data / "holdout.csv").write_text("file_name,transcription\naudio/me_0001.wav,Allo\n")
    run = {"id": "def456def456", "recipe": "personal",
           "params": {"output_name": "t3_me_y", "base_model": "t3_qc_x.safetensors"}}
    steps = recipes.plan(run, settings, "cuda")
    assert [s.key for s in steps][0] == "mix"
    validate = next(s for s in steps if s.key == "validate").argv
    assert validate[validate.index("--base") + 1] == str(settings.models_dir / "t3_qc_x.safetensors")
    assert validate[validate.index("--smoke-reference") + 1] == str(settings.own_data / "audio" / "me_0001.wav")


def test_link_or_copy_is_atomic(tmp_path: Path):
    src = tmp_path / "a.safetensors"
    src.write_bytes(b"weights")
    dst = tmp_path / "models" / "b.safetensors"
    recipes.link_or_copy(src, dst)
    recipes.link_or_copy(src, dst)  # overwrite works too
    assert dst.read_bytes() == b"weights" and not list(dst.parent.glob("*.partial"))


# --------------------------------------------------------------------------- service
class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(seconds, 1.0)


@pytest.fixture
def db(settings: TrainerSettings) -> Database:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return Database(settings.db_path)


def make_trainer(db, settings, clock, gpu=GPU_3090, **kwargs) -> service.Trainer:
    return service.Trainer(db, settings, gpu_probe=lambda: gpu, clock=clock, sleep=clock.sleep, **kwargs)


def fake_plan(steps_for):
    return lambda run, settings, device: steps_for(run, settings, device)


def test_successful_run_records_progress_and_output(db, settings, tmp_path, monkeypatch):
    clock = Clock()
    seen = {}

    def steps(run, s, device):
        seen["device"] = device
        return [
            Step("prep", "Preparing", action=lambda: None),
            Step("train", "Training", training=True, argv=script(tmp_path, """
                import sys
                print("Speech logits shape: torch.Size([2, 3, 4])")
                print("Epoch 1/2:  50%|x| 1/2 [00:01<00:01, 0.50s/it, loss=1.2500, lr=2e-05]")
                print("Epoch 2/2: 100%|x| 2/2 [00:02<00:00, 0.50s/it, loss=0.7500, lr=2e-05]")
                print("Saved checkpoint to somewhere")
            """)),
        ]

    monkeypatch.setattr(service, "plan", fake_plan(steps))
    run = db.create_run("Quebec", "fr_ca", {"epochs": 2, "output_name": "t3_quebec_ab12"})
    make_trainer(db, settings, clock).tick()
    done = db.get_run(run["id"])
    assert done["status"] == "done" and done["output_model"] == "t3_quebec_ab12.safetensors"
    assert (done["step"], done["total_steps"], done["loss"], done["progress"]) == (4, 4, 0.75, 1.0)
    assert seen["device"] == "cuda"
    log = (settings.runs_dir / run["id"] / RUN_LOG_NAME).read_text()
    assert "=== Training ===" in log and "Saved checkpoint" in log and "logits shape" not in log


def test_failed_step_ends_the_run_with_its_output(db, settings, tmp_path, monkeypatch):
    monkeypatch.setattr(service, "plan", fake_plan(lambda r, s, d: [
        Step("train", "Training", argv=script(tmp_path, "import sys\nprint('CUDA out of memory')\nsys.exit(1)"))]))
    run = db.create_run("x", "fr_ca", {"output_name": "t3_x"})
    make_trainer(db, settings, Clock()).tick()
    failed = db.get_run(run["id"])
    assert failed["status"] == "failed" and "Training failed (exit 1): CUDA out of memory" in failed["message"]


def test_cancel_during_training(db, settings, tmp_path, monkeypatch):
    def steps(run, s, device):
        return [Step("train", "Training", argv=script(tmp_path, "import time\nprint('go', flush=True)\n"
                                                                 "time.sleep(60)"))]

    monkeypatch.setattr(service, "plan", fake_plan(steps))
    run = db.create_run("x", "fr_ca", {"output_name": "t3_x"})

    def run_and_cancel(argv, cwd, on_line, should_cancel, **kw):
        db.claim_run(run["id"], "running", "cancelling")  # the user presses Cancel
        return runner.run(argv, cwd, on_line, should_cancel, poll_seconds=0.05, grace_seconds=5)

    make_trainer(db, settings, Clock(), run_command=run_and_cancel).tick()
    assert db.get_run(run["id"])["status"] == "cancelled"


def test_waits_until_the_app_unloads_its_model(db, settings, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(service, "plan", fake_plan(lambda r, s, d: [Step("a", "A", action=lambda: None)]))
    db.set_kv(APP_HEARTBEAT_KEY, {"t": clock.now, "engine_loaded": True})
    run = db.create_run("x", "fr_ca", {"output_name": "t3_x"})
    waits = []

    def sleep(seconds):
        waits.append(db.get_run(run["id"])["message"])
        clock.now += 1
        if len(waits) == 3:  # the app finishes its generation and unloads
            db.set_kv(APP_HEARTBEAT_KEY, {"t": clock.now, "engine_loaded": False})

    trainer = service.Trainer(db, settings, gpu_probe=lambda: GPU_3090, clock=clock, sleep=sleep)
    trainer.tick()
    assert db.get_run(run["id"])["status"] == "done"
    assert len(waits) == 3 and "the app still holds its model" in waits[0]


def test_stale_app_heartbeat_means_the_app_is_down(db, settings, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(service, "plan", fake_plan(lambda r, s, d: [Step("a", "A", action=lambda: None)]))
    db.set_kv(APP_HEARTBEAT_KEY, {"t": clock.now - 600, "engine_loaded": True})
    run = db.create_run("x", "fr_ca", {"output_name": "t3_x"})
    make_trainer(db, settings, clock).tick()
    assert db.get_run(run["id"])["status"] == "done"


def test_gpu_that_never_frees_fails_after_the_timeout(db, settings, monkeypatch):
    monkeypatch.setattr(service, "plan", fake_plan(lambda r, s, d: []))
    run = db.create_run("x", "fr_ca", {"output_name": "t3_x"})
    busy = {**GPU_3090, "free_gb": 6.0}
    make_trainer(db, settings, Clock(), gpu=busy).tick()
    failed = db.get_run(run["id"])
    assert failed["status"] == "failed" and "6.0 GB free, 16 GB needed" in failed["message"]


def test_no_gpu_fails_unless_cpu_is_allowed(db, settings, monkeypatch):
    monkeypatch.setattr(service, "plan", fake_plan(lambda r, s, d: [Step("a", "A", action=lambda: None)]))
    run = db.create_run("x", "fr_ca", {"output_name": "t3_x"})
    make_trainer(db, settings, Clock(), gpu=None).tick()
    assert "No NVIDIA GPU" in db.get_run(run["id"])["message"]
    settings.allow_cpu = True
    run = db.create_run("y", "fr_ca", {"output_name": "t3_y"})
    make_trainer(db, settings, Clock(), gpu=None).tick()
    assert db.get_run(run["id"])["status"] == "done"


def test_personal_run_needs_its_base_model_and_recordings(db, settings):
    run = db.create_run("me", "personal", {"output_name": "t3_me", "base_model": "t3_missing.safetensors"})
    make_trainer(db, settings, Clock()).tick()
    assert "Base model not found" in db.get_run(run["id"])["message"]


def test_restart_fails_interrupted_runs_and_cancels_orphans(db, settings):
    running = db.create_run("a", "fr_ca", {})
    db.claim_run(running["id"], "queued", "running")
    trainer = make_trainer(db, settings, Clock())
    trainer.recover()
    assert "cannot resume" in db.get_run(running["id"])["message"]
    orphan = db.create_run("b", "fr_ca", {})
    db.claim_run(orphan["id"], "queued", "cancelling")
    trainer.tick()
    assert db.get_run(orphan["id"])["status"] == "cancelled"


def test_heartbeat_reports_the_gpu(db, settings):
    make_trainer(db, settings, Clock()).heartbeat()
    beat = db.get_kv("trainer_heartbeat")
    assert beat["gpu"]["name"].endswith("3090") and beat["busy"] is None
