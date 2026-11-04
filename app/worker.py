"""Background generation worker.

One GPU means one job at a time, so a single daemon thread drains a FIFO
queue.  Model loading happens on that same thread (lazily on the first job,
or eagerly at startup when ``PRELOAD_MODEL`` is set).

The GPU is shared with the trainer service, one at a time.  While a training
run is active the worker unloads the model as soon as no job is running,
holds queued jobs, and publishes a heartbeat (``app_heartbeat`` in the kv
table) saying whether it still holds the model; the trainer waits for it.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from app import respeak
from app.audio import concat_with_silence, peak_normalize, write_wav
from app.config import Settings
from app.db import Database
from app.engines.base import Engine, SynthesisParams
from app.prompt_clip import prepare_prompt_clip
from app.stt import Transcriber
from app.text import split_text
from app.training import APP_HEARTBEAT_KEY

log = logging.getLogger(__name__)

_LOAD = object()
_STOP = object()


@dataclass(frozen=True)
class SetModel:
    """Control message: switch the T3 checkpoint (applied on the worker thread)."""

    t3_model: str
    t3_path: Path | None


class Worker:
    def __init__(self, db: Database, engine: Engine, settings: Settings, poll_seconds: float = 2.0,
                 transcriber: Transcriber | None = None) -> None:
        self.db = db
        self.engine = engine
        self.transcriber = transcriber
        self.settings = settings
        self.poll_seconds = poll_seconds
        self._queue: queue.Queue[object] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._last_beat = 0.0
        self.paused_for_training = False
        self.model_error: str | None = None
        self._load_lock = threading.Lock()
        self.model_loading = False
        self.load_error: str | None = None
        self.load_seconds: float | None = None
        self.current_job_id: str | None = None
        self.completed = 0
        self.failed = 0

    # Lifecycle -------------------------------------------------------------
    def start(self, preload: bool = False) -> None:
        requeued = self.db.requeue_stale()
        if requeued:
            log.info("Re-queued %d job(s) interrupted by a previous shutdown", requeued)
        if preload:
            self._queue.put(_LOAD)
        for job_id in self.db.queued_job_ids():
            self._queue.put(job_id)
        self._thread = threading.Thread(target=self._run, name="tts-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stopping.set()
        self._queue.put(_STOP)
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def submit(self, job_id: str) -> None:
        self._queue.put(job_id)

    def request_model(self, t3_model: str, t3_path: Path | None) -> None:
        self._queue.put(SetModel(t3_model, t3_path))

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def status(self) -> dict:
        info = self.engine.info()
        info.update(
            {
                "model_loading": self.model_loading,
                "load_error": self.load_error,
                "load_seconds": self.load_seconds,
                "queue_size": self.queue_size,
                "current_job_id": self.current_job_id,
                "completed": self.completed,
                "failed": self.failed,
                "paused_for_training": self.paused_for_training,
                "model_error": self.model_error,
            }
        )
        return info

    # Internals -------------------------------------------------------------
    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=self.poll_seconds)
            except queue.Empty:
                self._housekeeping()
                continue
            if item is _STOP:
                break
            if isinstance(item, SetModel):
                self._apply_model(item)
                continue
            if not self._wait_while_training():
                break
            if item is _LOAD:
                self._ensure_loaded()
                continue
            job = self.db.get_job(str(item))
            if not job or job["status"] != "queued":
                continue
            self._process(job)
            self._housekeeping()

    # GPU sharing with the trainer -----------------------------------------
    def _housekeeping(self, force_beat: bool = False) -> None:
        """Release the GPU for an active training run and publish the heartbeat."""
        training = self.db.active_run() is not None
        if training and self.engine.loaded and self.current_job_id is None:
            log.info("Training run active: unloading the model to free the GPU")
            self.engine.unload()
        if training and self.transcriber is not None and self.transcriber.loaded:
            self.transcriber.unload()
        self.paused_for_training = training
        now = time.time()
        if force_beat or training or now - self._last_beat >= 5 * self.poll_seconds:
            self.db.set_kv(APP_HEARTBEAT_KEY, {"t": now, "engine_loaded": self.engine.loaded})
            self._last_beat = now

    def _wait_while_training(self) -> bool:
        """Hold queued work while a run owns the GPU; False if the worker is stopping."""
        while self.db.active_run() is not None:
            self._housekeeping()
            if self._stopping.wait(self.poll_seconds):
                return False
        self.paused_for_training = False
        return not self._stopping.is_set()

    def _apply_model(self, request: SetModel) -> None:
        try:
            self.engine.set_t3(request.t3_model, request.t3_path)
            self.model_error = None
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI, keep the old model
            log.exception("Could not switch the T3 checkpoint")
            self.model_error = f"{type(exc).__name__}: {exc}"

    def _ensure_loaded(self) -> bool:
        if self.engine.loaded:
            return True
        with self._load_lock:
            if self.engine.loaded:
                return True
            self.model_loading = True
            self.load_error = None
            started = time.time()
            try:
                self.engine.load()
                self.load_seconds = round(time.time() - started, 1)
                log.info("Engine '%s' ready in %.1fs", self.engine.name, self.load_seconds)
                return True
            except Exception as exc:  # noqa: BLE001 - surface any load failure to the UI
                log.exception("Failed to load engine '%s'", self.engine.name)
                self.load_error = f"{type(exc).__name__}: {exc}"
                return False
            finally:
                self.model_loading = False

    def _process(self, job: dict) -> None:
        job_id = job["id"]
        self.current_job_id = job_id
        started = time.time()
        try:
            if not self._ensure_loaded():
                raise RuntimeError(f"Model failed to load: {self.load_error}")

            voice = self.db.get_voice(job["voice_id"])
            if not voice:
                raise RuntimeError("The voice used by this job no longer exists")
            reference = self.settings.voices_dir / voice["filename"]
            if not reference.exists():
                raise RuntimeError("Reference audio file is missing on disk")

            chunks = split_text(job["text"], self.settings.max_chunk_chars)
            if not chunks:
                raise RuntimeError("Nothing to synthesise: the text is empty")

            self.db.update_job(
                job_id, status="running", started_at=started, progress_total=len(chunks), progress_done=0
            )
            log.info(
                "Job %s: %d chunk(s), voice '%s', language %s", job_id, len(chunks), voice["name"], job["language"]
            )

            reference_text = voice.get("transcript")
            if self.engine.uses_reference_text:
                # A short, sentence-aligned part of the voice: a long reference gets re-spoken.
                prompt = prepare_prompt_clip(reference, reference_text, self.settings.prompt_reference_seconds,
                                             self.transcriber, voice.get("language"))
                if prompt is not None:
                    reference, reference_text = prompt.path, prompt.text

            params = job.get("params") or {}
            base_seed = params.get("seed")
            pieces = []
            for index, chunk in enumerate(chunks):
                current = self.db.get_job(job_id)
                if current is None:
                    log.info("Job %s deleted mid-flight; stopping", job_id)
                    return
                if current["status"] == "cancelling":
                    self.db.update_job(job_id, status="cancelled", finished_at=time.time())
                    log.info("Job %s cancelled after %d/%d chunk(s)", job_id, index, len(chunks))
                    return
                synth = SynthesisParams(
                    language=job.get("language") or "en",
                    exaggeration=float(params.get("exaggeration", 0.5)),
                    cfg_weight=float(params.get("cfg_weight", 0.5)),
                    temperature=float(params.get("temperature", 0.8)),
                    seed=(int(base_seed) + index) if base_seed is not None else None,
                    reference_text=reference_text,
                )
                pieces.append(self._synthesize_checked(chunk, reference, synth, voice.get("transcript")))
                self.db.update_job(job_id, progress_done=index + 1)

            sr = self.engine.sample_rate
            audio = peak_normalize(concat_with_silence(pieces, sr))
            output_name = f"{job_id}.wav"
            write_wav(self.settings.outputs_dir / output_name, audio, sr)
            seconds = len(audio) / sr
            self.db.update_job(
                job_id,
                status="done",
                output_filename=output_name,
                duration=seconds,
                finished_at=time.time(),
            )
            self.completed += 1
            log.info("Job %s done: %.1fs of audio in %.1fs", job_id, seconds, time.time() - started)
            self._prune()
        except Exception as exc:  # noqa: BLE001 - any failure marks the job as failed
            log.exception("Job %s failed", job_id)
            self.db.update_job(job_id, status="failed", error=str(exc)[:1000], finished_at=time.time())
            self.failed += 1
        finally:
            self.current_job_id = None

    def _heard(self, audio: np.ndarray, language: str | None):
        """Speech-to-text (with word times) of generated audio; None when no transcriber can run."""
        if self.transcriber is None or not self.transcriber.available:
            return None
        path = self.settings.uploads_dir / f"check-{uuid.uuid4().hex}.wav"
        try:
            write_wav(path, audio, self.engine.sample_rate)
            return self.transcriber.transcribe(path, language, words=True)
        except Exception:  # noqa: BLE001 - the check is best-effort
            log.exception("Could not check a generated chunk")
            return None
        finally:
            path.unlink(missing_ok=True)

    def _trim_reference_tail(self, audio: np.ndarray, heard, reference_text: str, chunk: str) -> np.ndarray:
        """Cut off the end of the reference clip when the model opens with it."""
        words = heard.words or []
        spill = respeak.leading_spill([w.text for w in words], reference_text, chunk)
        if not spill or spill >= len(words):
            return audio
        end = words[spill - 1].end
        cut = (end + words[spill].start) / 2 if words[spill].start > end else words[spill].start
        log.warning("Chunk opened with the end of the reference (%r): trimming %.2fs",
                    " ".join(w.text for w in words[:spill]), cut)
        return audio[int(cut * self.engine.sample_rate):]

    def _synthesize_checked(self, chunk: str, reference: Path, synth: SynthesisParams,
                            full_transcript: str | None) -> np.ndarray:
        """Synthesise ``chunk`` and clean it up with speech-to-text.

        An opening that repeats the end of the reference is trimmed; a chunk that re-speaks
        the reference is redone with a new seed.
        """
        checked = self.engine.uses_reference_text and bool(full_transcript)
        retries = self.settings.respeak_retries if checked else 0
        sr = self.engine.sample_rate
        best: tuple[float, np.ndarray] | None = None
        for attempt in range(retries + 1):
            attempt_synth = synth
            if attempt and synth.seed is not None:  # unseeded generations already differ on each try
                attempt_synth = replace(synth, seed=synth.seed + 1000 * attempt)
            audio = self.engine.synthesize(chunk, reference, attempt_synth)
            if not checked or not self.settings.respeak_retries:
                return audio
            heard = self._heard(audio, synth.language)
            if heard is None:
                leaked = respeak.duration_certain(chunk, len(audio) / sr)
            else:
                audio = self._trim_reference_tail(audio, heard, synth.reference_text or full_transcript, chunk)
                leaked = respeak.leaks_reference(heard.text, full_transcript, chunk)
            seconds = len(audio) / sr
            if not leaked:
                return audio
            log.warning("Chunk re-spoke the reference (%.1fs for %d characters); attempt %d/%d",
                        seconds, len(chunk), attempt + 1, retries + 1)
            if best is None or seconds < best[0]:
                best = (seconds, audio)
        log.warning("Keeping the shortest of %d attempts for a chunk that kept re-speaking the reference",
                    retries + 1)
        assert best is not None
        return best[1]

    def _prune(self) -> None:
        removed = self.db.prune_jobs(self.settings.history_limit)
        for filename in removed:
            try:
                (self.settings.outputs_dir / filename).unlink(missing_ok=True)
            except OSError:
                log.warning("Could not delete pruned output %s", filename)
