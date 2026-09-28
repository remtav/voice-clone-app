"""Background generation worker.

One GPU means one job at a time, so a single daemon thread drains a FIFO
queue.  Model loading happens on that same thread (lazily on the first job,
or eagerly at startup when ``PRELOAD_MODEL`` is set).
"""

from __future__ import annotations

import logging
import queue
import threading
import time

from app.audio import concat_with_silence, peak_normalize, write_wav
from app.config import Settings
from app.db import Database
from app.engines.base import Engine, SynthesisParams
from app.text import split_text

log = logging.getLogger(__name__)

_LOAD = object()
_STOP = object()


class Worker:
    def __init__(self, db: Database, engine: Engine, settings: Settings) -> None:
        self.db = db
        self.engine = engine
        self.settings = settings
        self._queue: queue.Queue[object] = queue.Queue()
        self._thread: threading.Thread | None = None
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
        self._queue.put(_STOP)
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def submit(self, job_id: str) -> None:
        self._queue.put(job_id)

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
            }
        )
        return info

    # Internals -------------------------------------------------------------
    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                break
            if item is _LOAD:
                self._ensure_loaded()
                continue
            job = self.db.get_job(str(item))
            if not job or job["status"] != "queued":
                continue
            self._process(job)

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
                )
                pieces.append(self.engine.synthesize(chunk, reference, synth))
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

    def _prune(self) -> None:
        removed = self.db.prune_jobs(self.settings.history_limit)
        for filename in removed:
            try:
                (self.settings.outputs_dir / filename).unlink(missing_ok=True)
            except OSError:
                log.warning("Could not delete pruned output %s", filename)
