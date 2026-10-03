"""FastAPI application: HTTP API + static single-page front-end.

Run with ``uvicorn app.main:create_app --factory``.
"""

from __future__ import annotations

import logging
import re
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.audio import TARGET_SR, AudioError, convert_to_wav
from app.auth import Auth
from app.config import Settings
from app.db import Database, new_id
from app.engines import create_engine
from app.models import ActiveModelRequest, GenerateRequest, LoginRequest, TrainingRunRequest, VoiceUpdateRequest
from app.training import ACTIVE_MODEL_KEY, RECIPE_DEFAULTS, RUN_METRICS_NAME, TRAINER_HEARTBEAT_KEY, read_log_tail
from app.worker import Worker

log = logging.getLogger("voiceclone")

STATIC_DIR = Path(__file__).parent / "static"
SESSION_COOKIE = "vc_session"
PUBLIC_API_PATHS = {"/api/config", "/api/login", "/api/health"}
ALLOWED_UPLOAD_SUFFIXES = {
    ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".flac", ".webm", ".mp4", ".mkv", ".mov", ".wma", ".aiff",
}


def _slug(value: str, fallback: str = "voice") -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return slug[:40] or fallback


OFFICIAL_T3 = ("v3", "v2")
MODEL_FILE_RE = re.compile(r"^[A-Za-z0-9_.-]+\.safetensors$")
TRAINER_STALE_SECONDS = 45


def _csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as fh:
        return max(0, sum(1 for line in fh if line.strip()) - 1)


def _client_ip(request: Request) -> str:
    for header in ("cf-connecting-ip", "x-forwarded-for"):
        value = request.headers.get(header)
        if value:
            return value.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _is_https(request: Request) -> bool:
    forwarded = request.headers.get("x-forwarded-proto", "")
    return request.url.scheme == "https" or forwarded.split(",")[0].strip() == "https"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.ensure_dirs()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    db = Database(settings.db_path)
    engine = create_engine(settings)
    auth = Auth(settings.app_password, settings.secret_key, settings.session_hours)
    worker = Worker(db, engine, settings)

    def model_path(name: str) -> Path | None:
        """None for an official checkpoint; the file for a fine-tuned one (validated name)."""
        if name in OFFICIAL_T3:
            return None
        if not MODEL_FILE_RE.match(name):
            raise HTTPException(400, f"Invalid model name '{name}'")
        path = settings.models_dir / name
        if not path.is_file():
            raise HTTPException(404, f"Model '{name}' not found in data/models")
        return path

    def restore_active_model() -> None:
        saved = (db.get_kv(ACTIVE_MODEL_KEY) or {}).get("model")
        if not saved:
            return
        try:
            engine.set_t3(saved, model_path(saved))
            log.info("Active T3 checkpoint restored: %s", saved)
        except (HTTPException, NotImplementedError, ValueError, FileNotFoundError) as exc:
            detail = exc.detail if isinstance(exc, HTTPException) else exc
            log.warning("Ignoring saved active model %s: %s", saved, detail)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        restore_active_model()
        if not auth.enabled:
            log.warning("APP_PASSWORD is not set: the API is unauthenticated. Do NOT expose this to the internet.")
        log.info("Engine: %s (%s), data dir: %s", engine.name, engine.variant, settings.data_dir.resolve())
        worker.start(preload=settings.preload_model)
        yield
        worker.stop()

    app = FastAPI(
        title="Voice Clone App",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings
    app.state.db = db
    app.state.engine = engine
    app.state.worker = worker
    app.state.auth = auth

    # Auth ------------------------------------------------------------------
    def is_authenticated(request: Request) -> bool:
        if not auth.enabled:
            return True
        return auth.verify_token(request.cookies.get(SESSION_COOKIE))

    @app.middleware("http")
    async def require_auth(request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/") and path not in PUBLIC_API_PATHS and not is_authenticated(request):
            return JSONResponse({"detail": "Authentication required"}, status_code=401)
        return await call_next(request)

    @app.post("/api/login")
    def login(body: LoginRequest, request: Request, response: Response):
        if not auth.enabled:
            return {"authenticated": True, "auth_required": False}
        key = _client_ip(request)
        wait = auth.retry_after(key)
        if wait > 0:
            raise HTTPException(429, f"Too many failed attempts. Try again in {int(wait) + 1} seconds.",
                                headers={"Retry-After": str(int(wait) + 1)})
        if not auth.check_password(body.password):
            auth.record_failure(key)
            log.warning("Failed login from %s", key)
            raise HTTPException(401, "Wrong password")
        auth.reset(key)
        response.set_cookie(
            SESSION_COOKIE,
            auth.issue_token(),
            max_age=auth.session_seconds,
            httponly=True,
            samesite="lax",
            secure=_is_https(request),
            path="/",
        )
        return {"authenticated": True, "auth_required": True}

    @app.post("/api/logout")
    def logout(response: Response):
        response.delete_cookie(SESSION_COOKIE, path="/")
        return {"authenticated": False, "auth_required": auth.enabled}

    # Meta ------------------------------------------------------------------
    @app.get("/api/config")
    def config(request: Request):
        return {
            "version": __version__,
            "auth_required": auth.enabled,
            "authenticated": is_authenticated(request),
            "engine": engine.info(),
            "limits": {
                "max_text_chars": settings.max_text_chars,
                "max_chunk_chars": settings.max_chunk_chars,
                "max_upload_mb": settings.max_upload_mb,
                "max_reference_seconds": settings.max_reference_seconds,
                "min_reference_seconds": settings.min_reference_seconds,
            },
        }

    @app.get("/api/health")
    def health():
        return {"ok": True, "engine": engine.name, "model_loaded": engine.loaded, "queue_size": worker.queue_size}

    @app.get("/api/status")
    def status():
        data = worker.status()
        run = db.active_run()
        data["training"] = (
            {k: run[k] for k in ("id", "name", "status", "stage", "progress", "eta_seconds")} if run else None
        )
        return data

    # Voices ----------------------------------------------------------------
    @app.get("/api/voices")
    def list_voices():
        return db.list_voices()

    @app.post("/api/voices", status_code=201)
    async def create_voice(
        file: UploadFile = File(...),
        name: str = Form(""),
        language: str = Form(""),
    ):
        original = file.filename or "recording"
        suffix = Path(original).suffix.lower()
        content_type = (file.content_type or "").lower()
        if suffix not in ALLOWED_UPLOAD_SUFFIXES and not content_type.startswith(("audio/", "video/")):
            raise HTTPException(400, f"Unsupported file type '{suffix or content_type or 'unknown'}'")
        if not suffix:
            suffix = ".webm" if "webm" in content_type else ".bin"

        clean_name = name.strip()[:120] or Path(original).stem[:120] or "Voice"
        lang = language.strip().lower() or None
        if lang and engine.multilingual and lang not in engine.languages:
            raise HTTPException(400, f"Unsupported language '{lang}'")

        voice_id = new_id()
        tmp_path = settings.uploads_dir / f"{voice_id}{suffix}"
        dst = settings.voices_dir / f"{voice_id}.wav"
        limit = settings.max_upload_mb * 1024 * 1024
        written = 0
        try:
            with tmp_path.open("wb") as fh:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > limit:
                        raise HTTPException(413, f"File larger than {settings.max_upload_mb} MB")
                    fh.write(chunk)
            if written == 0:
                raise HTTPException(400, "Empty upload")
            try:
                seconds = await run_in_threadpool(
                    convert_to_wav, tmp_path, dst, TARGET_SR, settings.max_reference_seconds
                )
            except AudioError as exc:
                raise HTTPException(400, str(exc)) from exc
            if seconds < settings.min_reference_seconds:
                raise HTTPException(400, f"Reference clip is too short ({seconds:.1f}s); record at least "
                                         f"{settings.min_reference_seconds:g}s, ideally 5-15s of clean speech.")
        except Exception:
            dst.unlink(missing_ok=True)
            raise
        finally:
            tmp_path.unlink(missing_ok=True)

        voice = db.create_voice(clean_name, lang, dst.name, seconds, voice_id=voice_id)
        log.info("Voice '%s' created (%s, %.1fs)", voice["name"], voice["id"], seconds)
        return voice

    @app.get("/api/voices/{voice_id}")
    def get_voice(voice_id: str):
        voice = db.get_voice(voice_id)
        if not voice:
            raise HTTPException(404, "Voice not found")
        return voice

    @app.patch("/api/voices/{voice_id}")
    def update_voice(voice_id: str, body: VoiceUpdateRequest):
        if not db.get_voice(voice_id):
            raise HTTPException(404, "Voice not found")
        fields = {}
        if body.name is not None:
            fields["name"] = body.name.strip()[:120] or "Voice"
        if body.language is not None:
            lang = body.language.strip().lower() or None
            if lang and engine.multilingual and lang not in engine.languages:
                raise HTTPException(400, f"Unsupported language '{lang}'")
            fields["language"] = lang
        return db.update_voice(voice_id, **fields)

    @app.get("/api/voices/{voice_id}/audio")
    def voice_audio(voice_id: str):
        voice = db.get_voice(voice_id)
        if not voice:
            raise HTTPException(404, "Voice not found")
        path = settings.voices_dir / voice["filename"]
        if not path.exists():
            raise HTTPException(404, "Reference audio missing")
        return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "private, max-age=3600"})

    @app.delete("/api/voices/{voice_id}", status_code=204)
    def delete_voice(voice_id: str):
        voice = db.get_voice(voice_id)
        if not voice:
            raise HTTPException(404, "Voice not found")
        db.delete_voice(voice_id)
        (settings.voices_dir / voice["filename"]).unlink(missing_ok=True)
        return Response(status_code=204)

    # Generation --------------------------------------------------------------
    @app.post("/api/generate", status_code=202)
    def generate(body: GenerateRequest):
        run = db.active_run()
        if run:
            raise HTTPException(409, f"Training run '{run['name']}' is using the GPU; generation resumes when it "
                                     "finishes (or cancel it in Fine-tuning).")
        voice = db.get_voice(body.voice_id)
        if not voice:
            raise HTTPException(404, "Voice not found")
        text = body.text.strip()
        if not text:
            raise HTTPException(400, "Text is empty")
        if len(text) > settings.max_text_chars:
            raise HTTPException(400, f"Text is too long ({len(text)} chars, max {settings.max_text_chars})")
        language = (body.language or voice.get("language") or "en").lower()
        if engine.multilingual and language not in engine.languages:
            raise HTTPException(400, f"Unsupported language '{language}'")
        if not engine.multilingual:
            language = "en"
        params = {
            "exaggeration": body.exaggeration,
            "cfg_weight": body.cfg_weight,
            "temperature": body.temperature,
            "seed": body.seed,
        }
        job = db.create_job(voice, text, language, params)
        worker.submit(job["id"])
        return job

    @app.get("/api/jobs")
    def list_jobs(limit: int = Query(50, ge=1, le=500)):
        return db.list_jobs(limit)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = db.get_job(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return job

    @app.get("/api/jobs/{job_id}/audio")
    def job_audio(job_id: str, download: bool = False):
        job = db.get_job(job_id)
        if not job or not job["has_audio"]:
            raise HTTPException(404, "No audio for this job")
        path = settings.outputs_dir / job["output_filename"]
        if not path.exists():
            raise HTTPException(404, "Audio file missing")
        filename = f"{_slug(job['voice_name'])}-{job_id}.wav" if download else None
        return FileResponse(path, media_type="audio/wav", filename=filename,
                            headers={"Cache-Control": "private, max-age=3600"})

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str):
        """Cancel an active job, or delete a finished one along with its audio."""
        job = db.get_job(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job["status"] in ("queued",):
            db.update_job(job_id, status="cancelled", finished_at=time.time())
        elif job["status"] == "running":
            db.update_job(job_id, status="cancelling")
        elif job["status"] == "cancelling":
            pass
        else:
            db.delete_job(job_id)
            if job.get("output_filename"):
                (settings.outputs_dir / job["output_filename"]).unlink(missing_ok=True)
        return Response(status_code=204)

    # Models ------------------------------------------------------------------
    def list_models() -> dict:
        custom = []
        if settings.models_dir.is_dir():
            for path in sorted(settings.models_dir.glob("*.safetensors"), key=lambda p: -p.stat().st_mtime):
                stat = path.stat()
                custom.append({"name": path.name, "size_bytes": stat.st_size, "created_at": stat.st_mtime})
        info = engine.info()
        return {"active": info.get("t3_model"), "model_id": info.get("model_id"), "official": list(OFFICIAL_T3),
                "custom": custom, "switchable": engine.multilingual, "error": worker.model_error}

    @app.get("/api/models")
    def get_models():
        return list_models()

    @app.post("/api/models/active")
    def set_active_model(body: ActiveModelRequest):
        if not engine.multilingual:
            raise HTTPException(400, "Only the multilingual engine can switch T3 checkpoints")
        path = model_path(body.model)
        db.set_kv(ACTIVE_MODEL_KEY, {"model": body.model})
        worker.request_model(body.model, path)
        return {"model": body.model, "pending": True}

    @app.delete("/api/models/{name}", status_code=204)
    def delete_model(name: str):
        path = model_path(name)
        if path is None:
            raise HTTPException(400, "Official checkpoints cannot be deleted")
        if engine.info().get("t3_model") == name or (db.get_kv(ACTIVE_MODEL_KEY) or {}).get("model") == name:
            raise HTTPException(409, "This model is active; switch to another one first")
        path.unlink(missing_ok=True)
        return Response(status_code=204)

    # Training ----------------------------------------------------------------
    def run_dir(run_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{12}", run_id):
            raise HTTPException(404, "Run not found")
        return settings.finetune_dir / "runs" / run_id

    def get_run_or_404(run_id: str) -> dict:
        run = db.get_run(run_id) if re.fullmatch(r"[0-9a-f]{12}", run_id) else None
        if not run:
            raise HTTPException(404, "Run not found")
        run["has_metrics"] = (run_dir(run_id) / RUN_METRICS_NAME).is_file()
        return run

    @app.get("/api/training")
    def training_overview():
        beat = db.get_kv(TRAINER_HEARTBEAT_KEY) or {}
        age = time.time() - float(beat.get("t", 0)) if beat else None
        own = settings.finetune_dir / "me" / "audio_data"
        qc = settings.finetune_dir / "qc" / "audio_data"
        runs = db.list_runs(20)
        for run in runs:
            run["has_metrics"] = (run_dir(run["id"]) / RUN_METRICS_NAME).is_file()
        return {
            "trainer": {"online": age is not None and age < TRAINER_STALE_SECONDS, "last_seen_seconds": age,
                        "gpu": beat.get("gpu"), "busy": beat.get("busy"), "allow_cpu": beat.get("allow_cpu")},
            "active": db.active_run(),
            "runs": runs,
            "defaults": RECIPE_DEFAULTS,
            "datasets": {
                "qc": {"ready": (qc / "metadata.csv").is_file(), "clips": _csv_rows(qc / "metadata.csv")},
                "own": {"ready": (own / "metadata.csv").is_file(), "clips": _csv_rows(own / "metadata.csv")},
            },
            "models": list_models()["custom"],
        }

    @app.post("/api/training/runs", status_code=201)
    def create_run(body: TrainingRunRequest):
        active = db.active_run()
        if active:
            raise HTTPException(409, f"Run '{active['name']}' is still {active['status']}; one run at a time")
        params = {**RECIPE_DEFAULTS[body.recipe],
                  **body.model_dump(exclude={"recipe", "name", "base_model"}, exclude_none=True)}
        if body.recipe == "personal":
            if not body.base_model:
                raise HTTPException(400, "Pick the fine-tuned model to start from (base_model)")
            if model_path(body.base_model) is None:
                raise HTTPException(400, "Start from a fine-tuned Quebec model, not an official checkpoint")
            if not (settings.finetune_dir / "me" / "audio_data" / "metadata.csv").is_file():
                raise HTTPException(400, "No recordings of your voice yet: run scripts.finetune.segment_recording "
                                         "into data/finetune/me/audio_data first (see the README)")
            params["base_model"] = body.base_model
        name = body.name.strip() or ("Quebec accent" if body.recipe == "fr_ca" else "My voice")
        params["output_name"] = f"t3_{_slug(name, 'model').replace('-', '_')}_{new_id()[:6]}"
        return db.create_run(name, body.recipe, params)

    @app.get("/api/training/runs/{run_id}")
    def get_run(run_id: str):
        return get_run_or_404(run_id)

    @app.get("/api/training/runs/{run_id}/log")
    def get_run_log(run_id: str, lines: int = Query(200, ge=1, le=2000)):
        get_run_or_404(run_id)
        return {"lines": read_log_tail(run_dir(run_id), lines)}

    @app.get("/api/training/runs/{run_id}/metrics.png")
    def get_run_metrics(run_id: str):
        get_run_or_404(run_id)
        path = run_dir(run_id) / RUN_METRICS_NAME
        if not path.is_file():
            raise HTTPException(404, "No training chart yet")
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-cache"})

    @app.delete("/api/training/runs/{run_id}", status_code=204)
    def delete_run(run_id: str):
        """Cancel an active run, or delete a finished one and its working files (not its model)."""
        run = get_run_or_404(run_id)
        if run["status"] == "queued" and db.claim_run(run_id, "queued", "cancelled"):
            db.update_run(run_id, message="Cancelled before it started", finished_at=time.time())
        elif run["status"] in ("waiting_gpu", "running"):
            db.claim_run(run_id, run["status"], "cancelling")
        elif run["status"] in ("done", "failed", "cancelled"):
            db.delete_run(run_id)
            shutil.rmtree(run_dir(run_id), ignore_errors=True)
        return Response(status_code=204)

    # Front-end ---------------------------------------------------------------
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    return app


__all__ = ["create_app"]
