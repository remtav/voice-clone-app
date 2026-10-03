"""Training runs, model switching and GPU sharing, through the HTTP API (fake engine)."""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import Database
from app.engines.fake import FakeEngine
from app.main import create_app
from app.training import ACTIVE_MODEL_KEY, APP_HEARTBEAT_KEY, RUN_LOG_NAME, TRAINER_HEARTBEAT_KEY
from app.worker import Worker
from tests.conftest import upload_voice, wait_for_job


def db_of(client: TestClient) -> Database:
    return client.app.state.db


def wait_until(predicate, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition not reached")


def new_run(client: TestClient, **body) -> dict:
    res = client.post("/api/training/runs", json={"recipe": "fr_ca", **body})
    assert res.status_code == 201, res.text
    return res.json()


# --------------------------------------------------------------------------- overview
def test_overview_shows_trainer_status_and_datasets(client: TestClient):
    data = client.get("/api/training").json()
    assert data["trainer"]["online"] is False and data["runs"] == [] and data["active"] is None
    assert data["datasets"]["qc"]["ready"] is False and data["defaults"]["fr_ca"]["epochs"] == 4
    gpu = {"name": "NVIDIA GeForce RTX 3090", "total_gb": 24.0, "free_gb": 23.5}
    db_of(client).set_kv(TRAINER_HEARTBEAT_KEY, {"t": time.time(), "gpu": gpu, "busy": None})
    trainer = client.get("/api/training").json()["trainer"]
    assert trainer["online"] is True and trainer["gpu"]["total_gb"] == 24.0


# --------------------------------------------------------------------------- runs
def test_create_run_merges_defaults_and_names_the_output(client: TestClient):
    run = new_run(client, name="Québec v1", epochs=3, grad_accum=1)
    assert run["status"] == "queued" and run["name"] == "Québec v1"
    params = run["params"]
    assert (params["epochs"], params["grad_accum"], params["lora_rank"], params["learning_rate"]) == (3, 1, 16, 2e-5)
    assert re.fullmatch(r"t3_qu_bec_v1_[0-9a-f]{6}", params["output_name"])  # safe file stem, unique suffix
    assert client.post("/api/training/runs", json={"recipe": "fr_ca"}).status_code == 409  # one at a time


@pytest.mark.parametrize("body, status, message", [
    ({"recipe": "personal"}, 400, "base_model"),
    ({"recipe": "personal", "base_model": "v3"}, 400, "not an official"),
    ({"recipe": "personal", "base_model": "t3_missing.safetensors"}, 404, "not found"),
    ({"recipe": "personal", "base_model": "../../etc/passwd.safetensors"}, 400, "Invalid model name"),
    ({"recipe": "fr_ca", "lora_rank": 12}, 422, ""),
])
def test_create_run_validation(client: TestClient, body, status, message):
    res = client.post("/api/training/runs", json=body)
    assert res.status_code == status and message in res.text


def test_personal_run_needs_recordings(client: TestClient, settings: Settings):
    settings.models_dir.mkdir(parents=True)
    (settings.models_dir / "t3_qc.safetensors").write_bytes(b"x")
    res = client.post("/api/training/runs", json={"recipe": "personal", "base_model": "t3_qc.safetensors"})
    assert res.status_code == 400 and "segment_recording" in res.text
    own = settings.finetune_dir / "me" / "audio_data"
    own.mkdir(parents=True)
    (own / "metadata.csv").write_text("file_name,transcription\naudio/a.wav,Allo\n")
    run = new_run(client, recipe="personal", base_model="t3_qc.safetensors")
    assert run["params"]["base_model"] == "t3_qc.safetensors" and run["params"]["learning_rate"] == 1e-5


def test_cancel_and_delete_follow_the_run_state(client: TestClient, settings: Settings):
    db = db_of(client)
    queued = new_run(client)
    assert client.delete(f"/api/training/runs/{queued['id']}").status_code == 204
    assert db.get_run(queued["id"])["status"] == "cancelled"

    running = new_run(client)
    db.claim_run(running["id"], "queued", "running")
    client.delete(f"/api/training/runs/{running['id']}")
    assert db.get_run(running["id"])["status"] == "cancelling"  # the trainer stops it
    db.update_run(running["id"], status="done")
    run_dir = settings.finetune_dir / "runs" / running["id"]
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint.pt").write_bytes(b"x")
    assert client.delete(f"/api/training/runs/{running['id']}").status_code == 204
    assert db.get_run(running["id"]) is None and not run_dir.exists()
    assert client.delete("/api/training/runs/zzz").status_code == 404


def test_log_and_metrics(client: TestClient, settings: Settings):
    run = new_run(client)
    assert client.get(f"/api/training/runs/{run['id']}/log").json() == {"lines": []}
    run_dir = settings.finetune_dir / "runs" / run["id"]
    run_dir.mkdir(parents=True)
    (run_dir / RUN_LOG_NAME).write_text("\n".join(f"line {i}" for i in range(500)))
    assert client.get(f"/api/training/runs/{run['id']}/log?lines=3").json()["lines"] == ["line 497", "line 498",
                                                                                         "line 499"]
    assert client.get(f"/api/training/runs/{run['id']}/metrics.png").status_code == 404
    (run_dir / "training_metrics.png").write_bytes(b"\x89PNG")
    assert client.get(f"/api/training/runs/{run['id']}/metrics.png").status_code == 200
    assert client.get(f"/api/training/runs/{run['id']}").json()["has_metrics"] is True


# --------------------------------------------------------------------------- GPU sharing
def test_generation_is_refused_while_training(client: TestClient, wav_file: Path):
    voice = upload_voice(client, wav_file)
    run = new_run(client, name="Nuit")
    res = client.post("/api/generate", json={"voice_id": voice["id"], "text": "Allo toi."})
    assert res.status_code == 409 and "Nuit" in res.json()["detail"]
    status = client.get("/api/status").json()
    assert status["training"]["id"] == run["id"]
    client.delete(f"/api/training/runs/{run['id']}")
    assert client.post("/api/generate", json={"voice_id": voice["id"], "text": "Allo toi."}).status_code == 202


def test_worker_releases_the_gpu_and_holds_jobs_until_training_ends(settings: Settings, wav_file: Path):
    settings.ensure_dirs()
    db = Database(settings.db_path)
    engine = FakeEngine()
    worker = Worker(db, engine, settings, poll_seconds=0.05)
    worker.start(preload=True)
    try:
        wait_until(lambda: engine.loaded)
        voice = db.create_voice("v", "fr", "v.wav", 2.0)
        (settings.voices_dir / "v.wav").write_bytes(wav_file.read_bytes())
        run = db.create_run("x", "fr_ca", {})
        wait_until(lambda: not engine.loaded)  # model unloaded for the trainer
        wait_until(lambda: (db.get_kv(APP_HEARTBEAT_KEY) or {}).get("engine_loaded") is False)
        job = db.create_job(voice, "Allo.", "fr", {})
        worker.submit(job["id"])
        time.sleep(0.4)
        assert db.get_job(job["id"])["status"] == "queued" and worker.status()["paused_for_training"]
        db.update_run(run["id"], status="done")  # training finished
        wait_until(lambda: db.get_job(job["id"])["status"] == "done")
        assert engine.loaded  # reloaded for the held job
    finally:
        worker.stop()


# --------------------------------------------------------------------------- models
def test_models_list_activate_and_delete(client: TestClient, settings: Settings):
    settings.models_dir.mkdir(parents=True)
    for name in ("t3_a.safetensors", "t3_b.safetensors"):
        (settings.models_dir / name).write_bytes(b"x")
    data = client.get("/api/models").json()
    assert data["active"] == "v3" and {m["name"] for m in data["custom"]} == {"t3_a.safetensors", "t3_b.safetensors"}

    assert client.post("/api/models/active", json={"model": "t3_a.safetensors"}).status_code == 200
    wait_until(lambda: client.get("/api/models").json()["active"] == "t3_a.safetensors")
    assert db_of(client).get_kv(ACTIVE_MODEL_KEY) == {"model": "t3_a.safetensors"}
    assert client.delete("/api/models/t3_a.safetensors").status_code == 409
    assert client.delete("/api/models/t3_b.safetensors").status_code == 204
    assert not (settings.models_dir / "t3_b.safetensors").exists()
    assert client.post("/api/models/active", json={"model": "../x.safetensors"}).status_code == 400
    assert client.post("/api/models/active", json={"model": "t3_zz.safetensors"}).status_code == 404
    assert client.delete("/api/models/v3").status_code == 400
    client.post("/api/models/active", json={"model": "v3"})
    wait_until(lambda: client.get("/api/models").json()["active"] == "v3")


def test_active_model_is_restored_at_startup(settings: Settings):
    settings.ensure_dirs()
    settings.models_dir.mkdir(parents=True)
    (settings.models_dir / "t3_saved.safetensors").write_bytes(b"x")
    Database(settings.db_path).set_kv(ACTIVE_MODEL_KEY, {"model": "t3_saved.safetensors"})
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/models").json()["active"] == "t3_saved.safetensors"
    (settings.models_dir / "t3_saved.safetensors").unlink()  # deleted behind the app's back
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/models").json()["active"] == "v3"


def test_jobs_still_complete_normally(client: TestClient, wav_file: Path):
    voice = upload_voice(client, wav_file)
    job = client.post("/api/generate", json={"voice_id": voice["id"], "text": "Bonjour."}).json()
    assert wait_for_job(client, job["id"])["status"] == "done"
