from __future__ import annotations

from pathlib import Path

from tests.conftest import make_wav, upload_voice, wait_for_job


def test_config_and_health(client):
    config = client.get("/api/config").json()
    assert config["auth_required"] is False
    assert config["authenticated"] is True
    assert config["engine"]["name"] == "fake"
    assert "en" in config["engine"]["languages"]
    assert config["limits"]["max_text_chars"] == 600

    health = client.get("/api/health").json()
    assert health["ok"] is True

    status = client.get("/api/status").json()
    assert status["name"] == "fake"
    assert status["load_error"] is None


def test_index_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    assert "Voice Clone" in res.text


def test_voice_lifecycle(client, wav_file, settings):
    voice = upload_voice(client, wav_file, name="Narrator", language="fr")
    assert voice["name"] == "Narrator"
    assert voice["language"] == "fr"
    assert 1.9 < voice["duration"] < 2.1
    assert (settings.voices_dir / voice["filename"]).exists()

    voices = client.get("/api/voices").json()
    assert [v["id"] for v in voices] == [voice["id"]]

    audio = client.get(f"/api/voices/{voice['id']}/audio")
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"
    assert audio.content[:4] == b"RIFF"

    renamed = client.patch(f"/api/voices/{voice['id']}", json={"name": "Story teller", "language": "de"}).json()
    assert renamed["name"] == "Story teller"
    assert renamed["language"] == "de"

    assert client.delete(f"/api/voices/{voice['id']}").status_code == 204
    assert client.get(f"/api/voices/{voice['id']}").status_code == 404
    assert not (settings.voices_dir / voice["filename"]).exists()
    assert client.delete(f"/api/voices/{voice['id']}").status_code == 404


def test_voice_defaults_name_from_filename(client, tmp_path):
    path = make_wav(tmp_path / "My Recording.wav", seconds=1.5, sr=44100, channels=2)
    with path.open("rb") as fh:
        res = client.post("/api/voices", files={"file": (path.name, fh, "audio/wav")})
    assert res.status_code == 201, res.text
    voice = res.json()
    assert voice["name"] == "My Recording"
    assert voice["language"] is None
    assert 1.4 < voice["duration"] < 1.6


def test_voice_rejects_short_and_invalid_files(client, tmp_path):
    short = make_wav(tmp_path / "short.wav", seconds=0.3)
    with short.open("rb") as fh:
        res = client.post("/api/voices", files={"file": ("short.wav", fh, "audio/wav")})
    assert res.status_code == 400
    assert "too short" in res.json()["detail"]

    res = client.post("/api/voices", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 400

    res = client.post("/api/voices", files={"file": ("broken.wav", b"not really audio", "audio/wav")})
    assert res.status_code == 400

    with (tmp_path / "ref.wav").open("wb") as fh:
        pass
    res = client.post("/api/voices", files={"file": ("empty.wav", b"", "audio/wav")})
    assert res.status_code == 400

    res = client.post("/api/voices", files={"file": ("x.wav", b"RIFF", "audio/wav")}, data={"language": "xx"})
    assert res.status_code == 400
    assert client.get("/api/voices").json() == []


def test_generate_flow(client, wav_file, settings):
    voice = upload_voice(client, wav_file, language="en")
    text = (
        "This is the first sentence. Here is a second one that is a bit longer than the first.\n\n"
        "A new paragraph starts here! And it keeps going for a while, with commas, and more words."
    )
    res = client.post(
        "/api/generate",
        json={"voice_id": voice["id"], "text": text, "language": "fr", "exaggeration": 0.7, "seed": 42},
    )
    assert res.status_code == 202, res.text
    job = res.json()
    assert job["status"] == "queued"
    assert job["voice_name"] == voice["name"]
    assert job["language"] == "fr"
    assert job["params"] == {"exaggeration": 0.7, "cfg_weight": 0.5, "temperature": 0.8, "seed": 42}

    done = wait_for_job(client, job["id"])
    assert done["status"] == "done", done
    assert done["has_audio"] is True
    assert done["progress_total"] >= 3
    assert done["progress_done"] == done["progress_total"]
    assert done["duration"] > 0.5
    assert done["started_at"] is not None and done["finished_at"] >= done["started_at"]

    audio = client.get(f"/api/jobs/{job['id']}/audio")
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"
    assert audio.content[:4] == b"RIFF"
    assert "content-disposition" not in audio.headers

    download = client.get(f"/api/jobs/{job['id']}/audio?download=1")
    assert download.status_code == 200
    assert "attachment" in download.headers["content-disposition"]
    assert job["id"] in download.headers["content-disposition"]

    listed = client.get("/api/jobs").json()
    assert [j["id"] for j in listed] == [job["id"]]

    # Same seed => identical output from the deterministic fake engine.
    again = client.post("/api/generate", json={"voice_id": voice["id"], "text": text, "seed": 42}).json()
    wait_for_job(client, again["id"])
    assert client.get(f"/api/jobs/{again['id']}/audio").content != audio.content  # exaggeration differs
    third = client.post(
        "/api/generate", json={"voice_id": voice["id"], "text": text, "language": "fr", "exaggeration": 0.7, "seed": 42}
    ).json()
    wait_for_job(client, third["id"])
    assert client.get(f"/api/jobs/{third['id']}/audio").content == audio.content

    output_path = settings.outputs_dir / done["output_filename"]
    assert output_path.exists()
    assert client.delete(f"/api/jobs/{job['id']}").status_code == 204
    assert not output_path.exists()
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404
    assert client.get(f"/api/jobs/{job['id']}/audio").status_code == 404


def test_generate_validation(client, wav_file):
    voice = upload_voice(client, wav_file)
    base = {"voice_id": voice["id"], "text": "Hello there."}

    assert client.post("/api/generate", json={**base, "voice_id": "nope"}).status_code == 404
    assert client.post("/api/generate", json={**base, "text": "   "}).status_code == 400
    assert client.post("/api/generate", json={**base, "text": ""}).status_code == 422
    assert client.post("/api/generate", json={**base, "text": "x" * 601}).status_code == 400
    assert client.post("/api/generate", json={**base, "language": "xx"}).status_code == 400
    assert client.post("/api/generate", json={**base, "cfg_weight": 5}).status_code == 422
    assert client.post("/api/generate", json={**base, "seed": -1}).status_code == 422
    assert client.get("/api/jobs/missing").status_code == 404
    assert client.delete("/api/jobs/missing").status_code == 404


def test_generate_uses_voice_language_by_default(client, wav_file):
    voice = upload_voice(client, wav_file, language="de")
    job = client.post("/api/generate", json={"voice_id": voice["id"], "text": "Guten Tag."}).json()
    assert job["language"] == "de"
    assert wait_for_job(client, job["id"])["status"] == "done"


def test_job_fails_when_voice_deleted_before_processing(client, wav_file):
    voice = upload_voice(client, wav_file)
    job = client.post("/api/generate", json={"voice_id": voice["id"], "text": "Short."}).json()
    finished = wait_for_job(client, job["id"])
    assert finished["status"] == "done"
    # Deleting the voice keeps the finished job and its audio available.
    client.delete(f"/api/voices/{voice['id']}")
    assert client.get(f"/api/jobs/{job['id']}/audio").status_code == 200


def test_jobs_are_requeued_after_restart(settings, wav_file):
    from fastapi.testclient import TestClient

    from app.db import Database
    from app.main import create_app

    with TestClient(create_app(settings)) as client:
        voice = upload_voice(client, wav_file)
        job = client.post("/api/generate", json={"voice_id": voice["id"], "text": "Hello."}).json()
        wait_for_job(client, job["id"])

    # Simulate a crash mid-generation: leave a job in 'running' state on disk.
    db = Database(settings.db_path)
    crashed = db.create_job(voice, "Recovered after restart.", "en", {})
    db.update_job(crashed["id"], status="running")

    with TestClient(create_app(settings)) as client:
        recovered = wait_for_job(client, crashed["id"])
        assert recovered["status"] == "done"
        assert recovered["has_audio"] is True


def test_upload_size_limit(settings, tmp_path: Path):
    from fastapi.testclient import TestClient

    from app.main import create_app

    settings.max_upload_mb = 1
    with TestClient(create_app(settings)) as client:
        big = make_wav(tmp_path / "big.wav", seconds=40, sr=44100, channels=2)  # ~14 MB
        with big.open("rb") as fh:
            res = client.post("/api/voices", files={"file": ("big.wav", fh, "audio/wav")})
        assert res.status_code == 413
        assert not list(settings.uploads_dir.iterdir())
        assert not list(settings.voices_dir.iterdir())
