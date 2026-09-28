"""SQLite persistence for voices and generation jobs.

A tiny hand-rolled layer over the standard library ``sqlite3`` module.  Each
call opens its own connection guarded by a process-wide lock, which is more
than enough for a single-user, single-GPU service.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS voices (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    language    TEXT,
    filename    TEXT NOT NULL,
    duration    REAL,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id               TEXT PRIMARY KEY,
    voice_id         TEXT NOT NULL,
    voice_name       TEXT NOT NULL,
    text             TEXT NOT NULL,
    language         TEXT,
    params           TEXT NOT NULL,
    status           TEXT NOT NULL,
    error            TEXT,
    output_filename  TEXT,
    duration         REAL,
    progress_done    INTEGER NOT NULL DEFAULT 0,
    progress_total   INTEGER NOT NULL DEFAULT 0,
    created_at       REAL NOT NULL,
    started_at       REAL,
    finished_at      REAL
);
CREATE INDEX IF NOT EXISTS jobs_created_at ON jobs(created_at);
"""

JOB_STATUSES = ("queued", "running", "done", "failed", "cancelling", "cancelled")
ACTIVE_STATUSES = ("queued", "running", "cancelling")

_JOB_COLUMNS = {
    "status",
    "error",
    "output_filename",
    "duration",
    "progress_done",
    "progress_total",
    "started_at",
    "finished_at",
}


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # Voices --------------------------------------------------------------
    def create_voice(
        self, name: str, language: str | None, filename: str, duration: float, voice_id: str | None = None
    ) -> dict:
        voice = {
            "id": voice_id or new_id(),
            "name": name,
            "language": language or None,
            "filename": filename,
            "duration": float(duration),
            "created_at": time.time(),
        }
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO voices (id, name, language, filename, duration, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                tuple(voice[k] for k in ("id", "name", "language", "filename", "duration", "created_at")),
            )
        return voice

    def list_voices(self) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM voices ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def get_voice(self, voice_id: str) -> dict | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM voices WHERE id = ?", (voice_id,)).fetchone()
        return dict(row) if row else None

    def update_voice(self, voice_id: str, **fields: Any) -> dict | None:
        allowed = {k: v for k, v in fields.items() if k in {"name", "language"}}
        if allowed:
            assignments = ", ".join(f"{k} = ?" for k in allowed)
            with self._lock, self._connect() as conn:
                conn.execute(f"UPDATE voices SET {assignments} WHERE id = ?", (*allowed.values(), voice_id))
        return self.get_voice(voice_id)

    def delete_voice(self, voice_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM voices WHERE id = ?", (voice_id,))
        return cur.rowcount > 0

    # Jobs ----------------------------------------------------------------
    def create_job(self, voice: dict, text: str, language: str | None, params: dict) -> dict:
        job_id = new_id()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, voice_id, voice_name, text, language, params, status, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, 'queued', ?)",
                (job_id, voice["id"], voice["name"], text, language, json.dumps(params), time.time()),
            )
        job = self.get_job(job_id)
        assert job is not None
        return job

    def get_job(self, job_id: str) -> dict | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs(self, limit: int = 100) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def update_job(self, job_id: str, **fields: Any) -> None:
        allowed = {k: v for k, v in fields.items() if k in _JOB_COLUMNS}
        if not allowed:
            return
        assignments = ", ".join(f"{k} = ?" for k in allowed)
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", (*allowed.values(), job_id))

    def delete_job(self, job_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        return cur.rowcount > 0

    def queued_job_ids(self) -> list[str]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at ASC").fetchall()
        return [r["id"] for r in rows]

    def requeue_stale(self) -> int:
        """Reset jobs left 'running' by a crash so the worker picks them up again."""
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'cancelled', finished_at = ? WHERE status = 'cancelling'", (now,)
            )
            cur = conn.execute(
                "UPDATE jobs SET status = 'queued', started_at = NULL, progress_done = 0 WHERE status = 'running'"
            )
        return cur.rowcount

    def prune_jobs(self, keep: int) -> list[str]:
        """Delete finished jobs beyond the newest ``keep``; returns removed output filenames."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, output_filename FROM jobs WHERE status NOT IN ('queued', 'running', 'cancelling')"
                " ORDER BY created_at DESC LIMIT -1 OFFSET ?",
                (int(keep),),
            ).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                conn.executemany("DELETE FROM jobs WHERE id = ?", [(i,) for i in ids])
        return [r["output_filename"] for r in rows if r["output_filename"]]

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> dict:
        job = dict(row)
        try:
            job["params"] = json.loads(job.get("params") or "{}")
        except json.JSONDecodeError:
            job["params"] = {}
        job["has_audio"] = bool(job.get("output_filename")) and job.get("status") == "done"
        return job
