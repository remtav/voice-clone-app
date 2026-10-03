"""Run one pipeline command as a subprocess, streaming its output and honouring cancel."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Result:
    returncode: int
    cancelled: bool
    tail: list[str]


def _split_lines(chunk: bytes, pending: bytes) -> tuple[list[str], bytes]:
    """tqdm redraws with \\r, everything else ends with \\n: treat both as line ends."""
    data = (pending + chunk).replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    *lines, rest = data.split(b"\n")
    return [line.decode("utf-8", "replace") for line in lines], rest


def run(
    argv: list[str],
    cwd: Path,
    on_line: Callable[[str], None],
    should_cancel: Callable[[], bool],
    env: dict[str, str] | None = None,
    poll_seconds: float = 1.0,
    grace_seconds: float = 20.0,
    tail_size: int = 40,
) -> Result:
    """Run ``argv``; every output line goes to ``on_line``.  Cancel kills the whole process group
    (the toolkit starts DataLoader workers), SIGTERM first, SIGKILL after ``grace_seconds``."""
    proc = subprocess.Popen(
        argv, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        env={**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}, start_new_session=True,
    )
    tail: list[str] = []

    def pump() -> None:
        assert proc.stdout is not None
        pending = b""
        while chunk := proc.stdout.read1(65536):
            lines, pending = _split_lines(chunk, pending)
            for line in lines:
                if line.strip():
                    tail.append(line)
                    del tail[:-tail_size]
                    on_line(line)
        if pending.strip():
            on_line(pending.decode("utf-8", "replace"))

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    cancelled = False
    while proc.poll() is None:
        if not cancelled and should_cancel():
            cancelled = True
            _signal_group(proc, signal.SIGTERM)
            deadline = time.monotonic() + grace_seconds
            while proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.2)
            if proc.poll() is None:
                _signal_group(proc, signal.SIGKILL)
        time.sleep(poll_seconds)
    reader.join(timeout=5)
    return Result(proc.returncode, cancelled, list(tail))


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass
