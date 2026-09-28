"""Minimal password authentication with signed session cookies.

This is a single-user app.  The password is compared in constant time and
the session token is an HMAC-signed expiry timestamp, so no session state is
kept server-side.  Failed logins are throttled per client address.

For internet exposure, combine this with Cloudflare Access (see README).
"""

from __future__ import annotations

import hashlib
import hmac
import threading
import time


class Auth:
    def __init__(self, password: str, secret_key: str, session_hours: int = 24 * 7) -> None:
        self.password = password or ""
        self.secret = secret_key.encode("utf-8")
        self.session_seconds = int(session_hours * 3600)
        self._failures: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.password)

    # Passwords -----------------------------------------------------------
    def check_password(self, candidate: str) -> bool:
        if not self.enabled:
            return True
        return hmac.compare_digest(candidate.encode("utf-8"), self.password.encode("utf-8"))

    # Tokens --------------------------------------------------------------
    def _sign(self, expiry: int) -> str:
        return hmac.new(self.secret, str(expiry).encode("ascii"), hashlib.sha256).hexdigest()

    def issue_token(self, now: float | None = None) -> str:
        expiry = int((now if now is not None else time.time()) + self.session_seconds)
        return f"{expiry}.{self._sign(expiry)}"

    def verify_token(self, token: str | None, now: float | None = None) -> bool:
        if not token or "." not in token:
            return False
        expiry_raw, signature = token.split(".", 1)
        try:
            expiry = int(expiry_raw)
        except ValueError:
            return False
        if expiry < (now if now is not None else time.time()):
            return False
        return hmac.compare_digest(signature, self._sign(expiry))

    # Brute-force throttling ---------------------------------------------
    def retry_after(self, key: str) -> float:
        """Seconds the client must wait before another attempt (0 = none)."""
        with self._lock:
            failures, last = self._failures.get(key, (0, 0.0))
        if failures < 5:
            return 0.0
        wait = 30.0 if failures < 10 else 300.0
        return max(0.0, last + wait - time.time())

    def record_failure(self, key: str) -> None:
        with self._lock:
            failures, _ = self._failures.get(key, (0, 0.0))
            self._failures[key] = (failures + 1, time.time())

    def reset(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)
