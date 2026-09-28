from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.auth import Auth
from app.config import Settings
from app.main import SESSION_COOKIE, create_app


def test_token_roundtrip_and_expiry():
    auth = Auth("pw", "secret", session_hours=1)
    token = auth.issue_token(now=1000)
    assert auth.verify_token(token, now=1000 + 3599)
    assert not auth.verify_token(token, now=1000 + 3601)
    assert not auth.verify_token(token + "x", now=1000)
    assert not auth.verify_token("garbage", now=1000)
    assert not auth.verify_token(None)
    assert not Auth("pw", "other-secret").verify_token(token, now=1000)


def test_password_check_is_disabled_without_password():
    auth = Auth("", "secret")
    assert not auth.enabled
    assert auth.check_password("anything")
    assert Auth("pw", "secret").check_password("pw")
    assert not Auth("pw", "secret").check_password("PW")


def test_throttling_after_repeated_failures():
    auth = Auth("pw", "secret")
    for _ in range(5):
        assert auth.retry_after("1.2.3.4") == 0
        auth.record_failure("1.2.3.4")
    assert auth.retry_after("1.2.3.4") > 0
    assert auth.retry_after("5.6.7.8") == 0
    auth.reset("1.2.3.4")
    assert auth.retry_after("1.2.3.4") == 0


def test_login_flow(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", engine="fake", app_password="hunter2", preload_model=False)
    with TestClient(create_app(settings)) as client:
        config = client.get("/api/config").json()
        assert config["auth_required"] is True
        assert config["authenticated"] is False

        assert client.get("/api/health").status_code == 200  # public
        assert client.get("/api/voices").status_code == 401
        assert client.get("/").status_code == 200  # the SPA renders the login form itself

        res = client.post("/api/login", json={"password": "wrong"})
        assert res.status_code == 401
        assert SESSION_COOKIE not in client.cookies

        res = client.post("/api/login", json={"password": "hunter2"})
        assert res.status_code == 200
        assert res.json() == {"authenticated": True, "auth_required": True}
        assert SESSION_COOKIE in client.cookies
        assert client.get("/api/voices").status_code == 200
        assert client.get("/api/config").json()["authenticated"] is True

        assert client.post("/api/logout").status_code == 200
        assert client.get("/api/voices").status_code == 401


def test_login_throttled(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", engine="fake", app_password="hunter2", preload_model=False)
    with TestClient(create_app(settings)) as client:
        for _ in range(5):
            assert client.post("/api/login", json={"password": "nope"}).status_code == 401
        res = client.post("/api/login", json={"password": "hunter2"})
        assert res.status_code == 429
        assert int(res.headers["Retry-After"]) >= 1


def test_forged_cookie_rejected(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", engine="fake", app_password="hunter2", preload_model=False)
    with TestClient(create_app(settings)) as client:
        client.cookies.set(SESSION_COOKIE, f"{int(time.time()) + 3600}.deadbeef")
        assert client.get("/api/voices").status_code == 401
