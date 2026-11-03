"""The running version shown in the UI."""

from __future__ import annotations

import json

from app import __version__, version


def test_config_reports_version_and_commit(client):
    build = client.get("/api/config").json()["build"]
    assert build["version"] == __version__
    if build["commit"]:  # run from a git checkout
        assert build["commit_short"] == build["commit"][:7]
        assert build["url"].endswith(build["commit"])


def test_build_info_file_wins_over_git(tmp_path, monkeypatch):
    stamp = tmp_path / "build-info.json"
    stamp.write_text(json.dumps({"commit": "abcdef0123456789", "date": "2026-11-03T10:00:00-05:00"}))
    monkeypatch.setattr(version, "BUILD_INFO", stamp)
    version.build_info.cache_clear()
    try:
        info = version.build_info()
    finally:
        version.build_info.cache_clear()
    assert (info["commit_short"], info["date"]) == ("abcdef0", "2026-11-03T10:00:00-05:00")
    assert info["url"] == f"{version.REPO_URL}/commit/abcdef0123456789"


def test_unknown_commit(tmp_path, monkeypatch):
    stamp = tmp_path / "build-info.json"
    stamp.write_text("{}")  # image built from a source archive, without .git
    monkeypatch.setattr(version, "BUILD_INFO", stamp)
    version.build_info.cache_clear()
    try:
        info = version.build_info()
    finally:
        version.build_info.cache_clear()
    assert info["commit"] is None and info["url"] == version.REPO_URL
