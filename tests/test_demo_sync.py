"""The GitHub Pages demo (docs/) reuses the real Fine-tuning UI; keep the copies identical."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def card(html: str) -> str:
    start = html.index('<section class="card" id="training-card">')
    return html[start:html.index("</section>", start)]


def test_demo_runs_the_real_training_script():
    # docs/training-mock.js fakes the backend; the UI code itself must be the app's.
    assert (ROOT / "docs/training.js").read_bytes() == (ROOT / "app/static/training.js").read_bytes()


def test_demo_card_markup_matches_the_app():
    app = (ROOT / "app/static/index.html").read_text(encoding="utf-8")
    demo = (ROOT / "docs/index.html").read_text(encoding="utf-8")
    assert card(demo) == card(app)
    assert 'id="gen-training-note"' in demo
    assert demo.index("training-mock.js") < demo.index('src="training.js"')  # the mock must load first


def test_demo_has_the_fine_tuning_styles():
    app_css = (ROOT / "app/static/style.css").read_text(encoding="utf-8")
    block = app_css[app_css.index("/* Fine-tuning ---"):]
    assert block in (ROOT / "docs/style.css").read_text(encoding="utf-8")
