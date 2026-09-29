# Static demo (GitHub Pages)

This folder is a **non-functional mockup** of the Voice Clone front-end,
published to GitHub Pages so the UI can be viewed without setting anything up.

It renders the real interface with canned voices and simulates generation
entirely in the browser. There is **no backend**: no network calls, no uploads,
nothing stored. The "generated" audio is a short tone synthesised in-page so the
players and downloads work.

The real application needs an NVIDIA GPU and a Python server (see the project
[README](../README.md)); it cannot run on GitHub Pages.

## Files

- `index.html` — the page markup (mirrors `app/static/index.html`)
- `style.css` — a copy of the app stylesheet plus the demo banner
- `demo.js` — the mock: canned data and simulated generation

## Deploying

The workflow at `.github/workflows/pages.yml` publishes this folder on every
push that touches it, and turns Pages on automatically (`enablement: true`) the
first time it runs. The site then appears at
`https://<owner>.github.io/voice-clone-app/`. If your account restricts
auto-enablement, turn it on once under **Settings → Pages → Build and
deployment → Source: GitHub Actions** and re-run the workflow.

To preview locally:

```bash
python -m http.server -d docs 8080   # then open http://localhost:8080
```
