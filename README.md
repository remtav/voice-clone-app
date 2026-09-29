# Voice Clone App

A self-hosted voice cloning web app: record or upload a few seconds of a voice,
type text, and get it spoken back in that voice. Runs entirely on your own GPU,
ships as a Docker image, and is exposed to the internet through a Cloudflare
Tunnel with password protection.

Everything in the stack is open source and free software: the engine, the
model weights, the server, the front-end and the tunnel client.

**UI preview:** a static, non-functional mockup of the front-end is published to
GitHub Pages at `https://remtav.github.io/voice-clone-app/` (see [`docs/`](docs/)).
It simulates the interface in the browser with no backend; the real app needs a
GPU.

## Engine choice: Chatterbox (Resemble AI)

The engine is [Chatterbox](https://github.com/resemble-ai/chatterbox), picked
over the alternatives for these reasons:

| Criterion | Chatterbox | Why it matters |
| --- | --- | --- |
| License | **MIT for code *and* weights** | Coqui XTTS-v2, F5-TTS, Fish Speech and IndexTTS-2 all ship non-commercial weights; Chatterbox is genuinely free software end to end. |
| Cloning | Zero-shot from ~5–15 s of reference audio | No fine-tuning step, a voice is usable seconds after recording. |
| Languages | 23 (multilingual v3, June 2026) | One model for English, French, Spanish, German, Japanese, Chinese, Arabic… |
| Quality | Preferred over ElevenLabs in Resemble's side-by-side listening tests | Best-in-class among MIT options; 0.5B-parameter Llama backbone. |
| Hardware | ~4–6 GB VRAM in fp32 | Leaves most of a 24 GB RTX 3090 free; generation is a few times faster than real time. |
| Control | `exaggeration` (emotion), `cfg_weight` (pace/adherence), `temperature`, seed | Exposed as sliders in the UI. |
| Responsible use | Every output carries Resemble's inaudible PerTh watermark | Generated clips can be identified as synthetic. |

Three variants are selectable with `CHATTERBOX_MODEL`: `multilingual`
(default, v3, 23 languages), `english` (the original model) and `turbo`
(English, fastest, supports `[laugh]`/`[chuckle]` tags).

The engine sits behind a small interface (`app/engines/base.py`), so swapping
in another model later (Qwen3-TTS, Zonos, …) is a one-file change. A `fake`
engine that emits tones lets you run the UI and tests without a GPU.

## Architecture

```
browser ──HTTPS──▶ Cloudflare edge ──tunnel──▶ cloudflared container ──▶ app container :8000
                    (+ optional Access)                                  ├─ FastAPI (API + static SPA)
                                                                         ├─ worker thread (one job at a time)
                                                                         ├─ Chatterbox on CUDA
                                                                         └─ /data volume: voices, outputs, SQLite, HF cache
```

* **Front-end**: vanilla HTML/JS, no build step. Mic recording (MediaRecorder),
  file upload, voice library, generation form with advanced sliders, job
  history with inline players and downloads.
* **API**: FastAPI. Generation is asynchronous: `POST /api/generate` returns a
  job immediately and the browser polls it, so long texts never hit Cloudflare's
  100-second request limit.
* **Worker**: a single background thread drains a FIFO queue (one GPU, one job
  at a time). Long texts are split at sentence boundaries into ≤300-character
  chunks, synthesised one by one with progress reporting, then concatenated.
  Interrupted jobs are re-queued on restart.
* **Storage**: reference clips are normalised to 24 kHz mono WAV; outputs are
  16-bit WAV; metadata lives in SQLite. All of it in one `data/` volume.
* **Auth**: password login with an HMAC-signed, HttpOnly session cookie and
  per-IP throttling of failed attempts. Designed to sit behind Cloudflare Access
  as a second layer.

## Requirements

* Linux host with an NVIDIA GPU (tested target: RTX 3090, 24 GB), recent driver
* Docker Engine 24+ with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (`docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` must work)
* ~15 GB of disk for the image and ~5 GB for model weights
* A Cloudflare account with a domain on it (free plan is fine)

## Quick start

```bash
git clone <this repo> voice-clone-app && cd voice-clone-app
cp .env.example .env
# edit .env: set APP_PASSWORD (long and random) and TUNNEL_TOKEN (see below)

docker compose build            # ~10 min the first time
docker compose up -d
docker compose logs -f app      # first start downloads the weights (~4 GB)
```

Open <http://localhost:8000> locally, or your tunnel hostname from anywhere.
The status pill in the header shows the model loading, then
`Ready · ResembleAI/chatterbox (multilingual v3) · cuda · NVIDIA GeForce RTX 3090 …`.

To run only the app without a tunnel: `docker compose up -d app`.

To pre-download the weights instead of waiting on the first request:
`docker compose run --rm app python -m scripts.download_models`.

## Cloudflare Tunnel setup

1. In the [Zero Trust dashboard](https://one.dash.cloudflare.com/) go to
   **Networks → Tunnels → Create a tunnel → Cloudflared**, name it (e.g.
   `voice-clone`) and copy the token shown in the `cloudflared … run <TOKEN>`
   command into `TUNNEL_TOKEN` in `.env`.
2. On the tunnel's **Public Hostname** tab add a hostname, for example
   `voice.example.com`, with service **HTTP** and URL **`app:8000`**
   (`app` is the compose service name; both containers share a network).
3. `docker compose up -d`. The tunnel container connects outbound only; no
   ports are opened on your router and the app port is bound to `127.0.0.1`.
4. **Strongly recommended:** add Cloudflare Access in front of it. Under
   **Access → Applications → Add an application → Self-hosted**, enter the same
   hostname and create a policy that allows only your email (one-time PIN) or
   your identity provider. This adds SSO in front of the app password, so bots
   never even reach the login form. It is free for up to 50 users.

Notes:

* Browsers only allow microphone access on HTTPS or `localhost`. Through the
  tunnel you get HTTPS automatically.
* Cloudflare limits request bodies to 100 MB on the free plan; the app's own
  default upload limit is 50 MB, more than enough for a reference clip.

## Using the app

1. **Add a voice**: press *Record*, read a few sentences naturally for 5–15 s,
   stop, listen to the preview, name it and save. Or upload an existing clip.
   Clean audio, one speaker, no music. Record in the language you plan to
   generate; accents transfer across languages.
2. **Generate**: pick the voice and language, paste the text, hit *Generate*
   (or Ctrl/⌘+Enter). Progress shows per chunk; when done an inline player and
   a *Download WAV* link appear. History persists across restarts.
3. **Advanced**: `exaggeration` 0.5 is neutral, 0.7+ is more dramatic (and
   faster); lower `cfg_weight` to ~0.3 for slower, more expressive delivery or
   for fast speakers; `temperature` adds variety; a `seed` makes a result
   reproducible. *Reuse text* on any past job restores its text and settings.

## Configuration

All settings are environment variables (see `.env.example`).

| Variable | Default | Description |
| --- | --- | --- |
| `APP_PASSWORD` | *(empty)* | Login password. Empty disables auth. **Set it** before exposing the app. |
| `SECRET_KEY` | derived | Signs session cookies; set to a random string to make sessions independent of the password. |
| `SESSION_HOURS` | `168` | Session lifetime. |
| `TUNNEL_TOKEN` | | Cloudflare tunnel token (compose only). |
| `TTS_ENGINE` | `chatterbox` | `chatterbox` or `fake`. |
| `CHATTERBOX_MODEL` | `multilingual` | `multilingual`, `english` or `turbo`. |
| `CHATTERBOX_T3_MODEL` | `v3` | Multilingual T3 checkpoint: `v2`, `v3`, or a fine-tuned `.safetensors` file (absolute, or relative to `DATA_DIR`). See [Regional accents](#regional-accents). |
| `DEVICE` | `auto` | `auto`, `cuda` or `cpu` (CPU works but is very slow). |
| `PRELOAD_MODEL` | `1` | Load the model at startup rather than on first use. |
| `MAX_TEXT_CHARS` | `5000` | Max characters per generation. |
| `MAX_CHUNK_CHARS` | `300` | Chunk size for long texts. |
| `MAX_UPLOAD_MB` | `50` | Reference clip upload limit. |
| `MAX_REFERENCE_SECONDS` | `30` | Reference clips are trimmed to this length. |
| `MIN_REFERENCE_SECONDS` | `1` | Shorter clips are rejected. |
| `HISTORY_LIMIT` | `200` | Finished jobs kept; older ones and their audio are pruned. |
| `DATA_DIR` | `data` | Storage root (`/data` in Docker). |

### Regional accents

The multilingual model takes its accent from the language token, so a French
voice comes out with a France-French accent even when the reference speaks
Quebec French. Two levers, in order of cost:

1. **Settings.** Pick the *Faithful accent* preset (higher CFG weight follows
   the reference more closely) and make sure the **first 6 seconds** of the
   reference carry the accent: only they condition pronunciation.
2. **A fine-tuned T3 checkpoint.** Train a regional finetune (the way Resemble
   ships `pt-br` or `es-mx-latam`), copy it into the data volume and point
   `CHATTERBOX_T3_MODEL` at it, e.g. `models/t3_fr_ca.safetensors`. The status
   bar then shows `custom T3 t3_fr_ca.safetensors`; remove the variable to go
   back to the official v3. The full recipe for Quebec French is in
   [`docs/finetune-fr-ca-plan.md`](docs/finetune-fr-ca-plan.md).

## HTTP API

Interactive docs are at `/api/docs`. All endpoints except `/api/config`,
`/api/login` and `/api/health` require the session cookie.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/login` `{password}` | Sets the session cookie |
| `POST` | `/api/logout` | Clears it |
| `GET` | `/api/config` | Engine info, language list, limits, auth state |
| `GET` | `/api/status` | Model/GPU/queue status |
| `GET` | `/api/voices` | List voices |
| `POST` | `/api/voices` (multipart `file`, `name`, `language`) | Add a voice |
| `PATCH` | `/api/voices/{id}` `{name?, language?}` | Rename / relabel |
| `GET` | `/api/voices/{id}/audio` | Reference WAV |
| `DELETE` | `/api/voices/{id}` | Remove a voice |
| `POST` | `/api/generate` `{voice_id, text, language?, exaggeration?, cfg_weight?, temperature?, seed?}` | Queue a job (202) |
| `GET` | `/api/jobs?limit=` | Job history |
| `GET` | `/api/jobs/{id}` | Job status and progress |
| `GET` | `/api/jobs/{id}/audio?download=1` | Generated WAV |
| `DELETE` | `/api/jobs/{id}` | Cancel an active job or delete a finished one |

Example with `curl`:

```bash
curl -c jar -X POST localhost:8000/api/login -H 'content-type: application/json' -d '{"password":"…"}'
curl -b jar -F file=@me.wav -F name=Me -F language=en localhost:8000/api/voices
curl -b jar -X POST localhost:8000/api/generate -H 'content-type: application/json' \
     -d '{"voice_id":"<id>","text":"Hello from my own GPU.","language":"en"}'
curl -b jar localhost:8000/api/jobs/<job id>/audio -o out.wav
```

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
make dev            # http://localhost:8000 with the fake (tone) engine, hot reload
make test           # pytest, no GPU needed
make lint           # ruff
```

To run the real engine outside Docker: `pip install -r requirements-engine.txt`
(needs a CUDA-enabled torch 2.6 and ffmpeg on the PATH), then `make dev-gpu`.

Project layout:

```
app/
  main.py        FastAPI app factory, routes, auth middleware
  worker.py      background generation queue
  engines/       base interface, chatterbox.py, fake.py
  audio.py       ffmpeg/libsndfile conversion, concatenation
  text.py        sentence-aware chunking
  db.py          SQLite (voices, jobs)
  auth.py        password + signed cookie sessions
  static/        index.html, app.js, style.css
tests/           pytest suite using the fake engine
scripts/         download_models.py
```

## Troubleshooting

* **`could not select device driver "nvidia"`**: install the NVIDIA Container
  Toolkit and restart Docker.
* **Status shows `Model failed to load: … CUDA out of memory`**: another
  process is using the GPU; free it or set `CHATTERBOX_MODEL=turbo`.
* **First generation takes minutes**: the weights are being downloaded to
  `data/hf-cache`. Subsequent starts are ~30 s.
* **Microphone button says recording is unsupported**: you are on plain HTTP
  from a non-localhost address. Use the tunnel hostname (HTTPS) or upload a file.
* **Robotic or wrong-accent output**: use a cleaner reference clip in the
  target language, 8–15 s long, and keep `exaggeration` near 0.5.
* **Sessions reset after restart**: set a fixed `SECRET_KEY`.

## Responsible use

Only clone voices you own or have explicit permission to use, and tell
listeners when audio is synthetic. Chatterbox embeds a PerTh watermark in every
output so generated clips can be detected. Check the laws that apply to you
before publishing cloned speech.

## License

This project is MIT licensed. Chatterbox code and weights are MIT (Resemble
AI). `cloudflared` is Apache 2.0.
