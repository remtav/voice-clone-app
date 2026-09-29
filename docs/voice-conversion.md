# Voice conversion — design sketch

## Goal

Keep a speaker's **original accent** (e.g. Quebec French) in the output.

TTS models decide the accent from their *language token*, not from the
reference clip, so Chatterbox reads French with a France-French accent no
matter how the reference sounds. Voice conversion (VC) avoids this: the user
**speaks** the sentence, and the model keeps that recording's content, timing
and accent while transferring only the *timbre* of a target voice.

```
TTS (today):   text ─┐
                     ├─► [ language token = accent ] ─► audio  (accent = model's)
   reference voice ──┘

VC (proposed): source speech ─┐
                              ├─► [ timbre only ] ─► audio  (accent = source's)
        reference voice ──────┘
```

## What is already in place

* `app/engines/base.py` — `Engine.supports_conversion` flag +
  `Engine.convert(source_wav, reference_wav, params)` + `ConversionParams`.
* `app/engines/vc.py` — `VoiceConversionEngine` skeleton (backend: seed-vc,
  with kNN-VC / OpenVoice v2 noted as alternatives). Import-safe; `load()`
  and `_run_model()` raise `NotImplementedError` until a backend is vendored.
* `app/engines/__init__.py` — `TTS_ENGINE=vc` selects it.

Nothing above changes existing behaviour: the TTS path is untouched.

## Backend choice

| Backend | Zero-shot | Target audio needed | Accent preservation | License |
|---|---|---|---|---|
| **seed-vc** | yes | ~5–15 s (one clip) | strong | MIT |
| **kNN-VC** | yes | ~1 min | strongest (raw feature matching) | MIT |
| **OpenVoice v2** | yes | ~5–15 s | strong (tone/style split by design) | MIT |

seed-vc is the default in the sketch because it needs a single short target
clip — the exact shape of an existing "voice" in this app. Only
`load()`/`_run_model()` in `vc.py` are backend-specific.

Add to `requirements-engine.txt` (example, pin to a commit):

```
# seed-vc (MIT) zero-shot voice conversion
seed-vc @ git+https://github.com/Plachtaa/seed-vc.git@<commit>
```

## Remaining wiring (needs review before implementing)

VC introduces a new job kind: input is a **source recording**, not text. The
cleanest approach is a parallel "convert" job type that reuses voices,
storage, the queue and history.

### 1. Config (`app/config.py`)

```python
enable_conversion: bool = False   # ENABLE_CONVERSION
max_source_seconds: int = 120     # MAX_SOURCE_SECONDS — source clips are longer than references
```

An engine can do both jobs; expose VC in the UI when
`engine.supports_conversion` is true.

### 2. DB (`app/db.py`)

Add a nullable `kind TEXT NOT NULL DEFAULT 'tts'` column to `jobs`
(`'tts' | 'convert'`) and a `source_filename TEXT` column. `CREATE TABLE IF
NOT EXISTS` + a tiny `ALTER TABLE … ADD COLUMN` migration guarded by a
`PRAGMA table_info` check keeps existing DBs working. Add `create_convert_job(
voice, source_filename, params)`.

### 3. API (`app/main.py`)

* `POST /api/convert` (multipart): fields `voice_id` + `file` (the source
  take). Reuse the exact upload/convert-to-wav path from `create_voice`, but
  cap at `max_source_seconds` and store under a new `data/sources/` dir.
  Create a `kind='convert'` job and `worker.submit(...)`.
* `ConvertRequest`-style params in `app/models.py`: `strength` (0–1),
  `pitch_shift` (semitones), `steps`, `seed` — mirror `ConversionParams`.
* Output download reuses `/api/jobs/{id}/audio` unchanged.

### 4. Worker (`app/worker.py`)

Branch in `_process` on `job["kind"]`:

```python
if job.get("kind") == "convert":
    source = self.settings.sources_dir / job["source_filename"]
    cparams = ConversionParams(strength=..., pitch_shift=..., steps=..., seed=...)
    audio = self.engine.convert(source, reference, cparams)   # no chunking
else:
    ...  # existing TTS path (split_text + synthesize loop)
```

VC runs on the whole clip at once (no `split_text`), so no chunk loop.
Everything after (`peak_normalize`, `write_wav`, history, prune) is shared.
Prune `sources_dir` alongside `outputs_dir`.

### 5. Front-end (`app/static/`)

Add a "Convert my recording" mode next to the text box: record/upload a
source clip, pick a target voice, expose `strength` / `pitch_shift` sliders.
Everything else (voice list, job history, audio player) is reused.

## Tests

* Engine contract: a fake VC path (or extend `FakeEngine` with a passthrough
  `convert`) so the convert job flow is testable without torch.
* `POST /api/convert` → job created → worker produces audio (fake engine).
* DB migration idempotency on an existing pre-`kind` database.

## Open questions

* One engine at a time (current design) vs. running Chatterbox **and** a VC
  engine together — the latter needs a small multi-engine registry and more
  VRAM.
* Source language mismatch: VC is language-agnostic, but very long clips may
  need optional silence-based segmentation for memory.
