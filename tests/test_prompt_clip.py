"""Sentence-aligned prompt clip and re-speak detection."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from app import respeak
from app.config import Settings
from app.engines.base import SynthesisParams
from app.engines.fake import FakeEngine
from app.prompt_clip import cut_transcript, pick_cut, prepare_prompt_clip, remove_prompt_clip
from app.stt import FakeTranscriber, Word
from tests.conftest import make_wav, wait_for_job

SENTENCES = "Bon, je vais vous expliquer. Ensuite on y va. Pis ça marche? Oui, certain! Voilà."


def timed(text: str, step: float = 1.0) -> list[Word]:
    return [Word(i * step, (i + 1) * step - 0.1, t) for i, t in enumerate(text.split())]


def test_cut_prefers_the_last_sentence_end_within_the_limit():
    words = timed(SENTENCES)  # 15 words, one per second
    assert words[pick_cut(words, 9.0)].text == "va."
    assert words[pick_cut(words, 12.0)].text == "marche?"
    assert pick_cut(timed("un deux trois quatre cinq six sept huit"), 6.0) == 5  # no punctuation: last word


def test_cut_falls_back_to_a_pause():
    words = timed("un deux trois quatre cinq six sept huit")
    words[4] = Word(4.0, 4.2, "cinq")  # long gap after "cinq"
    assert pick_cut(words, 7.0) == 4


def test_transcript_is_cut_where_the_audio_is():
    words = timed(SENTENCES)
    # the user corrected the spelling Whisper wrote
    user = "Bon, je vais vous expliquer. Ensuite on y va. Pis ça marche? Oui, certain! Voilà."
    assert cut_transcript(user, words, 11) == "Bon, je vais vous expliquer. Ensuite on y va. Pis ça marche?"
    assert cut_transcript(user.replace("Ensuite", "Après"), words, 11).endswith("marche?")
    assert cut_transcript("tout autre chose", words, 11) is None


def test_prompt_clip_is_cut_cached_and_invalidated(tmp_path):
    wav = make_wav(tmp_path / "v.wav", seconds=30.0)
    stt = FakeTranscriber("fake")  # the fake transcript is one sentence spread over the clip
    assert prepare_prompt_clip(wav, FakeTranscriber.TEXT, 60, stt) is None  # already short enough
    assert prepare_prompt_clip(wav, None, 10, stt) is None  # no transcript: timbre-only mode
    clip = prepare_prompt_clip(wav, FakeTranscriber.TEXT, 10, stt)
    assert clip and clip.trimmed and 3.0 <= clip.seconds <= 10.0
    assert sf.info(str(clip.path)).duration == pytest.approx(clip.seconds, abs=0.01)
    assert FakeTranscriber.TEXT.startswith(clip.text.split()[0])
    before = clip.path.stat().st_mtime_ns
    assert prepare_prompt_clip(wav, FakeTranscriber.TEXT, 10, stt).path.stat().st_mtime_ns == before  # cached
    assert prepare_prompt_clip(wav, FakeTranscriber.TEXT, 9, None) is None  # new key, no STT: whole clip
    remove_prompt_clip(wav)
    assert not clip.path.exists()


def test_leak_detection():
    ref = "Bon, je vais vous expliquer comment ça marche aujourd'hui."
    assert respeak.leaks_reference("Bonjour. Bon je vais vous expliquer comment. Salut à tous", ref, "Salut à tous")
    assert not respeak.leaks_reference("Salut à tous, on y va", ref, "Salut à tous on y va")
    # the requested text may itself quote the reference
    assert not respeak.leaks_reference("Bon je vais vous expliquer", ref, "Bon, je vais vous expliquer")
    assert respeak.duration_suspect("Salut", 4.0) and not respeak.duration_suspect("Salut à tous", 1.2)


def test_leading_spill_is_the_end_of_the_reference():
    ref = "Bon, on va essayer ça, regarde, on ne le sait pas."
    heard = ["On", "ne", "le", "sait", "pas", "Allô", "on", "teste"]
    assert respeak.leading_spill(heard, ref, "Allo on teste") == 5
    assert respeak.leading_spill(heard, ref, "On ne le sait pas allô") == 0  # the text asks for it
    assert respeak.leading_spill(["Allô", "on", "teste"], ref, "Allô on teste") == 0
    assert respeak.leading_spill(["pas", "Allô"], ref, "Allô") == 0  # one word is too little to be sure


class LeakyEngine(FakeEngine):
    """Re-speaks the reference (a long output) on the first try of each chunk."""

    uses_reference_text = True

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[str, str | None, str]] = []

    def synthesize(self, text, reference_wav, params):
        self.calls.append((text, params.reference_text, reference_wav.name))
        audio = super().synthesize(text, reference_wav, params)
        if sum(1 for c in self.calls if c[0] == text) == 1:
            audio = audio.repeat(6)
        return audio


class ListeningTranscriber(FakeTranscriber):
    """Hears the reference in long audio and the requested words in short audio."""

    REFERENCE = "Bon je vais vous expliquer. Ensuite on y va maintenant. Voilà."

    def transcribe(self, wav, language=None, words=False):
        seconds = sf.info(str(wav)).duration
        text = self.REFERENCE if seconds > 3.0 else "Salut tout le monde."
        timed = None
        if words:
            tokens = text.split()
            step = seconds / len(tokens)
            timed = [Word(i * step, (i + 1) * step - 0.02, t) for i, t in enumerate(tokens)]
        return type(super().transcribe(wav, language))(text, language, timed)


def make_client(tmp_path, engine, transcriber):
    from fastapi.testclient import TestClient

    from app.main import create_app

    settings = Settings(data_dir=tmp_path / "data", engine="fake", stt_engine="fake", prompt_reference_seconds=10,
                        respeak_retries=2, max_chunk_chars=60)
    app = create_app(settings)
    app.state.worker.engine = engine
    app.state.worker.transcriber = transcriber
    return settings, TestClient(app)


def add_voice(client, tmp_path, transcript):
    wav = make_wav(tmp_path / "long.wav", seconds=30.0)
    with wav.open("rb") as fh:
        return client.post("/api/voices", files={"file": ("long.wav", fh, "audio/wav")},
                           data={"transcript": transcript, "language": "fr"}).json()


def test_worker_uses_prompt_clip_and_retries_a_respoken_chunk(tmp_path):
    engine = LeakyEngine()
    settings, client = make_client(tmp_path, engine, ListeningTranscriber("fake"))
    with client:
        client.app.state.worker.engine = engine
        engine.load()
        voice = add_voice(client, tmp_path, ListeningTranscriber.REFERENCE)
        job = client.post("/api/generate", json={"voice_id": voice["id"], "text": "Salut tout le monde."}).json()
        done = wait_for_job(client, job["id"])
        assert done["status"] == "done", done
        assert len(engine.calls) == 2  # leaky first try, clean retry
        text, ref_text, ref_name = engine.calls[0]
        assert ref_name.endswith(".prompt.wav") and ref_text != ListeningTranscriber.REFERENCE
        assert done["duration"] < 3.0  # the clean attempt was kept
        assert client.delete(f"/api/voices/{voice['id']}").status_code == 204
        assert list(settings.voices_dir.iterdir()) == []


class TailEngine(FakeEngine):
    """Opens every chunk with 1 s of audio standing for the end of the reference."""

    uses_reference_text = True

    def synthesize(self, text, reference_wav, params):
        return np.concatenate([np.full(24000, 0.5, dtype=np.float32), super().synthesize(text, reference_wav, params)])


class TailTranscriber(FakeTranscriber):
    """Hears the reference's last two words in the first second, then the requested text."""

    def transcribe(self, wav, language=None, words=False):
        data, sr = sf.read(str(wav))
        seconds = len(data) / sr
        spoken = ["Salut", "tout"]
        timed = [Word(0.0, 0.4, "voilà"), Word(0.5, 0.9, "pas"), Word(1.1, 1.4, spoken[0]), Word(1.5, seconds, spoken[1])]
        return type(super().transcribe(wav, language))("voilà pas Salut tout", language, timed if words else None)


def test_worker_trims_the_end_of_the_reference_from_the_start(tmp_path):
    engine = TailEngine()
    _, client = make_client(tmp_path, engine, TailTranscriber("fake"))
    with client:
        client.app.state.worker.engine = engine
        engine.load()
        voice = add_voice(client, tmp_path, "Bon, c'est tout, voilà pas.")
        plain = engine.synthesize("Salut tout", tmp_path / "x.wav", SynthesisParams())
        job = client.post("/api/generate", json={"voice_id": voice["id"], "text": "Salut tout"}).json()
        done = wait_for_job(client, job["id"])
        assert done["status"] == "done", done
        assert done["duration"] == pytest.approx(len(plain) / 24000 - 1.0, abs=0.05)  # the extra second is gone


def test_prompt_seconds_is_clamped(tmp_path):
    assert Settings(data_dir=tmp_path, prompt_reference_seconds=300).prompt_reference_seconds == 45
    assert Settings(data_dir=tmp_path, prompt_reference_seconds=0).prompt_reference_seconds == 0
