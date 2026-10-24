"""Turn a phone recording into a clean reference clip for voice cloning.

Converts any format ffmpeg reads (m4a, mp3, wav, webm…) to mono 24 kHz WAV,
removes the silence at both ends and brings the level to -20 LUFS.  The app
does none of this on upload, and the cloning engines copy whatever the clip
contains: leading silence, a quiet level or room noise.

With ``--transcribe`` (needs ``pip install faster-whisper``) it also writes the
clip's transcript next to it, using a Whisper model fine-tuned on Quebec
French.  Read it and fix it: Qwen3-TTS expects the exact words of the clip.

    python -m scripts.prepare_reference my-recording.m4a voice/reference.wav --transcribe
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 24000
ASR_MODEL = "ele-sage/whisper-large-v3-turbo-fr-quebecois-ct2"
# silenceremove only trims the start: reverse, trim again, reverse back for the end.
_TRIM = "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.15"
FILTERS = f"{_TRIM},areverse,{_TRIM},areverse,loudnorm=I=-20:TP=-2"


def prepare(src: Path, dst: Path) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("ffmpeg is required")
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-vn", "-af", FILTERS,
                    "-ar", str(SR), "-ac", "1", "-acodec", "pcm_s16le", str(dst)], check=True)
    return sf.read(str(dst), dtype="float32")[0]


def noise_floor_db(audio: np.ndarray) -> float:
    """Level of the quietest 10 % of samples, a rough measure of room noise."""
    return float(20 * np.log10(np.percentile(np.abs(audio), 10) + 1e-9))


def transcribe(audio: np.ndarray, language: str) -> str:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("--transcribe needs faster-whisper: pip install faster-whisper")
    # 24 kHz -> 16 kHz, Whisper's rate (linear interpolation is fine for speech recognition).
    n = int(len(audio) * 16000 / SR)
    audio16 = np.interp(np.linspace(0, len(audio), n, endpoint=False), np.arange(len(audio)), audio)
    model = WhisperModel(ASR_MODEL, device="auto", compute_type="int8")
    segments, _ = model.transcribe(audio16.astype(np.float32), language=language)
    return " ".join(s.text.strip() for s in segments)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", type=Path)
    parser.add_argument("output", type=Path, help="WAV file to write, e.g. voice/reference.wav")
    parser.add_argument("--transcribe", action="store_true", help="also write <output>.txt (faster-whisper)")
    parser.add_argument("--language", default="fr")
    args = parser.parse_args()

    audio = prepare(args.recording, args.output)
    print(f"{args.output}: {len(audio) / SR:.1f} s, noise floor {noise_floor_db(audio):.0f} dBFS "
          "(aim for below -55; quieter room or closer mic otherwise)")
    if args.transcribe:
        text = transcribe(audio, args.language)
        transcript = args.output.with_suffix(".txt")
        transcript.write_text(text + "\n", encoding="utf-8")
        print(f"{transcript}: check it word for word against the recording\n  {text}")


if __name__ == "__main__":
    main()
