"""Text normalisation and chunking.

Chatterbox (like most autoregressive TTS models) produces the best results on
short passages, roughly a sentence or two at a time.  Long inputs are split
into chunks that respect sentence boundaries where possible, then each chunk
is synthesised separately and the audio is concatenated.
"""

from __future__ import annotations

import re

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
# Split after sentence-ending punctuation followed by whitespace, or after CJK
# full-width terminators (which are usually not followed by a space).
_SENTENCE_END = re.compile(r"(?<=[.!?;…])\s+|(?<=[。！？])")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:，、；：])\s*")


def normalize_text(text: str) -> str:
    """Trim and collapse whitespace while keeping paragraph breaks."""
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in _PARAGRAPH_SPLIT.split(text.strip())]
    return "\n\n".join(p for p in paragraphs if p)


def split_text(text: str, max_chars: int = 300) -> list[str]:
    """Split ``text`` into chunks of at most ``max_chars`` characters.

    Paragraph boundaries always start a new chunk.  Within a paragraph,
    sentences are packed greedily.  Sentences longer than ``max_chars`` are
    split on clause punctuation and, as a last resort, on whitespace.
    """
    max_chars = max(20, int(max_chars))
    chunks: list[str] = []
    for paragraph in _PARAGRAPH_SPLIT.split(text.strip()):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if not paragraph:
            continue
        sentences = [s.strip() for s in _SENTENCE_END.split(paragraph) if s and s.strip()]
        pieces: list[str] = []
        for sentence in sentences:
            pieces.extend(_split_long(sentence, max_chars))
        chunks.extend(_pack(pieces, max_chars))
    return chunks


def _split_long(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    clauses = [c.strip() for c in _CLAUSE_SPLIT.split(sentence) if c and c.strip()]
    if len(clauses) > 1:
        out: list[str] = []
        for clause in clauses:
            out.extend(_split_long(clause, max_chars))
        return _pack(out, max_chars)
    # No punctuation to split on: break on whitespace, then hard-split.
    out = []
    current = ""
    for word in sentence.split(" "):
        if len(word) > max_chars:
            if current:
                out.append(current)
                current = ""
            out.extend(word[i : i + max_chars] for i in range(0, len(word), max_chars))
            continue
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            out.append(current)
            current = word
        else:
            current = candidate
    if current:
        out.append(current)
    return out


def _pack(pieces: list[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}" if current else piece
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks
