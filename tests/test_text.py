from __future__ import annotations

from app.text import normalize_text, split_text


def test_empty_and_whitespace():
    assert split_text("") == []
    assert split_text("   \n\n  ") == []


def test_short_text_is_one_chunk():
    assert split_text("Hello world.", 300) == ["Hello world."]


def test_sentences_are_packed_up_to_limit():
    text = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
    chunks = split_text(text, 32)
    assert chunks == ["One two three. Four five six.", "Seven eight nine.", "Ten eleven twelve."]
    assert all(len(c) <= 32 for c in chunks)
    assert " ".join(chunks) == text


def test_paragraphs_start_new_chunks():
    chunks = split_text("First para.\n\nSecond para.", 300)
    assert chunks == ["First para.", "Second para."]


def test_whitespace_is_collapsed_inside_paragraphs():
    assert split_text("Hello,\n   world.  How   are\tyou?", 300) == ["Hello, world. How are you?"]


def test_long_sentence_split_on_clauses_then_words():
    sentence = "alpha beta gamma, delta epsilon zeta, eta theta iota kappa lambda mu nu xi omicron pi"
    chunks = split_text(sentence, 30)
    assert all(len(c) <= 30 for c in chunks)
    assert " ".join(chunks).replace("  ", " ") == sentence
    assert chunks[:2] == ["alpha beta gamma,", "delta epsilon zeta,"]


def test_giant_token_is_hard_split():
    word = "x" * 75
    chunks = split_text(word, 30)
    assert chunks == ["x" * 30, "x" * 30, "x" * 15]


def test_cjk_sentences_without_spaces():
    chunks = split_text("今天天气很好。我们去公园吧！你觉得呢？", 20)
    assert chunks == ["今天天气很好。 我们去公园吧！", "你觉得呢？"]
    assert split_text("今天天气很好。我们去公园吧！你觉得呢？", 300) == ["今天天气很好。 我们去公园吧！ 你觉得呢？"]


def test_minimum_chunk_size_is_enforced():
    letters = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
    assert split_text(letters, 1) == ["a b c d e f g h i j", "k l m n o p q r s t", "u v w x y z"]


def test_normalize_text():
    assert normalize_text("  a  b\n\n\n c \n d ") == "a b\n\nc d"
