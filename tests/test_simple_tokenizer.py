from llm_internals.tokenization.simple_tokenizer import SimpleTokenizerV1, tokenize
from llm_internals.tokenization.vocabulary_builder import build_vocab


def test_tokenize_splits_words_and_punctuation():
    text = "Hello, world. Is this-- a test?"

    tokens = tokenize(text)

    assert tokens == [
        "Hello",
        ",",
        "world",
        ".",
        "Is",
        "this",
        "--",
        "a",
        "test",
        "?",
    ]


def test_tokenizer_encodes_text_to_ids():
    vocab = {
        "Hello": 0,
        ",": 1,
        "world": 2,
        ".": 3,
    }
    tokenizer = SimpleTokenizerV1(vocab)

    ids = tokenizer.encode("Hello, world.")

    assert ids == [0, 1, 2, 3]


def test_tokenizer_decodes_ids_to_text():
    vocab = {
        "Hello": 0,
        ",": 1,
        "world": 2,
        ".": 3,
    }
    tokenizer = SimpleTokenizerV1(vocab)

    text = tokenizer.decode([0, 1, 2, 3])

    assert text == "Hello, world."


def test_build_vocab_returns_token_to_id_mapping():
    raw_text = "This, is a --sample raw text."

    vocab = build_vocab(raw_text)

    assert vocab == {
        ",": 0,
        "--": 1,
        ".": 2,
        "This": 3,
        "a": 4,
        "is": 5,
        "raw": 6,
        "sample": 7,
        "text": 8,
    }

    vocab_with_special = build_vocab(
        raw_text, special_tokens=["<|endoftext|>", "<|unk|>"]
    )

    assert vocab_with_special == {
        ",": 0,
        "--": 1,
        ".": 2,
        "This": 3,
        "a": 4,
        "is": 5,
        "raw": 6,
        "sample": 7,
        "text": 8,
        "<|endoftext|>": 9,
        "<|unk|>": 10,
    }
