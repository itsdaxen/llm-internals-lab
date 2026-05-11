from llm_internals.tokenization.simple_tokenizer import SimpleTokenizerV1, tokenize


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
