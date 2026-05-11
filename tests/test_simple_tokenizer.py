from llm_internals.tokenization.simple_tokenizer import tokenize

def test_tokenization():

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