import re

TOKEN_SPLIT_PATTERN = r'([,.:;?_!"()\']|--|\s)'
PUNCTUATION_SPACING_PATTERN = r'\s+([,.:;?!"()\'])'


def tokenize(raw_text: str) -> list[str]:
    """Split text into tokens."""
    preprocessed = re.split(TOKEN_SPLIT_PATTERN, raw_text)
    return [item.strip() for item in preprocessed if item.strip()]


class SimpleTokenizerV1:
    """A simple tokenizer that can encode and decode (convert text to IDs and vice versa)."""

    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        ids = [self.str_to_int[token] for token in tokenize(text)]
        return ids

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(PUNCTUATION_SPACING_PATTERN, r"\1", text)
        return text
