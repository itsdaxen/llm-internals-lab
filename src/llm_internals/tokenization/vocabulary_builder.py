from .simple_tokenizer import tokenize


def build_vocab(
    raw_text: str, special_tokens: list[str] | None = None
) -> dict[str, int]:
    preprocessed = tokenize(raw_text)
    all_tokens = sorted(set(preprocessed))

    if special_tokens:
        all_tokens.extend(special_tokens)

    vocab = {token: token_id for token_id, token in enumerate(all_tokens)}

    return vocab
