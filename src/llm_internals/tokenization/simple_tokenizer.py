import re

TOKEN_SPLIT_PATTERN = r'([,.:;?_!"()\']|--|\s)'

def tokenize(raw_text: str) -> list[str]:
    """Split text into tokens."""
    preprocessed = re.split(TOKEN_SPLIT_PATTERN, raw_text)
    return [item.strip() for item in preprocessed if item.strip()]