import re
from collections.abc import Iterable

DEFAULT_STOPWORDS = frozenset({"a", "an", "and", "in", "of", "the", "to"})
TOKEN_PATTERN = re.compile(r"[^\W_]+")


class Tokenizer:
    def __init__(self, stopwords: Iterable[str] | None = None) -> None:
        words = DEFAULT_STOPWORDS if stopwords is None else stopwords
        self._stopwords = frozenset(word.lower() for word in words)

    def tokenize(self, text: str) -> list[str]:
        return [
            token
            for token in TOKEN_PATTERN.findall(text.lower())
            if token not in self._stopwords
        ]
