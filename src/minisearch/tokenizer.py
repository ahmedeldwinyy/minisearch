import re
from collections.abc import Iterable

DEFAULT_STOPWORDS = frozenset({"a", "an", "and", "in", "of", "the", "to"})
TOKEN_PATTERN = re.compile(r"[^\W_]+")
ARABIC_DIACRITICS = re.compile(r"[\u064b-\u0652\u0670]")
ARABIC_TRANSLATION_TABLE = str.maketrans(
    "أإآٱى٠١٢٣٤٥٦٧٨٩",
    "ااااي0123456789",
    "ـ",
)


class Tokenizer:
    def __init__(self, stopwords: Iterable[str] | None = None) -> None:
        words = DEFAULT_STOPWORDS if stopwords is None else stopwords
        self._stopwords = frozenset(word.lower() for word in words)

    def tokenize(self, text: str) -> list[str]:
        normalized_text = ARABIC_DIACRITICS.sub("", text).translate(
            ARABIC_TRANSLATION_TABLE
        )
        return [
            token
            for token in TOKEN_PATTERN.findall(normalized_text.lower())
            if token not in self._stopwords
        ]
