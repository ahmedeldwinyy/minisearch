from typing import get_type_hints

from minisearch.tokenizer import Tokenizer


def test_tokenize_empty_string() -> None:
    assert Tokenizer().tokenize("") == []


def test_tokenize_punctuation_only() -> None:
    assert Tokenizer().tokenize("!?.,--_()") == []


def test_tokenize_lowercases_mixed_case_text() -> None:
    assert Tokenizer().tokenize("PyThOn SeArCh") == ["python", "search"]


def test_tokenize_keeps_digits_in_tokens() -> None:
    assert Tokenizer().tokenize("BRCA1") == ["brca1"]


def test_tokenize_splits_hyphenated_words() -> None:
    assert Tokenizer(stopwords=()).tokenize("state-of-the-art") == [
        "state",
        "of",
        "the",
        "art",
    ]


def test_tokenize_collapses_repeated_whitespace() -> None:
    assert Tokenizer().tokenize("one  \t two\nthree") == ["one", "two", "three"]


def test_tokenize_removes_default_and_custom_stopwords() -> None:
    assert Tokenizer().tokenize("The quick and the fox") == ["quick", "fox"]
    assert Tokenizer(stopwords={"search"}).tokenize("The search") == ["the"]


def test_tokenize_preserves_arabic_words() -> None:
    assert Tokenizer(stopwords=()).tokenize("اللغة العربية جميلة") == [
        "اللغة",
        "العربية",
        "جميلة",
    ]


def test_tokenize_has_requested_type_hints() -> None:
    assert get_type_hints(Tokenizer.tokenize) == {
        "text": str,
        "return": list[str],
    }
