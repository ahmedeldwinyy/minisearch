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


def test_tokenize_removes_tashkeel_range() -> None:
    assert Tokenizer(stopwords=()).tokenize("ب\u064b\u0652") == ["ب"]


def test_tokenize_removes_dagger_alef_mark() -> None:
    assert Tokenizer(stopwords=()).tokenize("ه\u0670ذا") == ["هذا"]


def test_tokenize_removes_tatweel() -> None:
    assert Tokenizer(stopwords=()).tokenize("مـرحبا") == ["مرحبا"]


def test_tokenize_unifies_alef_forms() -> None:
    assert Tokenizer(stopwords=()).tokenize("أإآٱ") == ["اااا"]


def test_tokenize_unifies_alef_maqsura() -> None:
    assert Tokenizer(stopwords=()).tokenize("على") == ["علي"]


def test_tokenize_converts_arabic_indic_digits() -> None:
    assert Tokenizer(stopwords=()).tokenize("٠١٢٣٤٥٦٧٨٩") == ["0123456789"]


def test_tokenize_preserves_ta_marbuta() -> None:
    assert Tokenizer(stopwords=()).tokenize("مدرسة") == ["مدرسة"]


def test_tokenize_keeps_diacritized_word_as_one_token() -> None:
    assert Tokenizer(stopwords=()).tokenize("الْعَرَبِيَّة") == ["العربية"]


def test_tokenize_mixed_english_and_arabic_sentence() -> None:
    assert Tokenizer(stopwords=()).tokenize("MiniSearch، البَحْثُ ١٢٣") == [
        "minisearch",
        "البحث",
        "123",
    ]


def test_arabic_normalization_leaves_english_tokens_unchanged() -> None:
    assert Tokenizer(stopwords=()).tokenize("MiniSearch BRCA1") == [
        "minisearch",
        "brca1",
    ]
