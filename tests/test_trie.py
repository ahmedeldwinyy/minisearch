from minisearch.document import Document
from minisearch.inverted_index import InvertedIndex
from minisearch.trie import Trie


def test_empty_trie_has_no_words_or_completions() -> None:
    trie = Trie()

    assert not trie.contains("anything")
    assert trie.with_prefix("") == []


def test_prefix_not_found_returns_no_completions() -> None:
    trie = Trie()
    trie.insert("search")

    assert trie.with_prefix("absent") == []


def test_prefix_can_also_be_a_word() -> None:
    trie = Trie()
    trie.insert("car")
    trie.insert("cart")

    assert trie.contains("car")
    assert trie.with_prefix("car") == ["car", "cart"]


def test_shared_prefixes_are_returned_alphabetically_with_limit() -> None:
    trie = Trie()
    for word in ["cart", "carbon", "car", "cat"]:
        trie.insert(word)

    assert trie.with_prefix("car") == ["car", "carbon", "cart"]
    assert trie.with_prefix("car", limit=2) == ["car", "carbon"]
    assert trie.with_prefix("car", limit=0) == []


def test_trie_supports_arabic_words() -> None:
    trie = Trie()
    trie.insert("عربية")
    trie.insert("علم")
    trie.insert("عرب")

    assert trie.contains("عربية")
    assert trie.with_prefix("عر") == ["عرب", "عربية"]


def test_trie_can_be_built_from_index_vocabulary() -> None:
    index = InvertedIndex()
    index.add_document(Document(id="doc-1", title="cancer gene", body="cancer"))
    index.add_document(Document(id="doc-2", title="gene mouse", body=""))

    assert index.terms() == ["cancer", "gene", "mouse"]
    trie = Trie.from_index(index)

    assert trie.with_prefix("ge") == ["gene"]
