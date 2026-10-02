import pytest

from minisearch.document import Document
from minisearch.inverted_index import InvertedIndex


def test_add_single_document_tracks_postings_and_statistics() -> None:
    index = InvertedIndex()
    document = Document(id="doc-1", title="amber light", body="amber glows")

    index.add_document(document)

    assert index.postings("amber") == [("doc-1", 2)]
    assert index.doc_length("doc-1") == 4
    assert index.num_docs == 1
    assert index.doc_freq("amber") == 1


def test_add_multiple_documents_tracks_each_posting() -> None:
    index = InvertedIndex()
    index.add_document(Document(id="doc-1", title="red fox", body=""))
    index.add_document(Document(id="doc-2", title="blue fox", body=""))

    assert index.postings("red") == [("doc-1", 1)]
    assert index.postings("blue") == [("doc-2", 1)]
    assert index.postings("fox") == [("doc-1", 1), ("doc-2", 1)]
    assert index.num_docs == 2
    assert index.doc_freq("fox") == 2


def test_postings_are_sorted_by_document_id() -> None:
    index = InvertedIndex()
    index.add_document(Document(id="doc-3", title="shared", body=""))
    index.add_document(Document(id="doc-1", title="shared", body=""))
    index.add_document(Document(id="doc-2", title="shared", body=""))

    assert index.postings("shared") == [
        ("doc-1", 1),
        ("doc-2", 1),
        ("doc-3", 1),
    ]


def test_unknown_term_has_no_postings() -> None:
    index = InvertedIndex()

    assert index.postings("unknown") == []
    assert index.doc_freq("unknown") == 0


def test_remove_document_allows_adding_same_id_again() -> None:
    index = InvertedIndex()
    index.add_document(Document(id="doc-1", title="alpha shared", body="alpha"))

    index.remove_document("doc-1")

    assert index.postings("alpha") == []
    assert index.postings("shared") == []
    assert index.doc_freq("alpha") == 0
    assert index.num_docs == 0

    index.add_document(Document(id="doc-1", title="beta", body=""))

    assert index.postings("alpha") == []
    assert index.postings("beta") == [("doc-1", 1)]
    assert index.doc_length("doc-1") == 1


def test_adding_duplicate_document_id_raises_clear_error() -> None:
    index = InvertedIndex()
    document = Document(id="doc-1", title="first", body="")
    index.add_document(document)

    with pytest.raises(ValueError, match="already exists"):
        index.add_document(document)


def test_empty_document_is_counted_without_creating_postings() -> None:
    index = InvertedIndex()
    index.add_document(Document(id="empty", title="", body=""))

    assert index.doc_length("empty") == 0
    assert index.num_docs == 1
    assert index.postings("anything") == []
