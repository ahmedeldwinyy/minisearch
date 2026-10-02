import pytest

from minisearch.document import Document
from minisearch.inverted_index import InvertedIndex
from minisearch.tokenizer import Tokenizer
from minisearch.vectors import TFIDFDocumentVectors


@pytest.fixture
def vectors() -> TFIDFDocumentVectors:
    index = InvertedIndex(tokenizer=Tokenizer(stopwords=()))
    index.add_document(Document(id="doc-1", title="red apple", body="sweet fruit"))
    index.add_document(
        Document(id="doc-2", title="red apple", body="sweet fruit fruit")
    )
    index.add_document(Document(id="doc-3", title="blue ocean", body="deep water"))
    index.add_document(Document(id="doc-4", title="green leaf", body="plant"))
    return TFIDFDocumentVectors.from_index(index)


def test_vectorized_and_plain_python_versions_match(
    vectors: TFIDFDocumentVectors,
) -> None:
    assert vectors.more_like_this("doc-1", k=3) == vectors.more_like_this_loop(
        "doc-1", k=3
    )


def test_near_duplicate_is_the_most_similar_document(
    vectors: TFIDFDocumentVectors,
) -> None:
    recommendations = vectors.more_like_this("doc-1", k=3)

    assert recommendations[0][0] == "doc-2"
    assert all(doc_id != "doc-1" for doc_id, _ in recommendations)


def test_unknown_document_id_raises_clear_error(
    vectors: TFIDFDocumentVectors,
) -> None:
    with pytest.raises(KeyError, match="Unknown document ID"):
        vectors.more_like_this("missing")
