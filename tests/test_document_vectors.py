from math import log, sqrt

import numpy as np
import pytest

from minisearch.document import Document
from minisearch.inverted_index import InvertedIndex
from minisearch.tokenizer import Tokenizer
from minisearch.vectors import TFIDFDocumentVectors


@pytest.fixture
def vectors() -> TFIDFDocumentVectors:
    index = InvertedIndex(tokenizer=Tokenizer(stopwords=()))
    index.add_document(Document(id="doc-1", title="rare common common", body=""))
    index.add_document(Document(id="doc-2", title="common", body=""))
    index.add_document(Document(id="doc-3", title="", body=""))
    return TFIDFDocumentVectors.from_index(index)


def test_tfidf_csr_shape_and_hand_computed_weights(
    vectors: TFIDFDocumentVectors,
) -> None:
    assert vectors.matrix.shape == (3, 2)
    assert vectors.doc_ids == ("doc-1", "doc-2", "doc-3")
    assert vectors.terms == ("common", "rare")

    common_weight = (1 + log(2)) * log(3 / 2)
    rare_weight = log(3)
    row_norm = sqrt(common_weight**2 + rare_weight**2)

    assert vectors.matrix[0, vectors.term_to_column["common"]] == pytest.approx(
        common_weight / row_norm
    )
    assert vectors.matrix[0, vectors.term_to_column["rare"]] == pytest.approx(
        rare_weight / row_norm
    )
    assert vectors.matrix[1, vectors.term_to_column["common"]] == pytest.approx(1)


def test_every_nonempty_row_is_l2_normalized(vectors: TFIDFDocumentVectors) -> None:
    row_norms = np.sqrt(vectors.matrix.multiply(vectors.matrix).sum(axis=1)).A1

    assert row_norms[:2] == pytest.approx([1.0, 1.0])


def test_empty_document_has_finite_zero_row(vectors: TFIDFDocumentVectors) -> None:
    empty_row = vectors.matrix.getrow(vectors.doc_to_row["doc-3"])

    assert empty_row.nnz == 0
    assert np.isfinite(empty_row.data).all()


def test_document_and_term_mappings_round_trip(
    vectors: TFIDFDocumentVectors,
) -> None:
    for document_id, row in vectors.doc_to_row.items():
        assert vectors.doc_ids[row] == document_id
    for term, column in vectors.term_to_column.items():
        assert vectors.terms[column] == term
