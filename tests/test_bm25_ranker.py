from math import log

import pytest

from minisearch.document import Document
from minisearch.inverted_index import InvertedIndex
from minisearch.rankers import BM25Ranker
from minisearch.tokenizer import Tokenizer


def make_index(*documents: Document) -> InvertedIndex:
    index = InvertedIndex(tokenizer=Tokenizer(stopwords=()))
    for document in documents:
        index.add_document(document)
    return index


def test_bm25_matches_hand_computed_score() -> None:
    index = make_index(
        Document(id="doc-1", title="term term", body=""),
        Document(id="doc-2", title="term", body="filler filler filler"),
        Document(id="doc-3", title="other other other", body=""),
    )
    ranker = BM25Ranker(index, k1=1.5, b=0.75)

    ranked = ranker.rank("term")
    idf = log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    expected_doc_1 = idf * (2 * (1.5 + 1)) / (2 + 1.5 * (1 - 0.75 + 0.75 * 2 / 3))
    expected_doc_2 = idf * (1 * (1.5 + 1)) / (1 + 1.5 * (1 - 0.75 + 0.75 * 4 / 3))

    assert [doc_id for doc_id, _ in ranked] == ["doc-1", "doc-2"]
    assert ranked[0][1] == pytest.approx(expected_doc_1)
    assert ranked[1][1] == pytest.approx(expected_doc_2)


def test_longer_document_scores_lower_with_same_term_frequency() -> None:
    index = make_index(
        Document(id="short", title="term", body=""),
        Document(id="long", title="term", body="filler filler filler"),
    )

    ranked = BM25Ranker(index).rank("term")

    assert ranked[0][0] == "short"
    assert ranked[0][1] > ranked[1][1]


def test_zero_b_disables_document_length_normalization() -> None:
    index = make_index(
        Document(id="short", title="term", body=""),
        Document(id="long", title="term", body="filler filler filler"),
    )

    ranked = BM25Ranker(index, b=0).rank("term")

    assert ranked[0][1] == pytest.approx(ranked[1][1])


def test_term_frequency_gain_saturates() -> None:
    def score_for_frequency(frequency: int) -> float:
        index = make_index(
            Document(id="target", title=" ".join(["term"] * frequency), body=""),
            Document(id="other", title="unrelated", body=""),
        )
        return BM25Ranker(index, b=0).rank("term")[0][1]

    gain_from_one_to_two = score_for_frequency(2) - score_for_frequency(1)
    gain_from_ten_to_eleven = score_for_frequency(11) - score_for_frequency(10)

    assert gain_from_ten_to_eleven < gain_from_one_to_two


def test_average_document_length_recomputes_when_index_changes() -> None:
    index = make_index(Document(id="short", title="term", body=""))
    ranker = BM25Ranker(index)
    ranker.rank("term")

    index.add_document(Document(id="long", title="", body="filler " * 9))
    score_after_add = ranker.rank("term")[0][1]
    idf_after_add = log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    expected_after_add = idf_after_add * 2.5 / (1 + 1.5 * (0.25 + 0.75 / 5))
    assert score_after_add == pytest.approx(expected_after_add)

    index.remove_document("long")
    score_after_remove = ranker.rank("term")[0][1]
    idf_after_remove = log(1 + (1 - 1 + 0.5) / (1 + 0.5))
    expected_after_remove = idf_after_remove * 2.5 / (1 + 1.5)
    assert score_after_remove == pytest.approx(expected_after_remove)
