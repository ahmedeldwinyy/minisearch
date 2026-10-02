from math import log

import pytest

from minisearch.document import Document
from minisearch.inverted_index import InvertedIndex
from minisearch.rankers import TFIDFRanker
from minisearch.tokenizer import Tokenizer


@pytest.fixture
def index() -> InvertedIndex:
    result = InvertedIndex(tokenizer=Tokenizer(stopwords=()))
    result.add_document(Document(id="doc-1", title="rare common common", body="common"))
    result.add_document(Document(id="doc-2", title="common common", body=""))
    result.add_document(Document(id="doc-3", title="other", body=""))
    return result


def test_tf_idf_matches_hand_computed_scores(index: InvertedIndex) -> None:
    ranked = TFIDFRanker(index).rank("rare common")
    expected_doc_1 = log(3) + (1 + log(3)) * log(3 / 2)
    expected_doc_2 = (1 + log(2)) * log(3 / 2)

    assert [doc_id for doc_id, _ in ranked] == ["doc-1", "doc-2"]
    assert ranked[0][1] == pytest.approx(expected_doc_1)
    assert ranked[1][1] == pytest.approx(expected_doc_2)


def test_rare_term_outweighs_common_term(index: InvertedIndex) -> None:
    rare_score = TFIDFRanker(index).rank("rare")[0][1]
    common_score = TFIDFRanker(index).rank("common")[0][1]

    assert rare_score > common_score


def test_unknown_query_terms_are_skipped(index: InvertedIndex) -> None:
    ranker = TFIDFRanker(index)

    assert ranker.rank("missing") == []
    assert ranker.rank("rare missing") == ranker.rank("rare")


def test_empty_query_returns_no_results(index: InvertedIndex) -> None:
    assert TFIDFRanker(index).rank("   ") == []


def test_repeated_query_terms_repeat_their_score(index: InvertedIndex) -> None:
    ranker = TFIDFRanker(index)

    single_score = ranker.rank("rare")[0][1]
    repeated_score = ranker.rank("rare rare")[0][1]

    assert repeated_score == pytest.approx(2 * single_score)
