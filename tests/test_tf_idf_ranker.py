from math import log

import pytest

from minisearch.document import Document
from minisearch.inverted_index import InvertedIndex
from minisearch.rankers import BM25Ranker, TFIDFRanker
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


@pytest.mark.parametrize("ranker_type", [TFIDFRanker, BM25Ranker])
def test_ranker_caches_query_and_k_and_invalidates_after_index_change(
    index: InvertedIndex, ranker_type: type[TFIDFRanker] | type[BM25Ranker]
) -> None:
    ranker = ranker_type(index)
    ranker._cached_rank.clear()
    initial_hits = ranker._cached_rank.hits
    initial_misses = ranker._cached_rank.misses

    ranker.rank("rare", k=2)
    ranker.rank("rare", k=2)
    ranker.rank("rare", k=1)

    assert ranker._cached_rank.hits == initial_hits + 1
    assert ranker._cached_rank.misses == initial_misses + 2

    index.add_document(Document(id="doc-4", title="rare", body=""))
    ranker.rank("rare", k=2)

    assert ranker._cached_rank.misses == initial_misses + 3
