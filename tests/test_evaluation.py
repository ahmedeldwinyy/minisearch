import pytest

from minisearch.evaluation import (
    evaluate,
    ndcg_at_k,
    per_query_ndcg,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from minisearch.inverted_index import InvertedIndex
from minisearch.rankers import Ranker


class FixedRanker(Ranker):
    def rank(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return [("doc-1", 2.0), ("doc-2", 1.0)][:k]


def test_precision_at_k_matches_hand_calculation() -> None:
    # Two relevant results among four requested positions: precision@4 = 2 / 4.
    assert precision_at_k(
        ["doc-1", "other", "doc-2", "other-2"], {"doc-1", "doc-2"}, 4
    ) == pytest.approx(0.5)


def test_recall_at_k_matches_hand_calculation() -> None:
    # Two of the three relevant documents are retrieved: recall@3 = 2 / 3.
    assert recall_at_k(
        ["doc-1", "other", "doc-2"], {"doc-1", "doc-2", "doc-3"}, 3
    ) == pytest.approx(2 / 3)


def test_reciprocal_rank_for_relevant_result_at_rank_three() -> None:
    # The first relevant result is at rank 3: reciprocal rank = 1 / 3.
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)


def test_ndcg_is_one_for_perfect_relevant_order() -> None:
    # DCG = IDCG = 1/log2(2) + 1/log2(3), so NDCG = 1.
    assert ndcg_at_k(["doc-1", "doc-2", "other"], {"doc-1", "doc-2"}, 3) == 1


def test_ndcg_for_single_relevant_result_at_rank_three() -> None:
    # DCG = 1/log2(4) = 0.5 and IDCG = 1, so NDCG@3 = 0.5.
    assert ndcg_at_k(["a", "b", "c"], {"c"}, 3) == pytest.approx(0.5)


def test_metrics_return_zero_when_no_relevant_result_is_found() -> None:
    retrieved = ["other-1", "other-2", "other-3"]
    relevant = {"doc-1", "doc-2"}

    assert precision_at_k(retrieved, relevant, 3) == 0
    assert recall_at_k(retrieved, relevant, 3) == 0
    assert reciprocal_rank(retrieved, relevant) == 0
    assert ndcg_at_k(retrieved, relevant, 3) == 0


def test_metrics_return_zero_for_empty_relevance_and_nonpositive_k() -> None:
    assert precision_at_k(["doc-1"], set(), 2) == 0
    assert recall_at_k(["doc-1"], set(), 2) == 0
    assert ndcg_at_k(["doc-1"], {"doc-1"}, 0) == 0
    assert reciprocal_rank(["doc-1"], set()) == 0


def test_evaluate_averages_labeled_queries_only() -> None:
    ranker = FixedRanker(InvertedIndex())

    metrics = evaluate(
        ranker,
        queries={"q-1": "first", "q-2": "second", "q-3": "unlabeled"},
        qrels={"q-1": {"doc-1"}, "q-2": {"doc-3"}},
        k=2,
    )

    assert metrics == {
        "precision@2": pytest.approx(0.25),
        "recall@2": pytest.approx(0.5),
        "MRR": pytest.approx(0.5),
        "NDCG@2": pytest.approx(0.5),
    }


def test_per_query_ndcg_returns_scores_only_for_labeled_queries() -> None:
    ranker = FixedRanker(InvertedIndex())

    scores = per_query_ndcg(
        ranker,
        queries={"q-1": "first", "q-2": "second"},
        qrels={"q-1": {"doc-1"}},
        k=2,
    )

    assert scores == {"q-1": 1.0}
