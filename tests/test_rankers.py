import random
from typing import Any

import pytest

from minisearch.inverted_index import InvertedIndex
from minisearch.rankers import Ranker, top_k


class DummyRanker(Ranker):
    def rank(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return [(query, float(k))]


def test_ranker_base_cannot_be_instantiated() -> None:
    ranker_class: Any = Ranker
    with pytest.raises(TypeError, match="abstract"):
        ranker_class(InvertedIndex())


def test_dummy_ranker_uses_base_index_and_returns_ranked_results() -> None:
    index = InvertedIndex()
    ranker = DummyRanker(index)

    assert ranker.index is index
    assert ranker.rank("query", k=3) == [("query", 3.0)]


def test_top_k_returns_all_results_when_k_is_larger_than_input() -> None:
    scores = {"doc-2": 2.0, "doc-1": 1.0}

    assert top_k(scores, 10) == [("doc-2", 2.0), ("doc-1", 1.0)]


def test_top_k_returns_empty_list_for_zero_k() -> None:
    assert top_k({"doc-1": 1.0}, 0) == []


def test_top_k_breaks_ties_by_document_id() -> None:
    scores = {"doc-3": 1.0, "doc-1": 1.0, "doc-2": 1.0}

    assert top_k(scores, 2) == [("doc-1", 1.0), ("doc-2", 1.0)]


def test_top_k_matches_full_sort_on_seeded_scores() -> None:
    generator = random.Random(2026)
    scores = {
        f"doc-{document_number:03}": float(generator.randint(-10, 10))
        for document_number in range(100)
    }
    expected = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

    for k in (1, 7, 50, 100, 110):
        assert top_k(scores, k) == expected[:k]
