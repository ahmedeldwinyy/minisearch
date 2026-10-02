from typing import Any

import pytest

from minisearch.inverted_index import InvertedIndex
from minisearch.rankers import Ranker


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
