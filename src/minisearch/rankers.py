from abc import ABC, abstractmethod

from minisearch.inverted_index import InvertedIndex


class Ranker(ABC):
    def __init__(self, index: InvertedIndex) -> None:
        self.index = index

    @abstractmethod
    def rank(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return results by descending score, breaking ties by document ID."""
        raise NotImplementedError
