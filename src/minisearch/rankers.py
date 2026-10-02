import heapq
from abc import ABC, abstractmethod
from math import log

from minisearch.inverted_index import InvertedIndex


def top_k(scores: dict[str, float], k: int) -> list[tuple[str, float]]:
    """Select k results in O(n log k) time without sorting all scores."""
    if k <= 0:
        return []
    return heapq.nsmallest(
        k, scores.items(), key=lambda result: (-result[1], result[0])
    )


class Ranker(ABC):
    def __init__(self, index: InvertedIndex) -> None:
        self.index = index

    @abstractmethod
    def rank(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return results by descending score, breaking ties by document ID."""
        raise NotImplementedError


class TFIDFRanker(Ranker):
    """Score(d, q) = sum_t (1 + log(freq(t, d))) * log(N / df(t))."""

    def rank(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if k <= 0:
            return []

        query_terms = self.index.tokenizer.tokenize(query)
        if not query_terms or self.index.num_docs == 0:
            return []

        scores: dict[str, float] = {}
        for term in query_terms:
            postings = self.index.postings(term)
            document_frequency = len(postings)
            if document_frequency == 0:
                continue

            inverse_document_frequency = log(self.index.num_docs / document_frequency)
            for doc_id, frequency in postings:
                term_frequency = 1 + log(frequency)
                scores[doc_id] = scores.get(doc_id, 0.0) + (
                    term_frequency * inverse_document_frequency
                )

        return top_k(scores, k)
