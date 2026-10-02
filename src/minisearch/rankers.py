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


class BM25Ranker(Ranker):
    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        super().__init__(index)
        self.k1 = k1
        self.b = b
        self._average_document_length = 0.0
        self._cached_index_version = -1
        self._refresh_average_document_length()

    def _refresh_average_document_length(self) -> None:
        if self._cached_index_version == self.index.version:
            return

        if self.index.num_docs:
            total_length = sum(
                self.index.doc_length(doc_id) for doc_id in self.index.document_ids()
            )
            self._average_document_length = total_length / self.index.num_docs
        else:
            self._average_document_length = 0.0
        self._cached_index_version = self.index.version

    def rank(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if k <= 0:
            return []

        query_terms = self.index.tokenizer.tokenize(query)
        if not query_terms or self.index.num_docs == 0:
            return []

        self._refresh_average_document_length()
        scores: dict[str, float] = {}
        for term in query_terms:
            postings = self.index.postings(term)
            document_frequency = len(postings)
            if document_frequency == 0:
                continue

            inverse_document_frequency = log(
                1
                + (self.index.num_docs - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for doc_id, frequency in postings:
                document_length = self.index.doc_length(doc_id)
                length_ratio = (
                    document_length / self._average_document_length
                    if self._average_document_length
                    else 0.0
                )
                normalized_tf = (frequency * (self.k1 + 1)) / (
                    frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                )
                scores[doc_id] = scores.get(doc_id, 0.0) + (
                    inverse_document_frequency * normalized_tf
                )

        return top_k(scores, k)
