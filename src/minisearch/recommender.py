from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.sparse import csr_matrix

from minisearch.matrix_factorization import MatrixFactorization
from minisearch.rankers import top_k
from minisearch.recommender_evaluation import PopularityRecommender
from minisearch.synthetic import Interaction
from minisearch.vectors import TFIDFDocumentVectors


class Recommender:
    def __init__(
        self,
        vectors: TFIDFDocumentVectors,
        document_ids: Sequence[str],
        popularity: PopularityRecommender,
        matrix_factorization: MatrixFactorization,
    ) -> None:
        self.vectors = vectors
        self.document_ids = tuple(sorted(set(document_ids)))
        self.popularity = popularity
        self.matrix_factorization = matrix_factorization

    def strategy_for(self, user_id: str, history: Sequence[Interaction]) -> str:
        user_history = self._user_history(user_id, history)
        if not user_history:
            return "popularity"
        if len(user_history) >= 5:
            return "matrix_factorization"
        if any(item.rating >= 4 for item in user_history):
            return "content"
        return "popularity"

    def recommend(
        self, user_id: str, history: Sequence[Interaction], k: int = 10
    ) -> list[str]:
        if k <= 0:
            return []
        user_history = self._user_history(user_id, history)
        seen_doc_ids = {item.doc_id for item in user_history}
        strategy = self.strategy_for(user_id, user_history)

        if strategy == "popularity":
            return self.popularity.recommend(user_id, seen_doc_ids, k)
        if strategy == "content":
            liked_rows = [
                self.vectors.doc_to_row[item.doc_id]
                for item in user_history
                if item.rating >= 4 and item.doc_id in self.vectors.doc_to_row
            ]
            if not liked_rows:
                return self.popularity.recommend(user_id, seen_doc_ids, k)
            profile = csr_matrix(self.vectors.matrix[liked_rows].mean(axis=0))
            profile_norm = float(np.sqrt(profile.multiply(profile).sum()))
            if profile_norm == 0:
                return self.popularity.recommend(user_id, seen_doc_ids, k)
            profile = profile.multiply(1.0 / profile_norm)
            similarities = (profile @ self.vectors.matrix.T).toarray().ravel()
            scores = {
                doc_id: float(similarities[self.vectors.doc_to_row[doc_id]])
                for doc_id in self.document_ids
                if doc_id not in seen_doc_ids and doc_id in self.vectors.doc_to_row
            }
            return [doc_id for doc_id, _ in top_k(scores, k)]

        scores = {
            doc_id: self.matrix_factorization.predict(user_id, doc_id)
            for doc_id in self.document_ids
            if doc_id not in seen_doc_ids
        }
        return [doc_id for doc_id, _ in top_k(scores, k)]

    @staticmethod
    def _user_history(
        user_id: str, history: Sequence[Interaction]
    ) -> list[Interaction]:
        return [
            interaction for interaction in history if interaction.user_id == user_id
        ]
