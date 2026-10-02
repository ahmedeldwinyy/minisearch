from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Interaction:
    user_id: str
    doc_id: str
    rating: int


@dataclass(frozen=True)
class SyntheticInteractionDataset:
    interactions: tuple[Interaction, ...]
    user_topics: dict[str, tuple[int, ...]]
    document_topics: IntArray


class SphericalKMeans:
    def __init__(
        self, n_clusters: int = 20, max_iter: int = 20, seed: int = 42
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be positive")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed
        self.centroids_: NDArray[np.float64] | None = None

    def fit_predict(self, matrix: csr_matrix) -> IntArray:
        if matrix.shape[0] < self.n_clusters:
            raise ValueError("n_clusters cannot exceed the number of documents")

        nonzero_rows = np.flatnonzero(np.asarray(matrix.getnnz(axis=1)).ravel() > 0)
        if nonzero_rows.size < self.n_clusters:
            raise ValueError("n_clusters cannot exceed non-empty documents")

        random_generator = np.random.default_rng(self.seed)
        initial_rows = random_generator.choice(
            nonzero_rows, size=self.n_clusters, replace=False
        )
        centroids = np.asarray(matrix[initial_rows].toarray(), dtype=np.float64)
        centroids = self._normalize_rows(centroids)

        for _ in range(self.max_iter):
            similarities = np.asarray(matrix @ centroids.T)
            labels = np.argmax(similarities, axis=1).astype(np.int64)
            updated_centroids = centroids.copy()

            for topic in range(self.n_clusters):
                members = matrix[labels == topic]
                if members.shape[0] == 0:
                    continue
                centroid = np.asarray(members.sum(axis=0)).ravel()
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    updated_centroids[topic] = centroid / norm

            centroids = updated_centroids

        similarities = np.asarray(matrix @ centroids.T)
        labels = np.argmax(similarities, axis=1).astype(np.int64)
        self.centroids_ = centroids
        return cast(IntArray, labels)

    @staticmethod
    def _normalize_rows(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized = np.divide(
            matrix, norms, out=np.zeros_like(matrix), where=norms > 0
        )
        return cast(NDArray[np.float64], normalized)


def generate_synthetic_interactions(
    document_ids: list[str],
    document_topics: IntArray,
    n_users: int = 1000,
    ratings_per_user: int = 40,
    seed: int = 42,
) -> SyntheticInteractionDataset:
    topics = np.asarray(document_topics, dtype=np.int64)
    if len(document_ids) != topics.size:
        raise ValueError("document_ids and document_topics must have equal lengths")
    if not document_ids:
        raise ValueError("at least one document is required")
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("document IDs must be unique")
    if n_users < 1 or ratings_per_user < 1:
        raise ValueError("n_users and ratings_per_user must be positive")
    if ratings_per_user > len(document_ids):
        raise ValueError("ratings_per_user cannot exceed the document count")

    random_generator = np.random.default_rng(seed)
    available_topics = np.unique(topics)
    topic_count = min(3, len(available_topics))
    popularity_weights = np.minimum(
        random_generator.zipf(1.8, size=len(document_ids)), 50
    ).astype(np.float64)
    all_document_rows = np.arange(len(document_ids))
    preferred_quota = round(ratings_per_user * 0.8)
    interactions: list[Interaction] = []
    user_topics: dict[str, tuple[int, ...]] = {}

    for user_number in range(n_users):
        preferred_count = int(random_generator.integers(1, topic_count + 1))
        chosen_topics = random_generator.choice(
            available_topics, size=preferred_count, replace=False
        )
        preferred_topics = tuple(sorted(int(topic) for topic in chosen_topics))
        user_id = f"user-{user_number:04d}"
        user_topics[user_id] = preferred_topics

        preferred_candidates = np.flatnonzero(np.isin(topics, preferred_topics))
        preferred_sample_count = min(preferred_quota, preferred_candidates.size)
        selected_preferred = random_generator.choice(
            preferred_candidates,
            size=preferred_sample_count,
            replace=False,
            p=popularity_weights[preferred_candidates]
            / popularity_weights[preferred_candidates].sum(),
        )
        remaining_count = ratings_per_user - preferred_sample_count
        remaining_candidates = np.setdiff1d(
            all_document_rows, selected_preferred, assume_unique=True
        )
        selected_remaining = random_generator.choice(
            remaining_candidates,
            size=remaining_count,
            replace=False,
            p=popularity_weights[remaining_candidates]
            / popularity_weights[remaining_candidates].sum(),
        )
        selected_rows = random_generator.permutation(
            np.concatenate((selected_preferred, selected_remaining))
        )

        for document_row in selected_rows:
            preferred = int(topics[document_row]) in preferred_topics
            rating_mean = 4.3 if preferred else 2.3
            rating = int(
                np.clip(np.rint(random_generator.normal(rating_mean, 0.65)), 1, 5)
            )
            interactions.append(
                Interaction(
                    user_id=user_id,
                    doc_id=document_ids[int(document_row)],
                    rating=rating,
                )
            )

    return SyntheticInteractionDataset(
        interactions=tuple(interactions),
        user_topics=user_topics,
        document_topics=topics.copy(),
    )
