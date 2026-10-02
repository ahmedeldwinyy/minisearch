from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import blake2b
from typing import Protocol

import numpy as np

from minisearch.evaluation import precision_at_k as binary_precision_at_k
from minisearch.evaluation import recall_at_k as binary_recall_at_k
from minisearch.matrix_factorization import MatrixFactorization
from minisearch.rankers import top_k
from minisearch.synthetic import Interaction


class RecommendationStrategy(Protocol):
    def recommend(
        self, user_id: str, seen_doc_ids: set[str], k: int = 10
    ) -> list[str]: ...


@dataclass(frozen=True)
class RecommenderMetrics:
    rmse: float | None
    precision_at_k: float
    recall_at_k: float


def split_interactions_by_user(
    interactions: list[Interaction], test_fraction: float = 0.2, seed: int = 42
) -> tuple[list[Interaction], list[Interaction]]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    by_user: dict[str, list[Interaction]] = defaultdict(list)
    for interaction in interactions:
        by_user[interaction.user_id].append(interaction)

    random_generator = np.random.default_rng(seed)
    training: list[Interaction] = []
    testing: list[Interaction] = []
    for user_id in sorted(by_user):
        user_interactions = by_user[user_id]
        permutation = random_generator.permutation(len(user_interactions))
        if len(user_interactions) < 2:
            test_count = 0
        else:
            test_count = min(
                len(user_interactions) - 1,
                max(1, round(len(user_interactions) * test_fraction)),
            )
        test_indices = set(int(index) for index in permutation[:test_count])
        for index, interaction in enumerate(user_interactions):
            (testing if index in test_indices else training).append(interaction)
    return training, testing


def rmse(actual: list[float], predicted: list[float]) -> float:
    if not actual or len(actual) != len(predicted):
        raise ValueError("actual and predicted must have equal, non-empty lengths")
    errors = np.asarray(predicted, dtype=np.float64) - np.asarray(
        actual, dtype=np.float64
    )
    return float(np.sqrt(np.mean(errors**2)))


def precision_recall_at_k(
    retrieved: list[str], relevant: set[str], k: int
) -> tuple[float, float]:
    return (
        binary_precision_at_k(retrieved, relevant, k),
        binary_recall_at_k(retrieved, relevant, k),
    )


class PopularityRecommender:
    def __init__(self, training: list[Interaction], document_ids: list[str]) -> None:
        self.document_ids = tuple(sorted(set(document_ids)))
        self.popularity = Counter(interaction.doc_id for interaction in training)

    def recommend(self, user_id: str, seen_doc_ids: set[str], k: int = 10) -> list[str]:
        del user_id
        candidates = {
            doc_id: float(self.popularity.get(doc_id, 0))
            for doc_id in self.document_ids
            if doc_id not in seen_doc_ids
        }
        return [doc_id for doc_id, _ in top_k(candidates, k)]


class RandomRecommender:
    def __init__(self, document_ids: list[str], seed: int = 42) -> None:
        self.document_ids = tuple(sorted(set(document_ids)))
        self.seed = seed

    def recommend(self, user_id: str, seen_doc_ids: set[str], k: int = 10) -> list[str]:
        if k <= 0:
            return []
        candidates = [
            doc_id for doc_id in self.document_ids if doc_id not in seen_doc_ids
        ]
        seed_material = f"{self.seed}:{user_id}".encode()
        user_seed = int.from_bytes(
            blake2b(seed_material, digest_size=8).digest(), "big"
        )
        random_generator = np.random.default_rng(user_seed)
        order = random_generator.permutation(len(candidates))[:k]
        return [candidates[int(index)] for index in order]


def evaluate_recommenders(
    model: MatrixFactorization,
    training: list[Interaction],
    testing: list[Interaction],
    document_ids: list[str],
    k: int = 10,
    seed: int = 42,
) -> dict[str, RecommenderMetrics]:
    if not testing:
        raise ValueError("testing interactions must not be empty")

    user_ids = sorted(
        {interaction.user_id for interaction in training}
        | {interaction.user_id for interaction in testing}
    )
    model.fit(training, user_ids=user_ids, item_ids=document_ids)
    popularity = PopularityRecommender(training, document_ids)
    random_recommender = RandomRecommender(document_ids, seed=seed)
    train_by_user: dict[str, set[str]] = defaultdict(set)
    test_by_user: dict[str, list[Interaction]] = defaultdict(list)
    for interaction in training:
        train_by_user[interaction.user_id].add(interaction.doc_id)
    for interaction in testing:
        test_by_user[interaction.user_id].append(interaction)

    strategies: dict[str, RecommendationStrategy] = {
        "popularity": popularity,
        "random": random_recommender,
    }
    precision_totals = {name: 0.0 for name in strategies}
    recall_totals = {name: 0.0 for name in strategies}
    precision_totals["matrix_factorization"] = 0.0
    recall_totals["matrix_factorization"] = 0.0

    for user_id, user_testing in test_by_user.items():
        seen_doc_ids = train_by_user[user_id]
        candidate_ids = [
            doc_id for doc_id in document_ids if doc_id not in seen_doc_ids
        ]
        relevant = {
            interaction.doc_id
            for interaction in user_testing
            if interaction.rating >= 4
        }
        mf_scores = {doc_id: model.predict(user_id, doc_id) for doc_id in candidate_ids}
        mf_recommendations = [doc_id for doc_id, _ in top_k(mf_scores, k)]
        recommendations = {
            "matrix_factorization": mf_recommendations,
            **{
                name: strategy.recommend(user_id, seen_doc_ids, k)
                for name, strategy in strategies.items()
            },
        }
        for name, retrieved in recommendations.items():
            precision, recall = precision_recall_at_k(retrieved, relevant, k)
            precision_totals[name] += precision
            recall_totals[name] += recall

    user_count = len(test_by_user)
    actual_ratings = [float(interaction.rating) for interaction in testing]
    predicted_ratings = [
        model.predict(interaction.user_id, interaction.doc_id)
        for interaction in testing
    ]
    metrics = {
        name: RecommenderMetrics(
            rmse=None,
            precision_at_k=precision_totals[name] / user_count,
            recall_at_k=recall_totals[name] / user_count,
        )
        for name in precision_totals
    }
    metrics["matrix_factorization"] = RecommenderMetrics(
        rmse=rmse(actual_ratings, predicted_ratings),
        precision_at_k=precision_totals["matrix_factorization"] / user_count,
        recall_at_k=recall_totals["matrix_factorization"] / user_count,
    )
    return metrics
