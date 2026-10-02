from collections.abc import Mapping, Sequence
from math import log2

from minisearch.rankers import Ranker


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant_found = sum(document_id in relevant for document_id in retrieved[:k])
    return relevant_found / k


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    relevant_found = sum(document_id in relevant for document_id in retrieved[:k])
    return relevant_found / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for rank, document_id in enumerate(retrieved, start=1):
        if document_id in relevant:
            return 1 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0

    discounted_gain = sum(
        1 / log2(rank + 1)
        for rank, document_id in enumerate(retrieved[:k], start=1)
        if document_id in relevant
    )
    ideal_relevant_count = min(len(relevant), k)
    ideal_gain = sum(1 / log2(rank + 1) for rank in range(1, ideal_relevant_count + 1))
    return discounted_gain / ideal_gain


def evaluate(
    ranker: Ranker,
    queries: Mapping[str, str],
    qrels: Mapping[str, set[str]],
    k: int,
) -> dict[str, float]:
    metric_totals = {
        f"precision@{k}": 0.0,
        f"recall@{k}": 0.0,
        "MRR": 0.0,
        f"NDCG@{k}": 0.0,
    }
    labeled_query_count = 0

    for query_id, query in queries.items():
        if query_id not in qrels:
            continue

        relevant = qrels[query_id]
        retrieved = [doc_id for doc_id, _ in ranker.rank(query, k)]
        metric_totals[f"precision@{k}"] += precision_at_k(retrieved, relevant, k)
        metric_totals[f"recall@{k}"] += recall_at_k(retrieved, relevant, k)
        metric_totals["MRR"] += reciprocal_rank(retrieved, relevant)
        metric_totals[f"NDCG@{k}"] += ndcg_at_k(retrieved, relevant, k)
        labeled_query_count += 1

    if labeled_query_count == 0:
        return metric_totals
    return {
        metric: total / labeled_query_count for metric, total in metric_totals.items()
    }


def per_query_ndcg(
    ranker: Ranker,
    queries: Mapping[str, str],
    qrels: Mapping[str, set[str]],
    k: int = 10,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for query_id, query in queries.items():
        if query_id not in qrels:
            continue
        retrieved = [doc_id for doc_id, _ in ranker.rank(query, k)]
        scores[query_id] = ndcg_at_k(retrieved, qrels[query_id], k)
    return scores
