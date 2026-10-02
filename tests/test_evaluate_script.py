from scripts.evaluate import (
    BM25GridResult,
    EvaluationResult,
    format_bm25_grid_table,
    format_results_table,
    select_best_bm25,
)


def test_best_bm25_setting_uses_aggregate_ndcg() -> None:
    grid = [
        BM25GridResult(k1=0.9, b=0.4, metrics={"NDCG@10": 0.42}),
        BM25GridResult(k1=1.2, b=0.75, metrics={"NDCG@10": 0.61}),
        BM25GridResult(k1=2.0, b=0.9, metrics={"NDCG@10": 0.55}),
    ]

    assert select_best_bm25(grid) == grid[1]


def test_results_table_includes_metrics_and_average_latency() -> None:
    results = [
        EvaluationResult(
            name="TF-IDF",
            parameters="default",
            metrics={
                "precision@10": 0.1,
                "recall@10": 0.2,
                "MRR": 0.3,
                "NDCG@10": 0.4,
            },
            average_query_latency_seconds=0.001234,
        )
    ]

    table = format_results_table(results)

    assert (
        "| Ranker | Parameters | Precision@10 | Recall@10 | MRR | NDCG@10 | " in table
    )
    assert (
        "| TF-IDF | default | 0.100 | 0.200 | 0.300 | 0.400 | 1.234 ms/query |" in table
    )


def test_bm25_grid_table_includes_every_setting() -> None:
    grid = [
        BM25GridResult(
            k1=0.9,
            b=0.4,
            metrics={
                "precision@10": 0.1,
                "recall@10": 0.2,
                "MRR": 0.3,
                "NDCG@10": 0.42,
            },
        )
    ]

    assert "| 0.9 | 0.40 | 0.100 | 0.200 | 0.300 | 0.420 |" in format_bm25_grid_table(
        grid
    )
