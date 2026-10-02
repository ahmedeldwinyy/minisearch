from minisearch.recommender_evaluation import RecommenderMetrics
from scripts.recommend_eval import (
    ColdStartRow,
    format_cold_start_table,
    format_recommender_results_table,
)


def test_recommender_results_table_compares_mf_and_baselines() -> None:
    results = {
        "matrix_factorization": RecommenderMetrics(0.8, 0.31, 0.24),
        "popularity": RecommenderMetrics(None, 0.12, 0.09),
        "random": RecommenderMetrics(None, 0.03, 0.02),
    }

    table = format_recommender_results_table(results)

    assert "| Strategy | Test RMSE | Precision@10 | Recall@10 |" in table
    assert "| matrix_factorization | 0.800 | 0.310 | 0.240 |" in table
    assert "| popularity | N/A | 0.120 | 0.090 |" in table
    assert "| random | N/A | 0.030 | 0.020 |" in table


def test_cold_start_table_shows_history_size_and_selected_strategy() -> None:
    rows = [
        ColdStartRow(history_size=0, strategy="popularity", precision_at_10=0.1),
        ColdStartRow(history_size=3, strategy="content", precision_at_10=0.2),
        ColdStartRow(
            history_size=5, strategy="matrix_factorization", precision_at_10=0.3
        ),
    ]

    table = format_cold_start_table(rows)

    assert "| History ratings | Strategy | Precision@10 |" in table
    assert "| 0 | popularity | 0.100 |" in table
    assert "| 3 | content | 0.200 |" in table
    assert "| 5 | matrix_factorization | 0.300 |" in table
