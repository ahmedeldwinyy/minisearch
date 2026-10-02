import pytest

from minisearch.significance import (
    compare_paired_scores,
    format_significance_section,
)


def test_clear_paired_difference_has_small_p_values() -> None:
    tfidf = {f"q-{index}": 0.1 + index * 0.01 for index in range(8)}
    bm25 = {query_id: score + 0.6 for query_id, score in tfidf.items()}

    result = compare_paired_scores(bm25, tfidf, seed=2026, resamples=2000)

    assert result.mean_difference == pytest.approx(0.6)
    assert result.wins_first == 8
    assert result.wins_second == 0
    assert result.ties == 0
    assert result.ttest_p_value < 0.01
    assert result.wilcoxon_p_value < 0.01


def test_identical_scores_have_no_difference() -> None:
    scores = {f"q-{index}": index / 10 for index in range(5)}

    result = compare_paired_scores(scores, scores, seed=9, resamples=500)

    assert result.mean_difference == 0
    assert result.confidence_interval == (0.0, 0.0)
    assert result.ttest_p_value == 1.0
    assert result.wilcoxon_p_value == 1.0
    assert result.ties == 5


def test_bootstrap_confidence_interval_is_seeded() -> None:
    first = {f"q-{index}": index / 20 for index in range(12)}
    second = {query_id: score - 0.1 for query_id, score in first.items()}

    first_result = compare_paired_scores(first, second, seed=33, resamples=1000)
    second_result = compare_paired_scores(first, second, seed=33, resamples=1000)

    assert first_result.confidence_interval == second_result.confidence_interval


def test_significance_section_reports_difference_tests_and_interpretation() -> None:
    scores = {f"q-{index}": 0.3 + index / 100 for index in range(6)}
    result = compare_paired_scores(
        scores,
        {query_id: score - 0.05 for query_id, score in scores.items()},
        resamples=200,
    )

    report = format_significance_section(result)

    assert "Mean paired NDCG@10 difference (BM25 - TF-IDF)" in report
    assert "paired t-test" in report
    assert "Wilcoxon signed-rank" in report
    assert "bootstrap 95% CI" in report
    assert "BM25 wins | TF-IDF wins | Ties" in report
