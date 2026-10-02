from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import copysign, inf

import numpy as np
from numpy.typing import NDArray
from scipy.stats import ttest_rel, wilcoxon


@dataclass(frozen=True)
class PairedSignificanceResult:
    query_count: int
    mean_difference: float
    t_statistic: float
    ttest_p_value: float
    wilcoxon_statistic: float
    wilcoxon_p_value: float
    confidence_interval: tuple[float, float]
    wins_first: int
    wins_second: int
    ties: int


def format_significance_section(result: PairedSignificanceResult) -> str:
    lower, upper = result.confidence_interval
    return "\n".join(
        [
            "## Paired Ranker Significance",
            "",
            f"Paired over {result.query_count:,} SciFact queries at NDCG@10; "
            "difference is BM25 minus TF-IDF.",
            "",
            "| Mean paired NDCG@10 difference (BM25 - TF-IDF) | "
            "Paired t statistic | paired t-test p | "
            "Wilcoxon signed-rank statistic | Wilcoxon p | bootstrap 95% CI | "
            "BM25 wins | TF-IDF wins | Ties |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {result.mean_difference:.6f} | {result.t_statistic:.4f} | "
            f"{result.ttest_p_value:.6g} | {result.wilcoxon_statistic:.4f} | "
            f"{result.wilcoxon_p_value:.6g} | [{lower:.6f}, {upper:.6f}] | "
            f"{result.wins_first} | {result.wins_second} | {result.ties} |",
            "",
            "A positive mean difference favors BM25, while a negative difference "
            "favors TF-IDF; the paired tests assess whether the per-query ranking "
            "differences are consistent rather than driven by one query.",
            "The bootstrap interval estimates uncertainty in the mean difference, "
            "and the win/loss/tie counts show how broadly the observed direction "
            "holds across queries.",
            "",
        ]
    )


def compare_paired_scores(
    first_scores: Mapping[str, float],
    second_scores: Mapping[str, float],
    seed: int = 42,
    resamples: int = 10_000,
) -> PairedSignificanceResult:
    if first_scores.keys() != second_scores.keys():
        raise ValueError("paired score mappings must contain the same query IDs")
    if len(first_scores) < 2:
        raise ValueError("at least two paired queries are required")
    if resamples < 1:
        raise ValueError("resamples must be positive")

    query_ids = sorted(first_scores)
    first = np.asarray([first_scores[query_id] for query_id in query_ids])
    second = np.asarray([second_scores[query_id] for query_id in query_ids])
    differences = first - second
    mean_difference = float(np.mean(differences))
    wins_first = int(np.count_nonzero(differences > 0))
    wins_second = int(np.count_nonzero(differences < 0))
    ties = int(np.count_nonzero(differences == 0))

    if np.all(differences == 0):
        t_statistic = 0.0
        ttest_p_value = 1.0
        wilcoxon_statistic = 0.0
        wilcoxon_p_value = 1.0
        confidence_interval = (0.0, 0.0)
    else:
        constant_difference = np.allclose(
            differences, differences[0], rtol=1e-12, atol=1e-15
        )
        if constant_difference:
            t_statistic = copysign(inf, mean_difference)
            ttest_p_value = 0.0
        else:
            t_result = ttest_rel(first, second)
            t_statistic = float(t_result.statistic)
            ttest_p_value = float(t_result.pvalue)

        wilcoxon_result = wilcoxon(
            first,
            second,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
        wilcoxon_statistic = float(wilcoxon_result.statistic)
        wilcoxon_p_value = float(wilcoxon_result.pvalue)
        confidence_interval = _bootstrap_confidence_interval(
            differences, seed=seed, resamples=resamples
        )

    return PairedSignificanceResult(
        query_count=len(query_ids),
        mean_difference=mean_difference,
        t_statistic=t_statistic,
        ttest_p_value=ttest_p_value,
        wilcoxon_statistic=wilcoxon_statistic,
        wilcoxon_p_value=wilcoxon_p_value,
        confidence_interval=confidence_interval,
        wins_first=wins_first,
        wins_second=wins_second,
        ties=ties,
    )


def _bootstrap_confidence_interval(
    differences: NDArray[np.float64], seed: int, resamples: int
) -> tuple[float, float]:
    random_generator = np.random.default_rng(seed)
    query_count = len(differences)
    bootstrap_means = np.empty(resamples, dtype=np.float64)
    batch_size = 256

    for batch_start in range(0, resamples, batch_size):
        current_batch_size = min(batch_size, resamples - batch_start)
        sample_indices = random_generator.integers(
            0, query_count, size=(current_batch_size, query_count)
        )
        bootstrap_means[batch_start : batch_start + current_batch_size] = np.mean(
            differences[sample_indices], axis=1
        )

    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(lower), float(upper)
