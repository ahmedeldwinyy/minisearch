from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pandas as pd

from minisearch.document import Document
from minisearch.evaluation import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from minisearch.evaluation_data import load_qrels, load_queries
from minisearch.inverted_index import InvertedIndex
from minisearch.loader import load_documents
from minisearch.rankers import BM25Ranker, Ranker, TFIDFRanker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ARCHIVE = PROJECT_ROOT / "data" / "raw" / "scifact.zip"
K = 10
BM25_K1_VALUES = (0.9, 1.2, 1.5, 2.0)
BM25_B_VALUES = (0.4, 0.75, 0.9)
BM25_GRID = tuple((k1, b) for k1 in BM25_K1_VALUES for b in BM25_B_VALUES)
METRIC_NAMES = (f"precision@{K}", f"recall@{K}", "MRR", f"NDCG@{K}")


@dataclass(frozen=True)
class EvaluationResult:
    name: str
    parameters: str
    metrics: Mapping[str, float]
    average_query_latency_seconds: float


@dataclass(frozen=True)
class BM25GridResult:
    k1: float
    b: float
    metrics: Mapping[str, float]
    average_query_latency_seconds: float = 0.0


def select_best_bm25(results: list[BM25GridResult]) -> BM25GridResult:
    if not results:
        raise ValueError("BM25 grid results must not be empty")
    return max(results, key=lambda result: result.metrics[f"NDCG@{K}"])


def format_results_table(results: list[EvaluationResult]) -> str:
    rows = [
        "| Ranker | Parameters | Precision@10 | Recall@10 | MRR | NDCG@10 | "
        "Average query latency |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        rows.append(
            f"| {result.name} | {result.parameters} | "
            f"{result.metrics['precision@10']:.3f} | "
            f"{result.metrics['recall@10']:.3f} | "
            f"{result.metrics['MRR']:.3f} | "
            f"{result.metrics['NDCG@10']:.3f} | "
            f"{result.average_query_latency_seconds * 1000:.3f} ms/query |"
        )
    return "\n".join(rows)


def format_bm25_grid_table(results: list[BM25GridResult]) -> str:
    rows = [
        "| k1 | b | Precision@10 | Recall@10 | MRR | NDCG@10 | Average latency |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        rows.append(
            f"| {result.k1:.1f} | {result.b:.2f} | "
            f"{result.metrics['precision@10']:.3f} | "
            f"{result.metrics['recall@10']:.3f} | "
            f"{result.metrics['MRR']:.3f} | "
            f"{result.metrics['NDCG@10']:.3f} | "
            f"{result.average_query_latency_seconds * 1000:.3f} ms/query |"
        )
    return "\n".join(rows)


def _evaluate_ranker(
    ranker: Ranker,
    name: str,
    parameters: str,
    queries: Mapping[str, str],
    qrels: Mapping[str, set[str]],
) -> EvaluationResult:
    metric_totals = {metric: 0.0 for metric in METRIC_NAMES}
    total_query_seconds = 0.0
    query_count = 0

    for query_id, query in queries.items():
        relevant = qrels.get(query_id)
        if not relevant:
            continue

        start = perf_counter()
        ranked = ranker.rank(query, K)
        total_query_seconds += perf_counter() - start
        retrieved = [doc_id for doc_id, _ in ranked]

        metric_totals[f"precision@{K}"] += precision_at_k(retrieved, relevant, K)
        metric_totals[f"recall@{K}"] += recall_at_k(retrieved, relevant, K)
        metric_totals["MRR"] += reciprocal_rank(retrieved, relevant)
        metric_totals[f"NDCG@{K}"] += ndcg_at_k(retrieved, relevant, K)
        query_count += 1

    if query_count == 0:
        raise ValueError("SciFact contains no queries with relevance labels")

    return EvaluationResult(
        name=name,
        parameters=parameters,
        metrics={
            metric: total / query_count for metric, total in metric_totals.items()
        },
        average_query_latency_seconds=total_query_seconds / query_count,
    )


def _load_scipact_data(
    archive_path: Path,
) -> tuple[list[Document], dict[str, str], dict[str, set[str]]]:
    with ZipFile(archive_path) as archive:
        with TemporaryDirectory(prefix="minisearch-evaluation-") as directory:
            extracted = Path(directory)
            corpus_path = extracted / "corpus.jsonl"
            queries_path = extracted / "queries.jsonl"
            train_qrels_path = extracted / "train.tsv"
            test_qrels_path = extracted / "test.tsv"
            for member, destination in (
                ("scifact/corpus.jsonl", corpus_path),
                ("scifact/queries.jsonl", queries_path),
                ("scifact/qrels/train.tsv", train_qrels_path),
                ("scifact/qrels/test.tsv", test_qrels_path),
            ):
                with archive.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)

            queries = load_queries(queries_path)
            qrels = load_qrels(train_qrels_path)
            for query_id, relevant_documents in load_qrels(test_qrels_path).items():
                qrels.setdefault(query_id, set()).update(relevant_documents)
            documents = load_documents(corpus_path)
            return documents, queries, qrels


def _build_index(documents: list[Document]) -> InvertedIndex:
    index = InvertedIndex()
    for document in documents:
        index.add_document(document)
    return index


def _write_results_csv(results: list[EvaluationResult], output_path: Path) -> None:
    rows = [
        {
            "ranker": result.name,
            "parameters": result.parameters,
            **result.metrics,
            "average_query_latency_ms": (result.average_query_latency_seconds * 1000),
        }
        for result in results
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _write_bm25_heatmap(results: list[BM25GridResult], output_path: Path) -> None:
    grid_frame = pd.DataFrame(
        [
            {"k1": result.k1, "b": result.b, f"NDCG@{K}": result.metrics[f"NDCG@{K}"]}
            for result in results
        ]
    )
    pivot = grid_frame.pivot(index="b", columns="k1", values=f"NDCG@{K}")
    figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
    image = axis.imshow(pivot.values, aspect="auto", cmap="viridis")
    axis.set_xticks(
        range(len(pivot.columns)), labels=[f"{value:.1f}" for value in pivot.columns]
    )
    axis.set_yticks(
        range(len(pivot.index)), labels=[f"{value:.2f}" for value in pivot.index]
    )
    axis.set_xlabel("k1")
    axis.set_ylabel("b")
    axis.set_title("BM25 NDCG@10 across parameter grid")
    figure.colorbar(image, ax=axis, label="NDCG@10")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _build_report(
    results: list[EvaluationResult],
    grid_results: list[BM25GridResult],
    query_count: int,
) -> str:
    best = select_best_bm25(grid_results)
    winner = max(results, key=lambda result: result.metrics[f"NDCG@{K}"])
    other = next(result for result in results if result is not winner)
    if winner.name == "BM25":
        reasoning = (
            f"BM25 leads NDCG@10 at {winner.metrics[f'NDCG@{K}']:.3f}, placing "
            "relevant documents higher in the top ten for this query set. Its "
            "term-frequency saturation and document-length normalization differ "
            "from TF-IDF's logarithmic term frequency. TF-IDF's NDCG@10 is "
            f"{other.metrics[f'NDCG@{K}']:.3f}, showing a lower average ranking "
            "quality on these labels. The BM25 parameters were selected by "
            "aggregating NDCG@10 across all labeled queries, not a single query."
        )
    else:
        reasoning = (
            f"TF-IDF leads NDCG@10 at {winner.metrics[f'NDCG@{K}']:.3f}, placing "
            "relevant documents higher in the top ten for this query set. Its "
            "logarithmic term frequency and corpus inverse-document frequency "
            "differ from BM25's term-frequency saturation and length "
            "normalization. BM25's NDCG@10 is "
            f"{other.metrics[f'NDCG@{K}']:.3f}, showing a lower average ranking "
            "quality on these labels. The BM25 parameters were selected by "
            "aggregating NDCG@10 across all labeled queries, not a single query."
        )

    return "\n".join(
        [
            "# SciFact Ranker Evaluation",
            "",
            f"Evaluated {query_count:,} labeled SciFact queries at k = {K} "
            "using the combined train and test relevance labels.",
            "",
            "## Ranker Results",
            "",
            format_results_table(results),
            "",
            f"Best BM25 setting by aggregate NDCG@{K}: `k1={best.k1:g}, b={best.b:g}`.",
            "",
            "## BM25 Grid",
            "",
            format_bm25_grid_table(grid_results),
            "",
            "## Parameter Heatmap",
            "",
            "![BM25 NDCG@10 heatmap](bm25-grid-ndcg.png)",
            "",
            "## Why the Rankers Differ",
            "",
            reasoning,
            "",
            "Full ranker metrics are also available in `EVALUATION_RESULTS.csv`.",
            "",
        ]
    )


def run_evaluation(
    archive_path: Path = DEFAULT_DATASET_ARCHIVE,
    output_directory: Path = PROJECT_ROOT / "docs",
) -> tuple[list[EvaluationResult], list[BM25GridResult], int]:
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"SciFact archive not found at {archive_path}; "
            "run scripts/download_dataset.py first"
        )

    documents, queries, qrels = _load_scipact_data(archive_path)
    labeled_query_count = sum(query_id in qrels for query_id in queries)
    if labeled_query_count == 0:
        raise ValueError("SciFact contains no queries with relevance labels")

    index = _build_index(documents)
    tfidf_result = _evaluate_ranker(
        TFIDFRanker(index), "TF-IDF", "default", queries, qrels
    )

    grid_results: list[BM25GridResult] = []
    for k1, b in BM25_GRID:
        result = _evaluate_ranker(
            BM25Ranker(index, k1=k1, b=b),
            "BM25",
            f"k1={k1:g}, b={b:g}",
            queries,
            qrels,
        )
        grid_results.append(
            BM25GridResult(
                k1=k1,
                b=b,
                metrics=result.metrics,
                average_query_latency_seconds=result.average_query_latency_seconds,
            )
        )

    best_bm25 = select_best_bm25(grid_results)
    bm25_result = EvaluationResult(
        name="BM25",
        parameters=f"k1={best_bm25.k1:g}, b={best_bm25.b:g}",
        metrics=best_bm25.metrics,
        average_query_latency_seconds=best_bm25.average_query_latency_seconds,
    )
    results = [tfidf_result, bm25_result]

    output_directory.mkdir(parents=True, exist_ok=True)
    _write_results_csv(results, output_directory / "EVALUATION_RESULTS.csv")
    _write_bm25_heatmap(grid_results, output_directory / "bm25-grid-ndcg.png")
    report = _build_report(results, grid_results, labeled_query_count)
    (output_directory / "EVALUATION.md").write_text(report, encoding="utf-8")
    return results, grid_results, labeled_query_count


def main() -> None:
    results, grid_results, query_count = run_evaluation()
    print(f"Evaluated {query_count} labeled SciFact queries at k={K}\n")
    print(format_results_table(results))
    print(f"\nBM25 grid results:\n{format_bm25_grid_table(grid_results)}")
    best = select_best_bm25(grid_results)
    print(f"\nBest BM25 setting: k1={best.k1:g}, b={best.b:g}")
    print(f"Artifacts saved under {PROJECT_ROOT / 'docs'}")


if __name__ == "__main__":
    main()
