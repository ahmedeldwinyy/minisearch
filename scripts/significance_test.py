from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from minisearch.document import Document
from minisearch.evaluation import per_query_ndcg
from minisearch.evaluation_data import load_qrels, load_queries
from minisearch.inverted_index import InvertedIndex
from minisearch.loader import load_documents
from minisearch.rankers import BM25Ranker, TFIDFRanker
from minisearch.significance import (
    PairedSignificanceResult,
    compare_paired_scores,
    format_significance_section,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ARCHIVE = PROJECT_ROOT / "data" / "raw" / "scifact.zip"
REPORT_PATH = PROJECT_ROOT / "docs" / "EVALUATION.md"
BM25_K1 = 1.2
BM25_B = 0.75
BOOTSTRAP_SEED = 2026
BOOTSTRAP_RESAMPLES = 10_000


def _load_scifact_data(
    archive_path: Path,
) -> tuple[list[Document], dict[str, str], dict[str, set[str]]]:
    with ZipFile(archive_path) as archive:
        with TemporaryDirectory(prefix="minisearch-significance-") as directory:
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
            for query_id, labels in load_qrels(test_qrels_path).items():
                qrels.setdefault(query_id, set()).update(labels)
            return load_documents(corpus_path), queries, qrels


def _build_index(documents: list[Document]) -> InvertedIndex:
    index = InvertedIndex()
    for document in documents:
        index.add_document(document)
    return index


def run_significance_test(
    archive_path: Path = DEFAULT_DATASET_ARCHIVE,
    report_path: Path = REPORT_PATH,
) -> PairedSignificanceResult:
    if not archive_path.is_file():
        raise FileNotFoundError(f"SciFact archive not found at {archive_path}")

    documents, queries, qrels = _load_scifact_data(archive_path)
    index = _build_index(documents)
    tfidf_scores = per_query_ndcg(TFIDFRanker(index), queries, qrels, k=10)
    bm25_scores = per_query_ndcg(
        BM25Ranker(index, k1=BM25_K1, b=BM25_B), queries, qrels, k=10
    )
    result = compare_paired_scores(
        bm25_scores,
        tfidf_scores,
        seed=BOOTSTRAP_SEED,
        resamples=BOOTSTRAP_RESAMPLES,
    )

    section = format_significance_section(result)
    existing_report = report_path.read_text(encoding="utf-8")
    marker = "## Paired Ranker Significance"
    if marker in existing_report:
        existing_report = existing_report.split(marker, maxsplit=1)[0].rstrip()
    report_path.write_text(f"{existing_report.rstrip()}\n\n{section}", encoding="utf-8")
    return result


def main() -> None:
    result = run_significance_test()
    print(format_significance_section(result), end="")


if __name__ == "__main__":
    main()
