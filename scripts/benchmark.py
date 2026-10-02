from __future__ import annotations

import shutil
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

from minisearch.document import Document
from minisearch.inverted_index import InvertedIndex
from minisearch.loader import load_documents
from minisearch.query_parser import QueryParser
from minisearch.trie import Trie

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ARCHIVE = PROJECT_ROOT / "data" / "raw" / "scifact.zip"
CORPUS_MEMBER = "scifact/corpus.jsonl"
QUERY_TYPES = ("single term", "AND", "OR", "NOT", "nested")
QUERY_LABELS = {
    "single term": "Single-term",
    "AND": "AND",
    "OR": "OR",
    "NOT": "NOT",
    "nested": "Nested",
}


@dataclass(frozen=True)
class BenchmarkMeasurements:
    document_count: int
    query_count: int
    indexing_1000_seconds: float
    indexing_all_seconds: float
    query_averages_seconds: Mapping[str, float]
    trie_build_seconds: float
    trie_prefix: str
    trie_lookup_average_seconds: float


def format_benchmark_report(measurements: BenchmarkMeasurements) -> str:
    query_rows = [
        f"| {QUERY_LABELS[label]} query ({measurements.query_count} runs) | "
        f"{measurements.query_averages_seconds[label] * 1000:.3f} ms/query |"
        for label in QUERY_TYPES
    ]
    indexing_1000 = f"{measurements.indexing_1000_seconds:.3f} s"
    indexing_all = f"{measurements.indexing_all_seconds:.3f} s"
    and_query = f"{measurements.query_averages_seconds['AND'] * 1000:.3f} ms/query"
    or_query = f"{measurements.query_averages_seconds['OR'] * 1000:.3f} ms/query"
    not_query = f"{measurements.query_averages_seconds['NOT'] * 1000:.3f} ms/query"
    trie_lookup = f"{measurements.trie_lookup_average_seconds * 1000:.3f} ms/query"
    trie_build = f"{measurements.trie_build_seconds:.3f} s"
    prefix_lookup = (
        f"{trie_lookup} for prefix `{measurements.trie_prefix}` "
        f"over {measurements.query_count} runs"
    )

    return "\n".join(
        [
            "# MiniSearch Benchmarks",
            "",
            f"SciFact corpus: {measurements.document_count:,} documents. "
            f"Query averages use {measurements.query_count} runs.",
            "",
            "## Measurements",
            "",
            "| Operation | Measured time |",
            "| --- | ---: |",
            f"| Index build (1,000 documents) | {indexing_1000} |",
            f"| Index build (all {measurements.document_count:,} documents) "
            f"| {indexing_all} |",
            *query_rows,
            f"| Trie build ({measurements.document_count:,} documents) "
            f"| {trie_build} |",
            f"| Trie prefix lookup (`{measurements.trie_prefix}`, "
            f"{measurements.query_count} runs) | {trie_lookup} |",
            "",
            "## Big-O",
            "",
            "| Operation | Complexity | Measured time |",
            "| --- | --- | ---: |",
            f"| Index build (1,000 documents) | O(T) expected | {indexing_1000} |",
            f"| Index build (all documents) | O(T) expected | {indexing_all} |",
            f"| AND postings merge | O(|A| + |B|) | {and_query} |",
            f"| OR postings merge | O(|A| + |B|) | {or_query} |",
            f"| NOT postings merge | O(|A| + |U|) | {not_query} |",
            f"| Trie prefix lookup | O(|prefix| + V log sigma) | {prefix_lookup} |",
            "",
            "T is the number of indexed tokens, A and B are postings lists, U is "
            "the document universe, V is the trie nodes visited while finding "
            "completions, and sigma is the character alphabet size.",
            "",
        ]
    )


def _load_scifact_documents(archive_path: Path) -> list[Document]:
    with ZipFile(archive_path) as archive:
        with TemporaryDirectory(prefix="minisearch-benchmark-") as temporary_dir:
            corpus_path = Path(temporary_dir) / "corpus.jsonl"
            with (
                archive.open(CORPUS_MEMBER) as source,
                corpus_path.open("wb") as target,
            ):
                shutil.copyfileobj(source, target)
            return load_documents(corpus_path)


def _build_index(documents: list[Document]) -> InvertedIndex:
    index = InvertedIndex()
    for document in documents:
        index.add_document(document)
    return index


def _average_query_time(parser: QueryParser, query: str, runs: int) -> float:
    total_seconds = 0.0
    for _ in range(runs):
        start = perf_counter()
        parser.search(query)
        total_seconds += perf_counter() - start
    return total_seconds / runs


def _select_trie_prefix(terms: list[str]) -> str:
    if not terms:
        raise ValueError("cannot select a prefix from an empty vocabulary")

    prefix_counts = Counter(term[:2] for term in terms if len(term) > 2)
    if prefix_counts:
        return prefix_counts.most_common(1)[0][0]
    return terms[0][:1]


def run_benchmarks(
    archive_path: Path = DEFAULT_DATASET_ARCHIVE, query_count: int = 100
) -> BenchmarkMeasurements:
    if query_count < 1:
        raise ValueError("query_count must be positive")
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"SciFact archive not found at {archive_path}; "
            "run scripts/download_dataset.py first"
        )

    documents = _load_scifact_documents(archive_path)
    if len(documents) < 1000:
        raise ValueError("SciFact corpus must contain at least 1,000 documents")

    start = perf_counter()
    _build_index(documents[:1000])
    indexing_1000_seconds = perf_counter() - start

    start = perf_counter()
    index = _build_index(documents)
    indexing_all_seconds = perf_counter() - start

    terms = [term for term in index.terms() if term.upper() not in {"AND", "OR", "NOT"}]
    if len(terms) < 4:
        raise ValueError("SciFact corpus must contain at least four query terms")

    first, second, third, fourth = terms[:4]
    queries = {
        "single term": first,
        "AND": f"{first} AND {second}",
        "OR": f"{first} OR {second}",
        "NOT": f"NOT {first}",
        "nested": f"(({first} AND {second}) OR ({third} AND NOT {fourth}))",
    }
    parser = QueryParser(index)
    query_averages = {
        label: _average_query_time(parser, query, query_count)
        for label, query in queries.items()
    }

    start = perf_counter()
    trie = Trie.from_index(index)
    trie_build_seconds = perf_counter() - start

    trie_prefix = _select_trie_prefix(terms)
    start = perf_counter()
    for _ in range(query_count):
        trie.with_prefix(trie_prefix)
    trie_lookup_average_seconds = (perf_counter() - start) / query_count

    return BenchmarkMeasurements(
        document_count=len(documents),
        query_count=query_count,
        indexing_1000_seconds=indexing_1000_seconds,
        indexing_all_seconds=indexing_all_seconds,
        query_averages_seconds=query_averages,
        trie_build_seconds=trie_build_seconds,
        trie_prefix=trie_prefix,
        trie_lookup_average_seconds=trie_lookup_average_seconds,
    )


def main() -> None:
    measurements = run_benchmarks()
    report = format_benchmark_report(measurements)
    report_path = PROJECT_ROOT / "docs" / "BENCHMARKS.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
