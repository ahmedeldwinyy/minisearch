import json
from pathlib import Path

from minisearch.evaluation_data import load_qrels, load_queries


def test_load_queries_reads_jsonl(tmp_path: Path) -> None:
    queries_path = tmp_path / "queries.jsonl"
    records = [
        {"_id": "q-1", "text": "cancer therapy"},
        {"_id": "q-2", "text": "gene expression"},
    ]
    queries_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    assert load_queries(queries_path) == {
        "q-1": "cancer therapy",
        "q-2": "gene expression",
    }


def test_load_qrels_keeps_only_positive_relevance(tmp_path: Path) -> None:
    qrels_path = tmp_path / "qrels.tsv"
    qrels_path.write_text(
        "query-id\tcorpus-id\tscore\n"
        "q-1\tdoc-1\t1\n"
        "q-1\tdoc-2\t0\n"
        "q-2\tdoc-3\t2\n"
        "q-3\tdoc-4\t-1\n",
        encoding="utf-8",
    )

    assert load_qrels(qrels_path) == {"q-1": {"doc-1"}, "q-2": {"doc-3"}}
