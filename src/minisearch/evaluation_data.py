import csv
import json
from pathlib import Path


def load_queries(path: str | Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as query_file:
        for line in query_file:
            if not line.strip():
                continue
            record = json.loads(line)
            queries[record["_id"]] = record["text"]
    return queries


def load_qrels(path: str | Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with Path(path).open(encoding="utf-8", newline="") as qrels_file:
        rows = csv.reader(qrels_file, delimiter="\t")
        for row in rows:
            if not row:
                continue
            if row[0].strip().lower() == "query-id":
                continue
            if len(row) != 3:
                raise ValueError(f"Expected 3 columns in qrels row: {row!r}")

            query_id, document_id, relevance = (field.strip() for field in row)
            if int(relevance) > 0:
                qrels.setdefault(query_id, set()).add(document_id)
    return qrels
