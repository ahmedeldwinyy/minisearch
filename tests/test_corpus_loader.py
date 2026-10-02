import json
from pathlib import Path
from typing import get_type_hints

from minisearch.document import Document
from minisearch.loader import load_documents


def test_load_documents_reads_scifact_jsonl(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    records = [
        {"_id": "doc-1", "title": "First title", "text": "First body"},
        {"_id": "doc-2", "title": "Second title", "text": "Second body"},
    ]
    corpus_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n\n",
        encoding="utf-8",
    )

    documents = load_documents(corpus_path)

    assert documents == [
        Document(id="doc-1", title="First title", body="First body"),
        Document(id="doc-2", title="Second title", body="Second body"),
    ]
    assert get_type_hints(load_documents)["return"] == list[Document]
