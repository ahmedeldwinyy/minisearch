import json
from pathlib import Path

from minisearch.document import Document


def load_documents(path: str | Path) -> list[Document]:
    documents: list[Document] = []

    with Path(path).open(encoding="utf-8") as corpus_file:
        for line in corpus_file:
            if not line.strip():
                continue

            record = json.loads(line)
            documents.append(
                Document(
                    id=record["_id"],
                    title=record["title"],
                    body=record["text"],
                )
            )

    return documents
