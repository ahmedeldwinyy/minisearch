from dataclasses import FrozenInstanceError
from typing import get_type_hints

import pytest

from minisearch.document import Document


def test_document_has_typed_fields() -> None:
    document = Document(id="doc-1", title="A title", body="Document body")

    assert document.id == "doc-1"
    assert document.title == "A title"
    assert document.body == "Document body"
    assert get_type_hints(Document) == {"id": str, "title": str, "body": str}


def test_document_is_immutable() -> None:
    document = Document(id="doc-1", title="A title", body="Document body")
    attribute_name = "title"

    with pytest.raises(FrozenInstanceError):
        setattr(document, attribute_name, "Updated title")
