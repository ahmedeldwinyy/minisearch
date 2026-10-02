from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    body: str
