from collections import Counter

from minisearch.document import Document
from minisearch.tokenizer import Tokenizer

Posting = tuple[str, int]


class InvertedIndex:
    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self._tokenizer = tokenizer if tokenizer is not None else Tokenizer()
        self._postings: dict[str, dict[str, int]] = {}
        self._document_terms: dict[str, dict[str, int]] = {}

    def add_document(self, doc: Document) -> None:
        if doc.id in self._document_terms:
            raise ValueError(f"Document ID {doc.id!r} already exists")

        terms = Counter(self._tokenizer.tokenize(f"{doc.title} {doc.body}"))
        self._document_terms[doc.id] = dict(terms)
        for term, frequency in terms.items():
            self._postings.setdefault(term, {})[doc.id] = frequency

    def postings(self, term: str) -> list[Posting]:
        return sorted(self._postings.get(term, {}).items())

    def remove_document(self, doc_id: str) -> None:
        terms = self._document_terms.pop(doc_id, None)
        if terms is None:
            return

        for term in terms:
            term_postings = self._postings[term]
            del term_postings[doc_id]
            if not term_postings:
                del self._postings[term]

    def doc_length(self, doc_id: str) -> int:
        return sum(self._document_terms[doc_id].values())

    @property
    def num_docs(self) -> int:
        return len(self._document_terms)

    def document_ids(self) -> list[str]:
        return sorted(self._document_terms)

    def doc_freq(self, term: str) -> int:
        return len(self._postings.get(term, {}))
