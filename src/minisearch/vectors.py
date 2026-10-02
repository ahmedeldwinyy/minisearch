from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np
from scipy.sparse import csr_matrix, diags

from minisearch.inverted_index import InvertedIndex
from minisearch.rankers import top_k


@dataclass(frozen=True)
class TFIDFDocumentVectors:
    matrix: csr_matrix
    doc_ids: tuple[str, ...]
    terms: tuple[str, ...]
    doc_to_row: dict[str, int]
    term_to_column: dict[str, int]

    @classmethod
    def from_index(cls, index: InvertedIndex) -> TFIDFDocumentVectors:
        doc_ids = tuple(index.document_ids())
        terms = tuple(index.terms())
        doc_to_row = {doc_id: row for row, doc_id in enumerate(doc_ids)}
        term_to_column = {term: column for column, term in enumerate(terms)}

        rows: list[int] = []
        columns: list[int] = []
        weights: list[float] = []
        document_count = index.num_docs

        for term, column in term_to_column.items():
            postings = index.postings(term)
            document_frequency = len(postings)
            inverse_document_frequency = log(document_count / document_frequency)
            for doc_id, frequency in postings:
                rows.append(doc_to_row[doc_id])
                columns.append(column)
                weights.append((1 + log(frequency)) * inverse_document_frequency)

        matrix = csr_matrix(
            (weights, (rows, columns)),
            shape=(len(doc_ids), len(terms)),
            dtype=np.float64,
        )
        row_norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        inverse_norms = np.zeros_like(row_norms)
        np.divide(1.0, row_norms, out=inverse_norms, where=row_norms > 0)
        normalized_matrix = (diags(inverse_norms) @ matrix).tocsr()
        normalized_matrix.eliminate_zeros()

        return cls(
            matrix=normalized_matrix,
            doc_ids=doc_ids,
            terms=terms,
            doc_to_row=doc_to_row,
            term_to_column=term_to_column,
        )

    def more_like_this(self, doc_id: str, k: int = 10) -> list[tuple[str, float]]:
        row = self._get_document_row(doc_id)
        if k <= 0:
            return []

        similarities = (self.matrix.getrow(row) @ self.matrix.T).toarray().ravel()
        scores = {
            candidate_id: float(similarities[candidate_row])
            for candidate_row, candidate_id in enumerate(self.doc_ids)
            if candidate_row != row
        }
        return top_k(scores, k)

    def more_like_this_loop(self, doc_id: str, k: int = 10) -> list[tuple[str, float]]:
        row = self._get_document_row(doc_id)
        if k <= 0:
            return []

        query_start = self.matrix.indptr[row]
        query_end = self.matrix.indptr[row + 1]
        query_columns = self.matrix.indices[query_start:query_end]
        query_values = self.matrix.data[query_start:query_end]
        scores: dict[str, float] = {}

        for candidate_row, candidate_id in enumerate(self.doc_ids):
            if candidate_row == row:
                continue

            candidate_start = self.matrix.indptr[candidate_row]
            candidate_end = self.matrix.indptr[candidate_row + 1]
            candidate_columns = self.matrix.indices[candidate_start:candidate_end]
            candidate_values = self.matrix.data[candidate_start:candidate_end]
            candidate_vector = dict(
                zip(candidate_columns, candidate_values, strict=True)
            )
            score = sum(
                float(value) * float(candidate_vector.get(column, 0.0))
                for column, value in zip(query_columns, query_values, strict=True)
            )
            scores[candidate_id] = score

        return top_k(scores, k)

    def _get_document_row(self, doc_id: str) -> int:
        try:
            return self.doc_to_row[doc_id]
        except KeyError as error:
            raise KeyError(f"Unknown document ID {doc_id!r}") from error
