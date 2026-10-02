from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np
from scipy.sparse import csr_matrix, diags

from minisearch.inverted_index import InvertedIndex


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
