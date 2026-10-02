from minisearch.boolean_search import and_postings, not_postings, or_postings
from minisearch.document import Document
from minisearch.inverted_index import InvertedIndex


def test_boolean_operations_handle_empty_lists() -> None:
    assert and_postings([], []) == []
    assert or_postings([], []) == []
    assert or_postings(["doc-1"], []) == ["doc-1"]
    assert not_postings([], ["doc-1"]) == ["doc-1"]


def test_boolean_operations_handle_non_overlapping_lists() -> None:
    left = ["doc-1", "doc-3"]
    right = ["doc-2", "doc-4"]

    assert and_postings(left, right) == []
    assert or_postings(left, right) == ["doc-1", "doc-2", "doc-3", "doc-4"]


def test_boolean_operations_handle_full_overlap() -> None:
    left = ["doc-1", "doc-2"]
    right = ["doc-1", "doc-2"]

    assert and_postings(left, right) == ["doc-1", "doc-2"]
    assert or_postings(left, right) == ["doc-1", "doc-2"]


def test_boolean_merge_handles_one_much_longer_list() -> None:
    long_list = [f"doc-{index:03}" for index in range(100)]
    short_list = ["doc-050", "doc-101"]

    assert and_postings(long_list, short_list) == ["doc-050"]
    assert or_postings(long_list, short_list) == [*long_list, "doc-101"]


def test_unknown_index_term_has_empty_postings() -> None:
    index = InvertedIndex()
    index.add_document(Document(id="doc-1", title="cancer", body="gene"))
    cancer_doc_ids = [doc_id for doc_id, _ in index.postings("cancer")]
    unknown_doc_ids = [doc_id for doc_id, _ in index.postings("unknown")]

    assert index.postings("unknown") == []
    assert and_postings(cancer_doc_ids, unknown_doc_ids) == []


def test_not_postings_handles_everything_and_nothing() -> None:
    universe = ["doc-1", "doc-2", "doc-3"]

    assert not_postings(universe, universe) == []
    assert not_postings([], universe) == universe


def test_index_returns_sorted_document_id_universe() -> None:
    index = InvertedIndex()
    index.add_document(Document(id="doc-3", title="third", body=""))
    index.add_document(Document(id="doc-1", title="first", body=""))
    index.add_document(Document(id="doc-2", title="second", body=""))

    assert index.document_ids() == ["doc-1", "doc-2", "doc-3"]
