import pytest

from minisearch.document import Document
from minisearch.inverted_index import InvertedIndex
from minisearch.query_parser import QueryParseError, QueryParser


@pytest.fixture
def index() -> InvertedIndex:
    result = InvertedIndex()
    result.add_document(Document(id="doc-1", title="cancer gene", body=""))
    result.add_document(Document(id="doc-2", title="cancer gene mouse", body=""))
    result.add_document(Document(id="doc-3", title="gene mouse", body=""))
    result.add_document(Document(id="doc-4", title="cancer", body=""))
    result.add_document(Document(id="doc-5", title="السَّرَطَان جين", body=""))
    return result


def test_and_precedes_or(index: InvertedIndex) -> None:
    assert QueryParser(index).search("cancer OR gene AND mouse") == [
        "doc-1",
        "doc-2",
        "doc-3",
        "doc-4",
    ]


def test_nested_parentheses_and_implicit_and_not(index: InvertedIndex) -> None:
    assert QueryParser(index).search("((cancer OR gene) AND mouse) NOT cancer") == [
        "doc-3"
    ]


def test_operators_are_case_insensitive(index: InvertedIndex) -> None:
    assert QueryParser(index).search("CANCER aNd GENE") == ["doc-1", "doc-2"]


def test_unknown_terms_return_no_documents(index: InvertedIndex) -> None:
    assert QueryParser(index).search("unknown AND gene") == []


def test_query_terms_use_tokenizer_normalization(index: InvertedIndex) -> None:
    assert QueryParser(index).search("السَّرَطَان") == ["doc-5"]


def test_empty_query_raises_clear_error(index: InvertedIndex) -> None:
    with pytest.raises(QueryParseError, match="empty query"):
        QueryParser(index).search("   ")


@pytest.mark.parametrize("query", ["(cancer", "cancer)"])
def test_unbalanced_parentheses_raise_clear_error(
    index: InvertedIndex, query: str
) -> None:
    with pytest.raises(QueryParseError, match="unbalanced parentheses"):
        QueryParser(index).search(query)


@pytest.mark.parametrize(
    "query",
    ["AND cancer", "cancer AND", "cancer OR", "NOT", "cancer NOT"],
)
def test_missing_operator_operands_raise_clear_error(
    index: InvertedIndex, query: str
) -> None:
    with pytest.raises(QueryParseError, match="missing (left |right )?operand"):
        QueryParser(index).search(query)
