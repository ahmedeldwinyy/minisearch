from __future__ import annotations

import re
from dataclasses import dataclass

from minisearch.boolean_search import and_postings, not_postings, or_postings
from minisearch.inverted_index import InvertedIndex

QUERY_PART_PATTERN = re.compile(r"\(|\)|[^\s()]+")
OPERATORS = frozenset({"AND", "OR", "NOT"})


class QueryParseError(ValueError):
    pass


@dataclass(frozen=True)
class TermNode:
    term: str


@dataclass(frozen=True)
class NotNode:
    operand: QueryNode


@dataclass(frozen=True)
class AndNode:
    left: QueryNode
    right: QueryNode


@dataclass(frozen=True)
class OrNode:
    left: QueryNode
    right: QueryNode


QueryNode = TermNode | NotNode | AndNode | OrNode


class QueryParser:
    def __init__(self, index: InvertedIndex) -> None:
        self._index = index
        self._tokenizer = index.tokenizer
        self._tokens: list[str] = []
        self._position = 0

    def search(self, query: str) -> list[str]:
        self._tokens = self._lex(query)
        self._position = 0
        if not self._tokens:
            raise QueryParseError("empty query")

        expression = self._parse_or()
        token = self._peek()
        if token == ")":
            raise QueryParseError("unbalanced parentheses")
        if token is not None:
            raise QueryParseError(f"unexpected token {token!r}")
        return self._evaluate(expression)

    def _lex(self, query: str) -> list[str]:
        tokens: list[str] = []
        for part in QUERY_PART_PATTERN.findall(query):
            if part in {"(", ")"}:
                tokens.append(part)
            elif part.upper() in OPERATORS:
                tokens.append(part.upper())
            else:
                tokens.extend(self._tokenizer.tokenize(part))
        return tokens

    def _parse_or(self) -> QueryNode:
        expression = self._parse_and()
        while self._match("OR"):
            if not self._starts_unary():
                raise QueryParseError("missing right operand for OR")
            expression = OrNode(expression, self._parse_and())
        return expression

    def _parse_and(self) -> QueryNode:
        expression = self._parse_not()
        while True:
            if self._match("AND"):
                if not self._starts_unary():
                    raise QueryParseError("missing right operand for AND")
                expression = AndNode(expression, self._parse_not())
            elif self._peek() == "NOT":
                expression = AndNode(expression, self._parse_not())
            else:
                return expression

    def _parse_not(self) -> QueryNode:
        if self._match("NOT"):
            if not self._starts_unary():
                raise QueryParseError("missing right operand for NOT")
            return NotNode(self._parse_not())
        return self._parse_primary()

    def _parse_primary(self) -> QueryNode:
        token = self._peek()
        if token is None:
            raise QueryParseError("missing operand")
        if token in {"AND", "OR"}:
            raise QueryParseError(f"missing left operand for {token}")
        if token == ")":
            raise QueryParseError("unbalanced parentheses")
        if token == "(":
            self._position += 1
            if self._peek() == ")":
                raise QueryParseError("missing operand inside parentheses")
            expression = self._parse_or()
            if not self._match(")"):
                raise QueryParseError("unbalanced parentheses")
            return expression

        self._position += 1
        return TermNode(token)

    def _starts_unary(self) -> bool:
        token = self._peek()
        return token is not None and token not in {"AND", "OR", ")"}

    def _match(self, token: str) -> bool:
        if self._peek() != token:
            return False
        self._position += 1
        return True

    def _peek(self) -> str | None:
        if self._position >= len(self._tokens):
            return None
        return self._tokens[self._position]

    def _evaluate(self, expression: QueryNode) -> list[str]:
        if isinstance(expression, TermNode):
            return [doc_id for doc_id, _ in self._index.postings(expression.term)]
        if isinstance(expression, NotNode):
            return not_postings(
                self._evaluate(expression.operand), self._index.document_ids()
            )
        if isinstance(expression, AndNode):
            return and_postings(
                self._evaluate(expression.left), self._evaluate(expression.right)
            )
        return or_postings(
            self._evaluate(expression.left), self._evaluate(expression.right)
        )
