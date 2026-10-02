from __future__ import annotations

from dataclasses import dataclass, field

from minisearch.inverted_index import InvertedIndex


@dataclass(slots=True)
class _TrieNode:
    children: dict[str, _TrieNode] = field(default_factory=dict)
    is_word: bool = False


class Trie:
    def __init__(self) -> None:
        self._root = _TrieNode()

    @classmethod
    def from_index(cls, index: InvertedIndex) -> Trie:
        trie = cls()
        for term in index.terms():
            trie.insert(term)
        return trie

    def insert(self, word: str) -> None:
        node = self._root
        for character in word:
            node = node.children.setdefault(character, _TrieNode())
        node.is_word = True

    def contains(self, word: str) -> bool:
        node = self._find_node(word)
        return node is not None and node.is_word

    def with_prefix(self, prefix: str, limit: int = 10) -> list[str]:
        if limit <= 0:
            return []

        prefix_node = self._find_node(prefix)
        if prefix_node is None:
            return []

        matches: list[str] = []
        pending = [(prefix_node, prefix)]
        while pending and len(matches) < limit:
            node, word = pending.pop()
            if node.is_word:
                matches.append(word)
            for character in sorted(node.children, reverse=True):
                pending.append((node.children[character], word + character))

        return matches

    def _find_node(self, prefix: str) -> _TrieNode | None:
        node = self._root
        for character in prefix:
            child = node.children.get(character)
            if child is None:
                return None
            node = child
        return node
