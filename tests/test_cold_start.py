from collections.abc import Sequence

import pytest

from minisearch.document import Document
from minisearch.inverted_index import InvertedIndex
from minisearch.matrix_factorization import MatrixFactorization
from minisearch.recommender import Recommender
from minisearch.recommender_evaluation import (
    PopularityRecommender,
    evaluate_history_strategy_precision,
)
from minisearch.synthetic import Interaction
from minisearch.tokenizer import Tokenizer
from minisearch.vectors import TFIDFDocumentVectors


@pytest.fixture
def recommender() -> Recommender:
    documents = [
        Document(id="d1", title="red apple fruit", body="sweet"),
        Document(id="d2", title="red apple fruit sweet", body="ripe"),
        Document(id="d3", title="ocean water deep", body="blue"),
        Document(id="d4", title="green leaf plant", body=""),
        Document(id="d5", title="grain field wheat", body=""),
        Document(id="d6", title="stone mountain", body=""),
        Document(id="d7", title="red fruit orchard", body=""),
        Document(id="d8", title="river water", body=""),
    ]
    index = InvertedIndex(tokenizer=Tokenizer(stopwords=()))
    for document in documents:
        index.add_document(document)
    vectors = TFIDFDocumentVectors.from_index(index)
    document_ids = [document.id for document in documents]
    popularity_training = [
        Interaction(f"global-{user}", "d3", 5) for user in range(8)
    ] + [Interaction(f"global-{user}", "d2", 5) for user in range(2)]
    model_training = popularity_training + [
        Interaction("u1", "d1", 5),
        Interaction("u1", "d3", 2),
        Interaction("u1", "d4", 2),
        Interaction("u1", "d5", 3),
        Interaction("u1", "d6", 2),
    ]
    model = MatrixFactorization(n_factors=3, epochs=3, seed=37).fit(
        model_training, user_ids=["u1"], item_ids=document_ids
    )
    popularity = PopularityRecommender(popularity_training, document_ids)
    return Recommender(vectors, document_ids, popularity, model)


def test_cold_start_routes_by_history_length(recommender: Recommender) -> None:
    assert recommender.strategy_for("u1", []) == "popularity"
    assert recommender.strategy_for("u1", [Interaction("u1", "d1", 5)]) == "content"
    disliked_history = [Interaction("u1", f"d{index}", 2) for index in range(1, 5)]
    assert recommender.strategy_for("u1", disliked_history) == "popularity"
    five_ratings = [Interaction("u1", f"d{index}", 3) for index in range(1, 6)]
    assert recommender.strategy_for("u1", five_ratings) == "matrix_factorization"


def test_zero_history_uses_global_popularity_and_excludes_seen(
    recommender: Recommender,
) -> None:
    assert recommender.recommend("u1", [], k=2)[0] == "d3"
    assert "d3" not in recommender.recommend("u1", [Interaction("u1", "d3", 5)], k=10)


def test_liked_history_uses_content_similarity(recommender: Recommender) -> None:
    recommendations = recommender.recommend("u1", [Interaction("u1", "d1", 5)], k=3)

    assert recommendations[0] == "d2"
    assert "d1" not in recommendations


def test_history_without_liked_items_falls_back_to_popularity(
    recommender: Recommender,
) -> None:
    history = [Interaction("u1", "d1", 2)]

    assert recommender.strategy_for("u1", history) == "popularity"
    assert recommender.recommend("u1", history, k=2)[0] == "d3"


def test_matrix_factorization_strategy_excludes_all_seen_items(
    recommender: Recommender,
) -> None:
    history = [
        Interaction("u1", "d1", 5),
        Interaction("u1", "d3", 2),
        Interaction("u1", "d4", 2),
        Interaction("u1", "d5", 3),
        Interaction("u1", "d6", 2),
    ]

    assert recommender.strategy_for("u1", history) == "matrix_factorization"
    recommendations = set(recommender.recommend("u1", history))
    assert not ({item.doc_id for item in history} & recommendations)


class FixedColdStart:
    def strategy_for(self, user_id: str, history: Sequence[Interaction]) -> str:
        del user_id
        return "popularity" if not history else "content"

    def recommend(
        self, user_id: str, history: Sequence[Interaction], k: int = 10
    ) -> list[str]:
        del user_id, history
        return ["d1", "d2"][:k]


def test_history_evaluator_reports_precision_by_selected_strategy() -> None:
    recommender = FixedColdStart()
    histories = {"u1": [], "u2": [Interaction("u2", "d3", 3)]}
    heldout = {
        "u1": [Interaction("u1", "d1", 5)],
        "u2": [Interaction("u2", "d2", 4)],
    }

    precision_by_strategy = evaluate_history_strategy_precision(
        recommender, histories, heldout, k=2
    )

    assert precision_by_strategy == pytest.approx({"popularity": 0.5, "content": 0.5})
