import pytest

from minisearch.matrix_factorization import MatrixFactorization
from minisearch.recommender_evaluation import (
    PopularityRecommender,
    RandomRecommender,
    evaluate_recommenders,
    precision_recall_at_k,
    rmse,
    split_interactions_by_user,
)
from minisearch.synthetic import Interaction


def test_rmse_matches_hand_calculation() -> None:
    # Squared errors are 1 and 0: RMSE = sqrt((1 + 0) / 2).
    assert rmse([3.0, 4.0], [2.0, 4.0]) == pytest.approx((0.5) ** 0.5)


def test_precision_and_recall_at_k_match_hand_calculation() -> None:
    # One relevant result in the top two gives P@2 = 1/2 and R@2 = 1/2.
    precision, recall = precision_recall_at_k(
        ["relevant-1", "other", "relevant-2"], {"relevant-1", "relevant-2"}, 2
    )

    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)


def test_split_is_seeded_and_holds_out_each_user_ratings() -> None:
    interactions = [
        Interaction("u1", f"d{index}", index % 5 + 1) for index in range(5)
    ] + [Interaction("u2", f"x{index}", 4) for index in range(4)]

    train, test = split_interactions_by_user(interactions, seed=17)
    second_train, second_test = split_interactions_by_user(interactions, seed=17)

    assert train == second_train
    assert test == second_test
    assert [sum(item.user_id == user for item in test) for user in ("u1", "u2")] == [
        1,
        1,
    ]
    assert not (
        {(item.user_id, item.doc_id) for item in train}
        & {(item.user_id, item.doc_id) for item in test}
    )


def test_baselines_never_recommend_training_items() -> None:
    training = [
        Interaction("u1", "d1", 5),
        Interaction("u1", "d2", 4),
        Interaction("u2", "d1", 5),
        Interaction("u2", "d3", 4),
    ]
    seen = {item.doc_id for item in training if item.user_id == "u1"}
    popularity = PopularityRecommender(training, ["d1", "d2", "d3", "d4"])
    random_recommender = RandomRecommender(["d1", "d2", "d3", "d4"], seed=19)

    assert not (set(popularity.recommend("u1", seen, k=10)) & seen)
    assert not (set(random_recommender.recommend("u1", seen, k=10)) & seen)


def test_evaluate_recommenders_compares_mf_and_baselines() -> None:
    training = [
        Interaction("u1", "d1", 5),
        Interaction("u1", "d2", 4),
        Interaction("u2", "d2", 5),
        Interaction("u2", "d3", 2),
        Interaction("u3", "d1", 4),
        Interaction("u3", "d3", 1),
    ]
    testing = [
        Interaction("u1", "d3", 5),
        Interaction("u2", "d1", 4),
        Interaction("u3", "d2", 5),
    ]
    model = MatrixFactorization(
        n_factors=3, lr=0.02, reg=0.01, epochs=10, batch_size=3, seed=29
    )

    results = evaluate_recommenders(
        model, training, testing, ["d1", "d2", "d3", "d4"], k=2, seed=31
    )

    assert set(results) == {"matrix_factorization", "popularity", "random"}
    assert results["matrix_factorization"].rmse is not None
    assert results["popularity"].rmse is None
    assert results["random"].rmse is None
