import numpy as np
import pytest

from minisearch.matrix_factorization import MatrixFactorization
from minisearch.synthetic import Interaction


@pytest.fixture
def interactions() -> list[Interaction]:
    return [
        Interaction("u1", "d1", 5),
        Interaction("u1", "d2", 4),
        Interaction("u2", "d1", 4),
        Interaction("u2", "d2", 1),
    ]


def test_analytic_gradients_match_finite_differences(
    interactions: list[Interaction],
) -> None:
    model = MatrixFactorization(n_factors=2, lr=0.01, reg=0.03, epochs=3, seed=23)
    model.fit(interactions)
    model.user_factors[:] = [[0.2, -0.1], [0.4, 0.3]]
    model.item_factors[:] = [[-0.2, 0.5], [0.1, -0.3]]
    model.user_biases[:] = [0.05, -0.08]
    model.item_biases[:] = [0.07, -0.02]

    user_indices, item_indices, ratings = model._encode(interactions)
    _, gradients = model._loss_and_gradients(user_indices, item_indices, ratings)
    parameter_names = (
        "user_factors",
        "item_factors",
        "user_biases",
        "item_biases",
    )
    numerical_values: list[float] = []
    analytical_values: list[float] = []
    epsilon = 1e-6

    for name in parameter_names:
        parameter = getattr(model, name)
        analytical = gradients[name]
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + epsilon
            positive_loss = model._loss(user_indices, item_indices, ratings)
            parameter[index] = original - epsilon
            negative_loss = model._loss(user_indices, item_indices, ratings)
            parameter[index] = original
            numerical_values.append((positive_loss - negative_loss) / (2 * epsilon))
            analytical_values.append(float(analytical[index]))

    numerical = np.asarray(numerical_values)
    analytical = np.asarray(analytical_values)
    relative_error = float(np.linalg.norm(numerical - analytical)) / max(
        1.0, float(np.linalg.norm(numerical)), float(np.linalg.norm(analytical))
    )
    assert relative_error < 1e-5


def test_training_loss_decreases(interactions: list[Interaction]) -> None:
    model = MatrixFactorization(
        n_factors=4, lr=0.03, reg=0.01, epochs=30, batch_size=4, seed=7
    ).fit(interactions)

    assert model.training_loss_[0] > model.training_loss_[-1]


def test_same_seed_produces_identical_training_results(
    interactions: list[Interaction],
) -> None:
    first = MatrixFactorization(
        n_factors=3, lr=0.02, reg=0.01, epochs=8, batch_size=2, seed=31
    ).fit(interactions)
    second = MatrixFactorization(
        n_factors=3, lr=0.02, reg=0.01, epochs=8, batch_size=2, seed=31
    ).fit(interactions)

    assert first.training_loss_ == second.training_loss_
    assert first.predict_raw("u1", "d1") == second.predict_raw("u1", "d1")


def test_predictions_are_clipped_only_when_requested(
    interactions: list[Interaction],
) -> None:
    model = MatrixFactorization(n_factors=2, epochs=2, seed=5).fit(interactions)
    model.user_biases[model.user_to_index["u1"]] = 100.0

    assert model.predict_raw("u1", "d1") > 5
    assert model.predict("u1", "d1") == 5
