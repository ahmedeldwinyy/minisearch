from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from minisearch.synthetic import Interaction

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
GradientMap = dict[str, FloatArray]


class MatrixFactorization:
    def __init__(
        self,
        n_factors: int = 20,
        lr: float = 0.01,
        reg: float = 0.02,
        epochs: int = 20,
        batch_size: int = 256,
        seed: int = 42,
    ) -> None:
        if n_factors < 1:
            raise ValueError("n_factors must be positive")
        if lr <= 0 or reg < 0:
            raise ValueError("lr must be positive and reg cannot be negative")
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")

        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.user_to_index: dict[str, int] = {}
        self.item_to_index: dict[str, int] = {}
        self.global_mean = 0.0
        self.user_factors: FloatArray = np.zeros((0, n_factors), dtype=np.float64)
        self.item_factors: FloatArray = np.zeros((0, n_factors), dtype=np.float64)
        self.user_biases: FloatArray = np.zeros(0, dtype=np.float64)
        self.item_biases: FloatArray = np.zeros(0, dtype=np.float64)
        self.training_loss_: list[float] = []
        self.validation_loss_: list[float] = []
        self._is_fitted = False

    def fit(
        self,
        interactions: Sequence[Interaction],
        user_ids: Sequence[str] | None = None,
        item_ids: Sequence[str] | None = None,
        validation: Sequence[Interaction] | None = None,
    ) -> MatrixFactorization:
        if not interactions:
            raise ValueError("training interactions must not be empty")

        users = set(user_ids or ()) | {
            interaction.user_id for interaction in interactions
        }
        items = set(item_ids or ()) | {
            interaction.doc_id for interaction in interactions
        }
        if validation is not None:
            users.update(interaction.user_id for interaction in validation)
            items.update(interaction.doc_id for interaction in validation)
        self.user_to_index = {user_id: row for row, user_id in enumerate(sorted(users))}
        self.item_to_index = {item_id: row for row, item_id in enumerate(sorted(items))}

        user_indices, item_indices, ratings = self._encode(interactions)
        self.global_mean = float(np.mean(ratings))
        random_generator = np.random.default_rng(self.seed)
        self.user_factors = random_generator.normal(
            0.0, 0.1, size=(len(self.user_to_index), self.n_factors)
        )
        self.item_factors = random_generator.normal(
            0.0, 0.1, size=(len(self.item_to_index), self.n_factors)
        )
        self.user_biases = np.zeros(len(self.user_to_index), dtype=np.float64)
        self.item_biases = np.zeros(len(self.item_to_index), dtype=np.float64)

        if validation:
            validation_data = self._encode(validation)
        else:
            validation_data = None

        self.training_loss_ = []
        self.validation_loss_ = []
        for _ in range(self.epochs):
            shuffled_indices = random_generator.permutation(len(ratings))
            for batch_start in range(0, len(ratings), self.batch_size):
                batch_indices = shuffled_indices[
                    batch_start : batch_start + self.batch_size
                ]
                _, gradients = self._loss_and_gradients(
                    user_indices[batch_indices],
                    item_indices[batch_indices],
                    ratings[batch_indices],
                )
                self.user_factors -= self.lr * gradients["user_factors"]
                self.item_factors -= self.lr * gradients["item_factors"]
                self.user_biases -= self.lr * gradients["user_biases"]
                self.item_biases -= self.lr * gradients["item_biases"]

            self.training_loss_.append(self._loss(user_indices, item_indices, ratings))
            if validation_data is not None:
                self.validation_loss_.append(self._loss(*validation_data))

        self._is_fitted = True
        return self

    def _encode(
        self, interactions: Sequence[Interaction]
    ) -> tuple[IntArray, IntArray, FloatArray]:
        try:
            user_indices = np.fromiter(
                (self.user_to_index[item.user_id] for item in interactions),
                dtype=np.int64,
                count=len(interactions),
            )
            item_indices = np.fromiter(
                (self.item_to_index[item.doc_id] for item in interactions),
                dtype=np.int64,
                count=len(interactions),
            )
        except KeyError as error:
            raise ValueError(
                f"Unknown user or document ID: {error.args[0]!r}"
            ) from error
        ratings = np.fromiter(
            (item.rating for item in interactions),
            dtype=np.float64,
            count=len(interactions),
        )
        return user_indices, item_indices, ratings

    def _loss_and_gradients(
        self, user_indices: IntArray, item_indices: IntArray, ratings: FloatArray
    ) -> tuple[float, GradientMap]:
        sample_count = len(ratings)
        if sample_count == 0:
            raise ValueError("gradient batch must not be empty")

        user_vectors = self.user_factors[user_indices]
        item_vectors = self.item_factors[item_indices]
        predictions = (
            self.global_mean
            + self.user_biases[user_indices]
            + self.item_biases[item_indices]
            + np.sum(user_vectors * item_vectors, axis=1)
        )
        errors = predictions - ratings
        inverse_count = 1.0 / sample_count
        regularization = self.reg / (2 * sample_count)
        loss = 0.5 * float(np.mean(errors**2)) + regularization * float(
            np.sum(user_vectors**2)
            + np.sum(item_vectors**2)
            + np.sum(self.user_biases[user_indices] ** 2)
            + np.sum(self.item_biases[item_indices] ** 2)
        )

        user_factor_gradient = np.zeros_like(self.user_factors)
        item_factor_gradient = np.zeros_like(self.item_factors)
        user_bias_gradient = np.zeros_like(self.user_biases)
        item_bias_gradient = np.zeros_like(self.item_biases)
        np.add.at(
            user_factor_gradient,
            user_indices,
            (errors[:, np.newaxis] * item_vectors + self.reg * user_vectors)
            * inverse_count,
        )
        np.add.at(
            item_factor_gradient,
            item_indices,
            (errors[:, np.newaxis] * user_vectors + self.reg * item_vectors)
            * inverse_count,
        )
        np.add.at(
            user_bias_gradient,
            user_indices,
            (errors + self.reg * self.user_biases[user_indices]) * inverse_count,
        )
        np.add.at(
            item_bias_gradient,
            item_indices,
            (errors + self.reg * self.item_biases[item_indices]) * inverse_count,
        )
        return loss, {
            "user_factors": user_factor_gradient,
            "item_factors": item_factor_gradient,
            "user_biases": user_bias_gradient,
            "item_biases": item_bias_gradient,
        }

    def _loss(
        self, user_indices: IntArray, item_indices: IntArray, ratings: FloatArray
    ) -> float:
        loss, _ = self._loss_and_gradients(user_indices, item_indices, ratings)
        return loss

    def predict_raw(self, user_id: str, doc_id: str) -> float:
        self._check_fitted()
        try:
            user_index = self.user_to_index[user_id]
            item_index = self.item_to_index[doc_id]
        except KeyError as error:
            raise KeyError(f"Unknown user or document ID: {error.args[0]!r}") from error
        return float(
            self.global_mean
            + self.user_biases[user_index]
            + self.item_biases[item_index]
            + np.dot(self.user_factors[user_index], self.item_factors[item_index])
        )

    def predict(self, user_id: str, doc_id: str) -> float:
        return float(np.clip(self.predict_raw(user_id, doc_id), 1.0, 5.0))

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("MatrixFactorization must be fitted before prediction")
