from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import log

import numpy as np
from scipy.special import logsumexp

from minisearch.document import Document
from minisearch.tokenizer import Tokenizer


@dataclass(frozen=True)
class TopicClassifierEvaluation:
    accuracy: float
    majority_baseline_accuracy: float


class MultinomialNaiveBayes:
    def __init__(self, alpha: float = 1.0, tokenizer: Tokenizer | None = None) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.tokenizer = tokenizer if tokenizer is not None else Tokenizer()
        self.classes_: tuple[int, ...] = ()
        self.vocabulary_: frozenset[str] = frozenset()
        self.class_document_counts_: dict[int, int] = {}
        self.class_token_counts_: dict[int, int] = {}
        self.class_term_counts_: dict[int, Counter[str]] = {}
        self._fitted = False

    def fit(
        self, documents: Sequence[Document], labels: Sequence[int]
    ) -> MultinomialNaiveBayes:
        if not documents or len(documents) != len(labels):
            raise ValueError("documents and labels must have equal non-zero lengths")

        self.classes_ = tuple(sorted(set(labels)))
        self.class_document_counts_ = dict.fromkeys(self.classes_, 0)
        self.class_token_counts_ = dict.fromkeys(self.classes_, 0)
        self.class_term_counts_ = {label: Counter() for label in self.classes_}
        vocabulary: set[str] = set()

        for document, label in zip(documents, labels, strict=True):
            if label not in self.class_document_counts_:
                raise ValueError(f"Unknown class label {label!r}")
            terms = self.tokenizer.tokenize(f"{document.title} {document.body}")
            counts = Counter(terms)
            vocabulary.update(counts)
            self.class_document_counts_[label] += 1
            self.class_token_counts_[label] += sum(counts.values())
            self.class_term_counts_[label].update(counts)

        self.vocabulary_ = frozenset(vocabulary)
        self._fitted = True
        return self

    def predict_proba(self, text: str) -> dict[int, float]:
        self._check_fitted()
        terms = Counter(self.tokenizer.tokenize(text))
        total_documents = sum(self.class_document_counts_.values())
        vocabulary_size = len(self.vocabulary_) + 1
        log_scores: list[float] = []

        for label in self.classes_:
            class_documents = self.class_document_counts_[label]
            score = log(class_documents / total_documents)
            denominator = self.class_token_counts_[label] + self.alpha * vocabulary_size
            for term, frequency in terms.items():
                observed_term = term if term in self.vocabulary_ else "<OOV>"
                term_count = self.class_term_counts_[label][observed_term]
                score += frequency * log((term_count + self.alpha) / denominator)
            log_scores.append(score)

        normalizer = float(logsumexp(np.asarray(log_scores, dtype=np.float64)))
        probabilities = np.exp(np.asarray(log_scores, dtype=np.float64) - normalizer)
        return {
            label: float(probabilities[index])
            for index, label in enumerate(self.classes_)
        }

    def predict(self, text: str) -> int:
        probabilities = self.predict_proba(text)
        return max(self.classes_, key=lambda label: probabilities[label])

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("MultinomialNaiveBayes must be fitted before prediction")


def evaluate_topic_classifier(
    training_documents: Sequence[Document],
    training_labels: Sequence[int],
    testing_documents: Sequence[Document],
    testing_labels: Sequence[int],
    tokenizer: Tokenizer | None = None,
) -> TopicClassifierEvaluation:
    if not testing_documents or len(testing_documents) != len(testing_labels):
        raise ValueError(
            "testing documents and labels must have equal non-zero lengths"
        )

    classifier = MultinomialNaiveBayes(tokenizer=tokenizer).fit(
        training_documents, training_labels
    )
    predictions = [
        classifier.predict(f"{document.title} {document.body}")
        for document in testing_documents
    ]
    accuracy = sum(
        prediction == label
        for prediction, label in zip(predictions, testing_labels, strict=True)
    ) / len(testing_labels)
    training_counts = Counter(training_labels)
    majority_label = min(
        training_counts, key=lambda label: (-training_counts[label], label)
    )
    majority_accuracy = sum(label == majority_label for label in testing_labels) / len(
        testing_labels
    )
    return TopicClassifierEvaluation(
        accuracy=accuracy,
        majority_baseline_accuracy=majority_accuracy,
    )
