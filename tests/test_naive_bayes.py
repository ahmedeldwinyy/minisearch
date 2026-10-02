import pytest

from minisearch.document import Document
from minisearch.naive_bayes import (
    MultinomialNaiveBayes,
    evaluate_topic_classifier,
)
from minisearch.tokenizer import Tokenizer


def test_predict_proba_matches_hand_computed_laplace_probabilities() -> None:
    classifier = MultinomialNaiveBayes(tokenizer=Tokenizer(stopwords=()))
    classifier.fit(
        [
            Document(id="apple", title="apple apple", body=""),
            Document(id="banana", title="banana", body=""),
        ],
        [0, 1],
    )

    probabilities = classifier.predict_proba("apple")

    # P(apple|0)=3/5 and P(apple|1)=1/4 with equal priors; normalize 3/10,1/8.
    assert probabilities[0] == pytest.approx(12 / 17)
    assert probabilities[1] == pytest.approx(5 / 17)


def test_unseen_words_are_smoothed_in_log_space() -> None:
    classifier = MultinomialNaiveBayes(tokenizer=Tokenizer(stopwords=()))
    classifier.fit(
        [
            Document(id="a", title="apple", body=""),
            Document(id="b", title="banana banana", body=""),
        ],
        [0, 1],
    )

    probabilities = classifier.predict_proba("unseenword unseenword")

    assert set(probabilities) == {0, 1}
    assert all(0 < probability < 1 for probability in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_predict_proba_is_normalized_for_long_text() -> None:
    classifier = MultinomialNaiveBayes(tokenizer=Tokenizer(stopwords=()))
    classifier.fit(
        [
            Document(id="a", title="apple pear fruit", body=""),
            Document(id="b", title="banana citrus", body=""),
        ],
        [0, 1],
    )

    probabilities = classifier.predict_proba("apple " * 500 + "unknown " * 300)

    assert all(0 <= probability <= 1 for probability in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_held_out_accuracy_is_compared_with_training_majority() -> None:
    training_documents = [
        Document(id="a1", title="apple apple", body=""),
        Document(id="a2", title="apple orange", body=""),
        Document(id="b1", title="banana banana", body=""),
        Document(id="b2", title="banana pear", body=""),
    ]
    testing_documents = [
        Document(id="a-test", title="apple", body=""),
        Document(id="b-test", title="banana", body=""),
    ]

    result = evaluate_topic_classifier(
        training_documents,
        [0, 0, 1, 1],
        testing_documents,
        [0, 1],
        tokenizer=Tokenizer(stopwords=()),
    )

    assert result.accuracy == 1.0
    assert result.majority_baseline_accuracy == 0.5
