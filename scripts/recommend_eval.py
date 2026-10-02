from __future__ import annotations

import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np

from minisearch.document import Document
from minisearch.inverted_index import InvertedIndex
from minisearch.loader import load_documents
from minisearch.matrix_factorization import MatrixFactorization
from minisearch.naive_bayes import evaluate_topic_classifier
from minisearch.recommender import Recommender
from minisearch.recommender_evaluation import (
    PopularityRecommender,
    RecommenderMetrics,
    evaluate_history_strategy_precision,
    evaluate_recommenders,
    split_interactions_by_user,
)
from minisearch.synthetic import (
    Interaction,
    SphericalKMeans,
    generate_synthetic_interactions,
)
from minisearch.vectors import TFIDFDocumentVectors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ARCHIVE = PROJECT_ROOT / "data" / "raw" / "scifact.zip"
RANDOM_SEED = 2026
HISTORY_SIZES = (0, 1, 3, 5, 20)


def _new_matrix_factorization(seed: int) -> MatrixFactorization:
    return MatrixFactorization(
        n_factors=24,
        lr=0.015,
        reg=0.03,
        epochs=12,
        batch_size=512,
        seed=seed,
    )


@dataclass(frozen=True)
class ColdStartRow:
    history_size: int
    strategy: str
    precision_at_10: float


def format_recommender_results_table(
    results: Mapping[str, RecommenderMetrics],
) -> str:
    rows = [
        "| Strategy | Test RMSE | Precision@10 | Recall@10 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for strategy, metrics in results.items():
        rmse = "N/A" if metrics.rmse is None else f"{metrics.rmse:.3f}"
        rows.append(
            f"| {strategy} | {rmse} | {metrics.precision_at_k:.3f} | "
            f"{metrics.recall_at_k:.3f} |"
        )
    return "\n".join(rows)


def format_cold_start_table(rows: Sequence[ColdStartRow]) -> str:
    table = [
        "| History ratings | Strategy | Precision@10 |",
        "| ---: | --- | ---: |",
    ]
    table.extend(
        f"| {row.history_size} | {row.strategy} | {row.precision_at_10:.3f} |"
        for row in rows
    )
    return "\n".join(table)


def _load_corpus(archive_path: Path) -> list[Document]:
    with ZipFile(archive_path) as archive:
        with TemporaryDirectory(prefix="minisearch-recommend-") as directory:
            corpus_path = Path(directory) / "corpus.jsonl"
            with (
                archive.open("scifact/corpus.jsonl") as source,
                corpus_path.open("wb") as target,
            ):
                shutil.copyfileobj(source, target)
            return load_documents(corpus_path)


def _build_index(documents: Sequence[Document]) -> InvertedIndex:
    index = InvertedIndex()
    for document in documents:
        index.add_document(document)
    return index


def _make_history_prefixes(
    user_ids: Sequence[str],
    training_by_user: Mapping[str, Sequence[Interaction]],
    seed: int,
    max_history: int,
) -> dict[str, tuple[Interaction, ...]]:
    random_generator = np.random.default_rng(seed)
    prefixes: dict[str, tuple[Interaction, ...]] = {}
    for user_id in user_ids:
        interactions = list(training_by_user.get(user_id, ()))
        order = random_generator.permutation(len(interactions))
        prefixes[user_id] = tuple(
            interactions[int(index)] for index in order[:max_history]
        )
    return prefixes


def _cold_start_rows(
    vectors: TFIDFDocumentVectors,
    document_ids: list[str],
    training: list[Interaction],
    testing: list[Interaction],
    seed: int,
) -> list[ColdStartRow]:
    training_by_user: dict[str, list[Interaction]] = defaultdict(list)
    testing_by_user: dict[str, list[Interaction]] = defaultdict(list)
    for interaction in training:
        training_by_user[interaction.user_id].append(interaction)
    for interaction in testing:
        testing_by_user[interaction.user_id].append(interaction)

    user_ids = sorted(testing_by_user)
    if len(user_ids) < 2:
        raise ValueError("cold-start evaluation requires at least two users")
    random_generator = np.random.default_rng(seed)
    evaluation_user_count = max(1, round(len(user_ids) * 0.2))
    evaluation_users = sorted(
        random_generator.choice(user_ids, size=evaluation_user_count, replace=False)
    )
    evaluation_user_set = set(evaluation_users)
    background_training = [
        interaction
        for interaction in training
        if interaction.user_id not in evaluation_user_set
    ]
    history_prefixes = _make_history_prefixes(
        evaluation_users, training_by_user, seed + 1, max(HISTORY_SIZES)
    )
    heldout = {user_id: testing_by_user[user_id] for user_id in evaluation_users}
    rows: list[ColdStartRow] = []

    for history_size in HISTORY_SIZES:
        histories = {
            user_id: history_prefixes[user_id][:history_size]
            for user_id in evaluation_users
        }
        visible_training = [
            *background_training,
            *(interaction for history in histories.values() for interaction in history),
        ]
        if not visible_training:
            raise ValueError("cold-start training data must not be empty")

        model = _new_matrix_factorization(seed + history_size)
        if history_size >= 5:
            model.fit(
                visible_training,
                user_ids=evaluation_users,
                item_ids=document_ids,
            )
        popularity = PopularityRecommender(visible_training, document_ids)
        recommender = Recommender(vectors, document_ids, popularity, model)
        strategy_precision = evaluate_history_strategy_precision(
            recommender, histories, heldout, k=10
        )
        strategy_counts: dict[str, int] = defaultdict(int)
        for user_id, history in histories.items():
            strategy_counts[recommender.strategy_for(user_id, history)] += 1
        rows.extend(
            ColdStartRow(
                history_size=history_size,
                strategy=strategy,
                precision_at_10=precision,
            )
            for strategy, precision in sorted(strategy_precision.items())
            if strategy_counts[strategy] > 0
        )
    return rows


def _save_loss_plot(model: MatrixFactorization, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
    axis.plot(
        range(1, len(model.training_loss_) + 1),
        model.training_loss_,
        label="Train",
    )
    axis.plot(
        range(1, len(model.validation_loss_) + 1),
        model.validation_loss_,
        label="Validation",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Regularized squared loss")
    axis.set_title("Synthetic interaction matrix factorization loss")
    axis.legend()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _save_history_plot(rows: Sequence[ColdStartRow], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for strategy in ("popularity", "content", "matrix_factorization"):
        strategy_rows = [row for row in rows if row.strategy == strategy]
        if not strategy_rows:
            continue
        axis.plot(
            [row.history_size for row in strategy_rows],
            [row.precision_at_10 for row in strategy_rows],
            marker="o",
            label=strategy,
        )
    axis.set_xticks(HISTORY_SIZES)
    axis.set_xlabel("Revealed training ratings per user")
    axis.set_ylabel("Precision@10")
    axis.set_title("Cold-start precision by available history")
    axis.legend()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _update_report(
    report_path: Path,
    recommender_results: Mapping[str, RecommenderMetrics],
    cold_start_rows: Sequence[ColdStartRow],
    user_count: int,
    document_count: int,
    topic_accuracy: float,
    topic_majority_accuracy: float,
) -> None:
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    section_marker = "## Synthetic Recommendation Evaluation"
    if section_marker in existing:
        existing = existing.split(section_marker, maxsplit=1)[0].rstrip()
    if not existing:
        existing = "# MiniSearch Recommender\n\n"
    report = "\n".join(
        [
            existing.rstrip(),
            "",
            section_marker,
            "",
            "SciFact contains documents but no user-rating data. All user IDs, "
            "topic preferences, and ratings below are synthetic, generated with "
            f"fixed seeds from {document_count:,} documents; results validate the "
            "code and math, not real-world recommendation quality.",
            "",
            f"The matrix-factorization and baseline comparison uses the held-out "
            f"ratings of {user_count:,} synthetic users.",
            "",
            "### Matrix Factorization vs Baselines",
            "",
            format_recommender_results_table(recommender_results),
            "",
            "### Cold-Start Precision@10",
            "",
            format_cold_start_table(cold_start_rows),
            "",
            "Each user is held out from global training and reveals only the "
            "listed number of their training ratings. The strategy column shows "
            "the policy actually used; a 1–4 rating user without a liked item "
            "falls back to popularity.",
            "",
            "### Plots",
            "",
            "![Matrix factorization training and validation loss](mf-loss.png)",
            "",
            "![Precision@10 by revealed history size](cold-start-precision.png)",
            "",
            "Naive Bayes accuracy is measured against D3 cluster labels. It "
            f"reached {topic_accuracy:.3f} accuracy versus the "
            f"{topic_majority_accuracy:.3f} majority-class baseline on a held-out "
            "20% of documents. These generated labels demonstrate classifier "
            "behavior, not real query intent.",
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")


def run_recommendation_evaluation(
    archive_path: Path = DEFAULT_DATASET_ARCHIVE,
    output_directory: Path = PROJECT_ROOT / "docs",
    seed: int = RANDOM_SEED,
) -> tuple[dict[str, RecommenderMetrics], list[ColdStartRow]]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"SciFact archive not found at {archive_path}")

    documents = _load_corpus(archive_path)
    document_ids = [document.id for document in documents]
    index = _build_index(documents)
    vectors = TFIDFDocumentVectors.from_index(index)
    document_topics = SphericalKMeans(
        n_clusters=20, max_iter=20, seed=seed
    ).fit_predict(vectors.matrix)
    topic_split_rng = np.random.default_rng(seed + 7)
    topic_permutation = topic_split_rng.permutation(len(documents))
    topic_test_count = max(1, round(len(documents) * 0.2))
    topic_test_indices = topic_permutation[:topic_test_count]
    topic_train_indices = topic_permutation[topic_test_count:]
    topic_evaluation = evaluate_topic_classifier(
        [documents[int(index)] for index in topic_train_indices],
        [int(document_topics[index]) for index in topic_train_indices],
        [documents[int(index)] for index in topic_test_indices],
        [int(document_topics[index]) for index in topic_test_indices],
        tokenizer=index.tokenizer,
    )
    synthetic_data = generate_synthetic_interactions(
        document_ids, document_topics, n_users=1000, ratings_per_user=40, seed=seed + 1
    )
    interactions = list(synthetic_data.interactions)
    training, testing = split_interactions_by_user(interactions, seed=seed + 2)
    user_ids = sorted(synthetic_data.user_topics)
    loss_model = _new_matrix_factorization(seed + 3).fit(
        training,
        user_ids=user_ids,
        item_ids=document_ids,
        validation=testing,
    )
    recommender_results = evaluate_recommenders(
        _new_matrix_factorization(seed + 4),
        training,
        testing,
        document_ids,
        k=10,
        seed=seed + 5,
    )
    cold_start_rows = _cold_start_rows(
        vectors, document_ids, training, testing, seed + 6
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    _save_loss_plot(loss_model, output_directory / "mf-loss.png")
    _save_history_plot(cold_start_rows, output_directory / "cold-start-precision.png")
    _update_report(
        output_directory / "RECOMMENDER.md",
        recommender_results,
        cold_start_rows,
        user_count=len(user_ids),
        document_count=len(document_ids),
        topic_accuracy=topic_evaluation.accuracy,
        topic_majority_accuracy=topic_evaluation.majority_baseline_accuracy,
    )
    return recommender_results, cold_start_rows


def main() -> None:
    results, cold_start_rows = run_recommendation_evaluation()
    print("Synthetic MF vs baseline results:")
    print(format_recommender_results_table(results))
    print("\nCold-start results:")
    print(format_cold_start_table(cold_start_rows))
    print(f"\nArtifacts saved under {PROJECT_ROOT / 'docs'}")


if __name__ == "__main__":
    main()
