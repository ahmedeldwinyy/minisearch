import numpy as np
from scipy.sparse import csr_matrix

from minisearch.synthetic import SphericalKMeans, generate_synthetic_interactions


def test_spherical_kmeans_is_seeded_and_groups_directions() -> None:
    matrix = csr_matrix(
        np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        )
    )

    first = SphericalKMeans(n_clusters=2, max_iter=8, seed=42).fit_predict(matrix)
    second = SphericalKMeans(n_clusters=2, max_iter=8, seed=42).fit_predict(matrix)

    assert np.array_equal(first, second)
    assert first[0] == first[1]
    assert first[2] == first[3]
    assert first[0] != first[2]


def test_synthetic_interactions_are_reproducible() -> None:
    document_ids = [f"doc-{index:02}" for index in range(30)]
    document_topics = np.repeat(np.arange(3), 10)

    first = generate_synthetic_interactions(
        document_ids, document_topics, n_users=6, ratings_per_user=8, seed=71
    )
    second = generate_synthetic_interactions(
        document_ids, document_topics, n_users=6, ratings_per_user=8, seed=71
    )

    assert first.interactions == second.interactions
    assert first.user_topics == second.user_topics
    assert np.array_equal(first.document_topics, second.document_topics)


def test_synthetic_ratings_are_bounded_unique_and_complete() -> None:
    document_ids = [f"doc-{index:02}" for index in range(30)]
    document_topics = np.repeat(np.arange(3), 10)
    dataset = generate_synthetic_interactions(
        document_ids, document_topics, n_users=6, ratings_per_user=8, seed=11
    )

    assert len(dataset.interactions) == 6 * 8
    assert all(1 <= interaction.rating <= 5 for interaction in dataset.interactions)
    user_document_pairs = {
        (interaction.user_id, interaction.doc_id)
        for interaction in dataset.interactions
    }
    assert len(user_document_pairs) == len(dataset.interactions)
    assert all(
        sum(interaction.user_id == user_id for interaction in dataset.interactions) == 8
        for user_id in dataset.user_topics
    )


def test_preferred_topic_items_have_higher_mean_rating() -> None:
    document_ids = [f"doc-{index:02}" for index in range(60)]
    document_topics = np.repeat(np.arange(3), 20)
    dataset = generate_synthetic_interactions(
        document_ids, document_topics, n_users=30, ratings_per_user=20, seed=19
    )
    topic_by_document = dict(zip(document_ids, document_topics, strict=True))
    preferred_ratings: list[int] = []
    other_ratings: list[int] = []

    for interaction in dataset.interactions:
        if (
            topic_by_document[interaction.doc_id]
            in dataset.user_topics[interaction.user_id]
        ):
            preferred_ratings.append(interaction.rating)
        else:
            other_ratings.append(interaction.rating)

    assert np.mean(preferred_ratings) > np.mean(other_ratings)
