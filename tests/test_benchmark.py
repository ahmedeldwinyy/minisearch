from scripts.benchmark import (
    BenchmarkMeasurements,
    _select_trie_prefix,
    format_benchmark_report,
)


def test_select_trie_prefix_with_most_vocabulary_matches() -> None:
    assert _select_trie_prefix(["car", "carbon", "cart", "dog", "dove"]) == "ca"


def test_benchmark_report_includes_measurements_and_big_o() -> None:
    measurements = BenchmarkMeasurements(
        document_count=1200,
        query_count=100,
        indexing_1000_seconds=0.125,
        indexing_all_seconds=0.25,
        query_averages_seconds={
            "single term": 0.0005,
            "AND": 0.001,
            "OR": 0.0015,
            "NOT": 0.002,
            "nested": 0.003,
        },
        trie_build_seconds=0.05,
        trie_prefix="ca",
        trie_lookup_average_seconds=0.0002,
    )

    report = format_benchmark_report(measurements)

    assert "| Index build (1,000 documents) | 0.125 s |" in report
    assert "| Index build (all 1,200 documents) | 0.250 s |" in report
    assert "| Single-term query (100 runs) | 0.500 ms/query |" in report
    assert "| AND query (100 runs) | 1.000 ms/query |" in report
    assert "| OR query (100 runs) | 1.500 ms/query |" in report
    assert "| NOT query (100 runs) | 2.000 ms/query |" in report
    assert "| Nested query (100 runs) | 3.000 ms/query |" in report
    assert "| Trie prefix lookup (`ca`, 100 runs) | 0.200 ms/query |" in report
    assert "O(T) expected" in report
    assert "O(|A| + |B|)" in report
    assert "O(|A| + |U|)" in report
    assert "O(|prefix| + V log sigma)" in report
