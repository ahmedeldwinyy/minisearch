from minisearch.decorators import lru_cache, timed


def test_lru_cache_records_hits_and_misses() -> None:
    calls: list[int] = []

    @lru_cache(max_size=2)
    def double(value: int) -> int:
        calls.append(value)
        return value * 2

    assert double(3) == 6
    assert double(3) == 6
    assert calls == [3]
    assert double.hits == 1
    assert double.misses == 1


def test_lru_cache_evicts_least_recently_used_entry() -> None:
    calls: list[str] = []

    @lru_cache(max_size=2)
    def uppercase(value: str) -> str:
        calls.append(value)
        return value.upper()

    uppercase("a")
    uppercase("b")
    uppercase("a")
    uppercase("c")
    uppercase("b")

    assert calls == ["a", "b", "c", "b"]
    assert uppercase.hits == 1
    assert uppercase.misses == 4


def test_lru_cache_clear_resets_entries_and_counters() -> None:
    calls: list[int] = []

    @lru_cache(max_size=2)
    def identity(value: int) -> int:
        calls.append(value)
        return value

    identity(1)
    identity(1)
    identity.clear()
    identity(1)

    assert calls == [1, 1]
    assert identity.hits == 0
    assert identity.misses == 1


def test_timed_returns_result_and_preserves_function_metadata() -> None:
    @timed
    def describe(value: int) -> str:
        """Return a label for a number."""
        return f"number {value}"

    assert describe(7) == "number 7"
    assert describe.__name__ == "describe"
    assert describe.__doc__ == "Return a label for a number."
    assert describe.last_duration is not None
    assert describe.last_duration >= 0
