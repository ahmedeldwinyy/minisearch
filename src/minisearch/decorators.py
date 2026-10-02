from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable
from functools import wraps
from time import perf_counter
from typing import ParamSpec, Protocol, TypeVar, cast

Parameters = ParamSpec("Parameters")
Result = TypeVar("Result", covariant=True)


class TimedFunction(Protocol[Parameters, Result]):
    last_duration: float | None
    __name__: str
    __doc__: str | None

    def __call__(
        self, *args: Parameters.args, **kwargs: Parameters.kwargs
    ) -> Result: ...


class CachedFunction(Protocol[Parameters, Result]):
    hits: int
    misses: int
    clear: Callable[[], None]

    def __call__(
        self, *args: Parameters.args, **kwargs: Parameters.kwargs
    ) -> Result: ...


def timed(function: Callable[Parameters, Result]) -> TimedFunction[Parameters, Result]:
    @wraps(function)
    def wrapper(*args: Parameters.args, **kwargs: Parameters.kwargs) -> Result:
        start = perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            decorated.last_duration = perf_counter() - start

    decorated = cast(TimedFunction[Parameters, Result], wrapper)
    decorated.last_duration = None
    return decorated


def lru_cache(
    max_size: int = 128,
    key: Callable[..., Hashable] | None = None,
) -> Callable[[Callable[Parameters, Result]], CachedFunction[Parameters, Result]]:
    if max_size < 1:
        raise ValueError("max_size must be positive")

    def decorate(
        function: Callable[Parameters, Result],
    ) -> CachedFunction[Parameters, Result]:
        cache: OrderedDict[Hashable, Result] = OrderedDict()
        hits = 0
        misses = 0
        cached: CachedFunction[Parameters, Result]

        @wraps(function)
        def wrapper(*args: Parameters.args, **kwargs: Parameters.kwargs) -> Result:
            nonlocal hits, misses
            cache_key = (
                key(*args, **kwargs)
                if key is not None
                else cast(Hashable, (args, tuple(sorted(kwargs.items()))))
            )
            if cache_key in cache:
                hits += 1
                result = cache.pop(cache_key)
                cache[cache_key] = result
                cached.hits = hits
                return result

            misses += 1
            result = function(*args, **kwargs)
            cache[cache_key] = result
            if len(cache) > max_size:
                cache.popitem(last=False)
            cached.misses = misses
            return result

        def clear() -> None:
            nonlocal hits, misses
            cache.clear()
            hits = 0
            misses = 0
            cached.hits = hits
            cached.misses = misses

        cached = cast(CachedFunction[Parameters, Result], wrapper)
        cached.hits = hits
        cached.misses = misses
        cached.clear = clear
        return cached

    return decorate
