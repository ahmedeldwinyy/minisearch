# MiniSearch

MiniSearch is a search and recommendation engine built from scratch as a
portfolio project. The core will use a custom inverted index, ranking
algorithms, and recommendation methods, served by a FastAPI application.

## Project Principles

- Implement core search and recommendation algorithms in this repository.
- Use external search and recommendation systems only for later benchmarks.
- Add tests, type hints, and performance measurements as features are built.

## Development Setup

Requirements: Python 3.11 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
make check
```

Run `make format` to format Python files and `make pre-commit` to run the
configured hooks.

## Dataset

Download the BEIR SciFact archive with:

```sh
uv run python scripts/download_dataset.py
```

The archive is saved under `data/raw/`, which is ignored by Git.

## Project Layout

```text
src/minisearch/   Application package
tests/            Automated tests
scripts/          Dataset and development scripts
data/             Downloaded data (not committed)
docs/             Design and project documentation
```

## Status

Foundation setup is in progress. See the project history for completed phases
and upcoming work.
