.PHONY: install lint format typecheck test check pre-commit

install:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test

pre-commit:
	uv run pre-commit run --all-files