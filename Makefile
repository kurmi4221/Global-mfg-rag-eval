.PHONY: install lint format typecheck test demo selftest online-demo schema verify-ledger clean

install:
	uv sync --all-extras
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check . --fix

typecheck:
	uv run mypy

test:
	uv run pytest --cov

# Bless a baseline, re-run a clean candidate (PASS), run a broken candidate (FAIL)
demo:
	uv run mfg-rag-eval demo

# CI check: exits 0 only if the gate passes the good pipeline AND blocks the broken one
selftest:
	uv run mfg-rag-eval selftest

online-demo:
	uv run mfg-rag-eval online-demo

schema:
	uv run python scripts/export_schema.py

verify-ledger:
	uv run mfg-rag-eval ledger-verify runs/candidate_broken/ledger.jsonl

clean:
	rm -rf runs state .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
