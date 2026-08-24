.DEFAULT_GOAL := help
export UV_CACHE_DIR ?= $(CURDIR)/.cache/uv

.PHONY: help bootstrap doctor format lint typecheck test test-all verify

help:
	@echo "NodeLM development commands"
	@echo "  make bootstrap  Create the reproducible local environment"
	@echo "  make doctor     Validate local tools and project configuration"
	@echo "  make format     Format Python sources and tests"
	@echo "  make lint       Run static lint checks"
	@echo "  make typecheck  Run strict Python type checks"
	@echo "  make test       Run offline tests"
	@echo "  make verify     Run every local quality gate"

bootstrap:
	./scripts/bootstrap.sh

doctor:
	./scripts/doctor.sh

format:
	uv run --frozen ruff format src tests
	uv run --frozen ruff check --fix src tests

lint:
	uv run --frozen ruff format --check src tests
	uv run --frozen ruff check src tests

typecheck:
	uv run --frozen mypy src

test:
	uv run --frozen pytest -m "not network and not training" --cov=nodelm --cov-report=term-missing

test-all:
	uv run --frozen pytest

verify: doctor lint typecheck test
