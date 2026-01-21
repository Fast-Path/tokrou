# Makefile for Multi-Model LLM Cost Predictor

.PHONY: help install test format lint clean

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run all tests"
	@echo "  make format     - Format code with black"
	@echo "  make lint       - Lint code with ruff"
	@echo "  make clean      - Remove build artifacts"

install:
	uv sync --extra dev

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ --cov=backend --cov-report=html --cov-report=term

format:
	uv run black backend/ tests/
	uv run ruff check --fix backend/ tests/

lint:
	uv run ruff check backend/ tests/
	uv run mypy backend/

clean:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
