# Multi-Model LLM Cost Predictor

A cost analysis system for predicting operational expenses of a multi-model LLM orchestration architecture using Monte Carlo simulation.

## Overview

This system predicts costs for a multi-model LLM architecture that uses:
- **Gemini** for visual tasks
- **Coder** (Cerebras/Qwen) for code generation
- **Grok** for research and analysis
- **Classifier** for routing queries to the appropriate model

The system uses Monte Carlo simulation to model query routing, delegation patterns, and token usage across these models.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

### Install uv

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install Dependencies

```bash
# Install production dependencies
uv sync

# Install with development dependencies
uv sync --extra dev
```

## Project Structure

```
cost-predictor/
├── backend/
│   ├── core/          # Core calculation engine
│   ├── api/           # Flask REST API
│   └── cli/           # Command-line interface
├── config/
│   ├── profiles/      # Usage profile configurations
│   ├── pricing.yaml   # Model pricing
│   ├── token_estimation.yaml
│   └── simulation.yaml
├── tests/
│   ├── unit/          # Unit tests
│   ├── property/      # Property-based tests
│   └── integration/   # Integration tests
└── pyproject.toml     # Project configuration
```

## Configuration

All system parameters are externalized in YAML configuration files:

### Usage Profiles (`config/profiles/`)

Define query distributions, complexity, routing accuracy, and volume:
- `baseline.yaml` - Expected usage patterns
- `conservative.yaml` - Pessimistic assumptions (worst-case)
- `optimistic.yaml` - Best-case assumptions

### Pricing (`config/pricing.yaml`)

Per-token costs for all models (input and output tokens).

### Token Estimation (`config/token_estimation.yaml`)

Rules for estimating token usage:
- System prompt sizes
- Input token ranges by query type and complexity
- Output token multipliers
- Wrong model penalty

### Simulation (`config/simulation.yaml`)

Monte Carlo simulation parameters:
- Number of runs
- Days to simulate
- Sensitivity analysis ranges

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=backend

# Run specific test file
uv run pytest tests/unit/test_config_loader.py -v
```

## Development

### Code Formatting

```bash
# Format code with black
uv run black backend/ tests/

# Lint with ruff
uv run ruff check backend/ tests/
```

### Type Checking

```bash
uv run mypy backend/
```

## Next Steps

This completes the project setup. The next tasks will implement:
1. Core domain types (enums, dataclasses)
2. Token estimation logic
3. Cost prediction engine
4. Monte Carlo simulation
5. CLI and web interfaces

See `.kiro/specs/llm-cost-predictor/tasks.md` for the complete implementation plan.
