# LLM Cost Predictor

A CLI tool for predicting operational costs of multi-model LLM architectures using Monte Carlo simulation.

## Installation

```bash
uv sync
```

## Usage

```bash
cost-predictor simulate <profile>
cost-predictor compare <profile1> <profile2>
cost-predictor sensitivity <profile> --parameter <param>
cost-predictor forecast <profile> --months <N>
```
