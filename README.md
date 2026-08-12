# MiniVLLM

A lightweight, educational LLM inference framework inspired by vLLM, built on
PyTorch and Transformers.

## Goals

First milestone: single-GPU Llama inference with:

- Prefill / Decode separation
- Continuous batching
- Paged KV cache
- Basic OpenAI-compatible HTTP API

## Project Layout

```text
minivllm/
  config/           # Engine and model configuration
  request/          # Request and sequence state
  scheduler/        # Continuous batching scheduler
  kv_cache/         # Block manager and paged KV cache
  model_executor/   # Model loading and forward execution
  sampling/         # Logits processing and token sampling
  engine/           # Sync / async inference loops
  api_server/       # FastAPI OpenAI-compatible server
  cli.py            # Command-line entrypoint
tests/              # Unit and integration tests
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m minivllm.cli --help
```

## Current Status

This repository is in the scaffolding stage. Core inference components are
placeholders and will be implemented incrementally.
