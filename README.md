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

Run greedy generation with a Hugging Face Llama model:

```bash
minivllm generate \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "The capital of France is" \
  --max-new-tokens 16
```

## Current Status

The stage-one correctness baseline is implemented:

- Hugging Face Llama weight and tokenizer loading
- A PyTorch Llama forward pass (RMSNorm, RoPE, grouped-query attention, SwiGLU)
- Contiguous per-layer KV cache
- Separate prompt prefill and one-token decode paths
- Greedy generation
- Logit, cache, and generation parity tests against Transformers

This is deliberately a correctness implementation. It does not yet include
continuous batching, paged KV cache execution, optimized kernels, sampling, or
production serving.
