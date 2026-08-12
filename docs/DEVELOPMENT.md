# MiniVLLM development notes

## Implementation order

1. Model loading + plain KV cache correctness
2. Prefill / decode separation
3. Async request engine
4. Continuous batching
5. Paged KV cache
6. Triton / CUDA kernels
7. OpenAI-compatible API + benchmarks

## Conventions

- Keep public APIs small and typed.
- Prefer correctness tests before performance work.
- Avoid pulling Transformers `generate()` into the hot path.
