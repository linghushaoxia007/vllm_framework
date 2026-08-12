"""OpenAI-compatible HTTP API server stubs."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="MiniVLLM", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> dict[str, list[dict[str, str]]]:
    return {"data": [{"id": "not-loaded", "object": "model"}]}
