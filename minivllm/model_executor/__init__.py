"""Model loading and execution stubs."""

from __future__ import annotations

from minivllm.config import ModelConfig


class ModelRunner:
    """Loads a model and runs prefill / decode steps.

    This is intentionally a stub for the project scaffold.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.model = None

    def load_model(self) -> None:
        raise NotImplementedError("Model loading will be implemented in a later milestone.")

    def execute_model(self, input_ids: list[list[int]]) -> list[list[float]]:
        del input_ids
        raise NotImplementedError("Model execution will be implemented in a later milestone.")
