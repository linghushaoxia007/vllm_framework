"""Inference engine orchestration."""

from __future__ import annotations

from minivllm.config import EngineConfig
from minivllm.kv_cache import BlockManager
from minivllm.model_executor import ModelRunner
from minivllm.request import Request
from minivllm.scheduler import Scheduler


class LLMEngine:
    """Synchronous inference engine.

    Stage one exposes correctness-first single-request generation. Request
    scheduling is retained for the later continuous-batching milestone.
    """

    def __init__(self, config: EngineConfig, num_gpu_blocks: int = 1024) -> None:
        self.config = config
        self.scheduler = Scheduler(config.scheduler)
        self.block_manager = BlockManager(config.cache, num_gpu_blocks=num_gpu_blocks)
        self.model_runner = ModelRunner(config.model)

    def add_request(self, request: Request) -> None:
        self.scheduler.add_request(request)

    def load_model(self) -> None:
        self.model_runner.load_model()

    def generate(self, prompt: str, max_new_tokens: int = 16) -> str:
        return self.model_runner.generate(prompt, max_new_tokens=max_new_tokens)

    def step(self) -> list[Request]:
        """Run one scheduling + execution step.

        Placeholder: currently only returns the scheduled request set.
        """
        return self.scheduler.schedule()
