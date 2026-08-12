"""Inference engine orchestration."""

from __future__ import annotations

from minivllm.config import EngineConfig
from minivllm.kv_cache import BlockManager
from minivllm.model_executor import ModelRunner
from minivllm.request import Request
from minivllm.scheduler import Scheduler


class LLMEngine:
    """Synchronous inference engine skeleton."""

    def __init__(self, config: EngineConfig, num_gpu_blocks: int = 1024) -> None:
        self.config = config
        self.scheduler = Scheduler(config.scheduler)
        self.block_manager = BlockManager(config.cache, num_gpu_blocks=num_gpu_blocks)
        self.model_runner = ModelRunner(config.model)

    def add_request(self, request: Request) -> None:
        self.scheduler.add_request(request)

    def step(self) -> list[Request]:
        """Run one scheduling + execution step.

        Placeholder: currently only returns the scheduled request set.
        """
        return self.scheduler.schedule()
