"""Paged KV cache block management."""

from __future__ import annotations

from dataclasses import dataclass, field

from minivllm.config import CacheConfig


@dataclass(slots=True)
class BlockTable:
    """Logical-to-physical block mapping for one sequence."""

    blocks: list[int] = field(default_factory=list)


class BlockManager:
    """Allocate and free fixed-size KV cache blocks."""

    def __init__(self, config: CacheConfig, num_gpu_blocks: int) -> None:
        if num_gpu_blocks <= 0:
            raise ValueError("num_gpu_blocks must be positive")
        self.config = config
        self.num_gpu_blocks = num_gpu_blocks
        self.free_blocks: list[int] = list(range(num_gpu_blocks))
        self.tables: dict[str, BlockTable] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def allocate(self, request_id: str, num_blocks: int) -> BlockTable:
        if num_blocks > self.num_free_blocks:
            raise RuntimeError(
                f"Not enough KV blocks: need {num_blocks}, free {self.num_free_blocks}"
            )
        blocks = [self.free_blocks.pop() for _ in range(num_blocks)]
        table = BlockTable(blocks=blocks)
        self.tables[request_id] = table
        return table

    def free(self, request_id: str) -> None:
        table = self.tables.pop(request_id, None)
        if table is None:
            return
        self.free_blocks.extend(table.blocks)
