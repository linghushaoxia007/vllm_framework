"""Configuration dataclasses for MiniVLLM."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ModelConfig:
    """Model identity and architecture settings."""

    model: str
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    trust_remote_code: bool = False


@dataclass(slots=True)
class CacheConfig:
    """Paged KV cache settings."""

    block_size: int = 16
    gpu_memory_utilization: float = 0.9
    swap_space_gb: float = 0.0


@dataclass(slots=True)
class SchedulerConfig:
    """Continuous batching scheduler settings."""

    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    enable_chunked_prefill: bool = False


@dataclass(slots=True)
class EngineConfig:
    """Top-level engine configuration."""

    model: ModelConfig
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    seed: int = 0
