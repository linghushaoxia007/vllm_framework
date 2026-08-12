"""Smoke tests for project scaffolding."""

from __future__ import annotations

from minivllm import __version__
from minivllm.cli import build_parser, main
from minivllm.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from minivllm.engine import LLMEngine
from minivllm.kv_cache import BlockManager
from minivllm.request import Request, SamplingParams
from minivllm.scheduler import Scheduler


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_scheduler_promotes_waiting_requests() -> None:
    scheduler = Scheduler(SchedulerConfig(max_num_seqs=2))
    requests = [
        Request(prompt_token_ids=[1, 2], sampling_params=SamplingParams()),
        Request(prompt_token_ids=[3], sampling_params=SamplingParams()),
        Request(prompt_token_ids=[4, 5, 6], sampling_params=SamplingParams()),
    ]
    for request in requests:
        scheduler.add_request(request)

    scheduled = scheduler.schedule()
    assert len(scheduled) == 2
    assert len(scheduler.waiting) == 1
    assert len(scheduler.running) == 2


def test_block_manager_allocate_and_free() -> None:
    manager = BlockManager(CacheConfig(block_size=16), num_gpu_blocks=4)
    table = manager.allocate("req-1", num_blocks=2)
    assert len(table.blocks) == 2
    assert manager.num_free_blocks == 2

    manager.free("req-1")
    assert manager.num_free_blocks == 4
    assert "req-1" not in manager.tables


def test_engine_add_and_step() -> None:
    config = EngineConfig(model=ModelConfig(model="dummy/model"))
    engine = LLMEngine(config, num_gpu_blocks=8)
    request = Request(prompt_token_ids=[10, 11], sampling_params=SamplingParams(max_tokens=4))
    engine.add_request(request)

    scheduled = engine.step()
    assert len(scheduled) == 1
    assert scheduled[0].request_id == request.request_id


def test_cli_help_returns_zero() -> None:
    assert main([]) == 0
    parser = build_parser()
    args = parser.parse_args(["serve", "--model", "meta-llama/Llama-2-7b-hf"])
    assert args.command == "serve"
    assert args.model == "meta-llama/Llama-2-7b-hf"
