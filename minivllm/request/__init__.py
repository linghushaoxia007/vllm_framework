"""Request and sequence state definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from uuid import uuid4


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    PREEMPTED = auto()
    FINISHED = auto()
    ABORTED = auto()


@dataclass(slots=True)
class SamplingParams:
    """Generation sampling parameters."""

    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    max_tokens: int = 16
    ignore_eos: bool = False
    stop: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Request:
    """A single inference request tracked by the engine."""

    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    request_id: str = field(default_factory=lambda: uuid4().hex)
    status: RequestStatus = RequestStatus.WAITING
    output_token_ids: list[int] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)
