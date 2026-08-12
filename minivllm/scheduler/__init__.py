"""Continuous batching scheduler."""

from __future__ import annotations

from minivllm.config import SchedulerConfig
from minivllm.request import Request, RequestStatus


class Scheduler:
    """Skeleton scheduler for continuous batching."""

    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self.waiting: list[Request] = []
        self.running: list[Request] = []

    def add_request(self, request: Request) -> None:
        request.status = RequestStatus.WAITING
        self.waiting.append(request)

    def schedule(self) -> list[Request]:
        """Select requests for the next engine step.

        Placeholder: promote waiting requests until sequence budget is hit.
        """
        while self.waiting and len(self.running) < self.config.max_num_seqs:
            request = self.waiting.pop(0)
            request.status = RequestStatus.RUNNING
            self.running.append(request)
        return list(self.running)

    def abort_request(self, request_id: str) -> bool:
        for bucket in (self.waiting, self.running):
            for idx, request in enumerate(bucket):
                if request.request_id == request_id:
                    request.status = RequestStatus.ABORTED
                    del bucket[idx]
                    return True
        return False
