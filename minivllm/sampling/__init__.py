"""Token sampling utilities."""

from __future__ import annotations

from minivllm.request import SamplingParams


class Sampler:
    """Select next tokens from model logits."""

    def sample(self, logits: list[float], params: SamplingParams) -> int:
        del logits, params
        raise NotImplementedError("Sampling will be implemented in a later milestone.")
