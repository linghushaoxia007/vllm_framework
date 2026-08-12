"""Model loading, prefill, decode, and generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from minivllm.config import ModelConfig
from minivllm.model_executor.llama import (
    CausalLMOutput,
    ContiguousKVCache,
    MiniLlamaForCausalLM,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class ModelRunner:
    """Correctness-first single-device Llama runner."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self.dtype = self._resolve_dtype(config.dtype, self.device)
        self.model: MiniLlamaForCausalLM | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return resolved

    @staticmethod
    def _resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
        if dtype == "auto":
            return torch.bfloat16 if device.type == "cuda" else torch.float32
        supported = {
            "float32": torch.float32,
            "float": torch.float32,
            "float16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        try:
            return supported[dtype]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype {dtype!r}; choose one of {sorted(supported)}") from exc

    def load_model(self) -> None:
        """Load Hugging Face weights/tokenizer and copy them into MiniLlama."""

        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            torch_dtype=self.dtype,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.load_huggingface_model(hf_model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model,
            trust_remote_code=self.config.trust_remote_code,
        )

    def load_huggingface_model(self, hf_model: PreTrainedModel) -> None:
        """Load an already-created HF model; useful for tests and local construction."""

        model = MiniLlamaForCausalLM.from_huggingface(hf_model)
        model.to(device=self.device, dtype=self.dtype)
        model.eval()
        self.model = model

    def _require_model(self) -> MiniLlamaForCausalLM:
        if self.model is None:
            raise RuntimeError("Model is not loaded; call load_model() first")
        return self.model

    @torch.inference_mode()
    def prefill(self, input_ids: Tensor | list[int] | list[list[int]]) -> CausalLMOutput:
        """Process an entire prompt and create a fresh contiguous KV cache."""

        ids = self._prepare_input_ids(input_ids)
        if ids.shape[1] == 0:
            raise ValueError("The prompt must contain at least one token")
        if ids.shape[1] > self.config.max_model_len:
            raise ValueError(
                f"Prompt has {ids.shape[1]} tokens, exceeding max_model_len="
                f"{self.config.max_model_len}"
            )
        return self._require_model()(ids, cache=None, use_cache=True)

    @torch.inference_mode()
    def decode(
        self,
        input_ids: Tensor | list[int] | list[list[int]],
        cache: ContiguousKVCache,
    ) -> CausalLMOutput:
        """Decode one token per sequence and append it to ``cache``."""

        ids = self._prepare_input_ids(input_ids)
        if ids.shape[1] != 1:
            raise ValueError(f"decode expects one token per sequence, got {ids.shape[1]}")
        if cache.num_tokens + 1 > self.config.max_model_len:
            raise ValueError(f"Decode would exceed max_model_len={self.config.max_model_len}")
        return self._require_model()(ids, cache=cache, use_cache=True)

    @torch.inference_mode()
    def generate_token_ids(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> list[int]:
        """Greedy generation using one prefill followed by cached decode calls."""

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if len(prompt_token_ids) + max_new_tokens > self.config.max_model_len:
            raise ValueError("Requested generation exceeds max_model_len")
        if max_new_tokens == 0:
            return []

        output = self.prefill(prompt_token_ids)
        generated: list[int] = []
        for _ in range(max_new_tokens):
            next_token = int(output.logits[:, -1, :].argmax(dim=-1).item())
            generated.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break
            output = self.decode([next_token], output.cache)
        return generated

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 16) -> str:
        """Tokenize text, run greedy generation, and decode new tokens."""

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded; call load_model() first")
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        generated = self.generate_token_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def execute_model(
        self,
        input_ids: Tensor | list[int] | list[list[int]],
        cache: ContiguousKVCache | None = None,
    ) -> CausalLMOutput:
        """Compatibility entrypoint dispatching to prefill or cached decode."""

        return self.prefill(input_ids) if cache is None else self.decode(input_ids, cache)

    def _prepare_input_ids(self, input_ids: Tensor | list[int] | list[list[int]]) -> Tensor:
        ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2:
            raise ValueError(f"input_ids must be rank 1 or 2, got rank {ids.ndim}")
        return ids


__all__ = [
    "CausalLMOutput",
    "ContiguousKVCache",
    "MiniLlamaForCausalLM",
    "ModelRunner",
]
