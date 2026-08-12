"""A correctness-first Llama implementation with a contiguous KV cache."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from transformers import LlamaConfig


LayerKV = tuple[Tensor, Tensor]


@dataclass
class ContiguousKVCache:
    """Per-layer key/value tensors stored contiguously along sequence length."""

    layers: list[LayerKV | None]
    num_tokens: int = 0

    @classmethod
    def empty(cls, num_layers: int) -> ContiguousKVCache:
        return cls(layers=[None] * num_layers)

    def clear(self) -> None:
        self.layers = [None] * len(self.layers)
        self.num_tokens = 0


@dataclass
class CausalLMOutput:
    """Output of one prefill or decode forward pass."""

    logits: Tensor
    cache: ContiguousKVCache | None = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.square().mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position_embeddings: int, base: float) -> None:
        super().__init__()
        del max_position_embeddings  # Frequencies are generated lazily for exact positions.
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        frequencies = torch.einsum(
            "bi,j->bij", position_ids.float(), self.inv_freq.to(position_ids.device)
        )
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return embeddings.cos().to(dtype=dtype), embeddings.sin().to(dtype=dtype)


def rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    query: Tensor, key: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (query * cos) + (rotate_half(query) * sin), (key * cos) + (
        rotate_half(key) * sin
    )


def repeat_kv(hidden_states: Tensor, num_key_value_groups: int) -> Tensor:
    if num_key_value_groups == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, num_key_value_groups, seq_len, head_dim
    )
    return hidden_states.reshape(
        batch, num_kv_heads * num_key_value_groups, seq_len, head_dim
    )


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        attention_bias = getattr(config, "attention_bias", False)
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=attention_bias
        )
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            config.max_position_embeddings,
            getattr(config, "rope_theta", 10000.0),
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        past_key_value: LayerKV | None,
        use_cache: bool,
    ) -> tuple[Tensor, LayerKV | None]:
        batch_size, query_length, _ = hidden_states.shape

        query = self.q_proj(hidden_states).view(
            batch_size, query_length, self.num_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        cos, sin = self.rotary_emb(position_ids, query.dtype)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        past_length = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            past_length = past_key.shape[-2]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = (key, value) if use_cache else None
        repeated_key = repeat_kv(key, self.num_key_value_groups)
        repeated_value = repeat_kv(value, self.num_key_value_groups)

        attention_weights = torch.matmul(query, repeated_key.transpose(2, 3)) * self.scaling
        key_positions = torch.arange(key.shape[-2], device=hidden_states.device)
        query_positions = past_length + torch.arange(
            query_length, device=hidden_states.device
        )
        causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        attention_weights = attention_weights.masked_fill(
            causal_mask.view(1, 1, query_length, key.shape[-2]),
            torch.finfo(attention_weights.dtype).min,
        )
        attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        attention_output = torch.matmul(attention_weights, repeated_value)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, query_length, -1
        )
        return self.o_proj(attention_output), present


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        if config.hidden_act not in {"silu", "swish"}:
            raise ValueError(
                f"Only SiLU is supported in the correctness baseline, got {config.hidden_act!r}"
            )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        past_key_value: LayerKV | None,
        use_cache: bool,
    ) -> tuple[Tensor, LayerKV | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present = self.self_attn(
            hidden_states, position_ids, past_key_value, use_cache
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, present


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: Tensor,
        cache: ContiguousKVCache | None = None,
        use_cache: bool = True,
    ) -> tuple[Tensor, ContiguousKVCache | None]:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, sequence], got {input_ids.shape}")
        if cache is not None and len(cache.layers) != len(self.layers):
            raise ValueError("KV cache layer count does not match model layer count")

        past_length = cache.num_tokens if cache is not None else 0
        position_ids = torch.arange(
            past_length,
            past_length + input_ids.shape[1],
            device=input_ids.device,
        ).unsqueeze(0)
        position_ids = position_ids.expand(input_ids.shape[0], -1)

        hidden_states = self.embed_tokens(input_ids)
        next_cache = (
            cache if cache is not None else ContiguousKVCache.empty(len(self.layers))
        ) if use_cache else None

        for layer_idx, layer in enumerate(self.layers):
            past_key_value = None if cache is None else cache.layers[layer_idx]
            hidden_states, present = layer(
                hidden_states, position_ids, past_key_value, use_cache
            )
            if next_cache is not None:
                next_cache.layers[layer_idx] = present

        if next_cache is not None:
            next_cache.num_tokens = past_length + input_ids.shape[1]
        return self.norm(hidden_states), next_cache


class MiniLlamaForCausalLM(nn.Module):
    """Minimal Llama causal LM whose parameter names match Hugging Face Llama."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: Tensor,
        cache: ContiguousKVCache | None = None,
        use_cache: bool = True,
    ) -> CausalLMOutput:
        hidden_states, cache = self.model(input_ids, cache=cache, use_cache=use_cache)
        logits = self.lm_head(hidden_states).float()
        return CausalLMOutput(logits=logits, cache=cache)

    @classmethod
    def from_huggingface(cls, hf_model: nn.Module) -> MiniLlamaForCausalLM:
        config = hf_model.config
        if getattr(config, "model_type", None) != "llama":
            raise ValueError(f"Expected a Llama model, got {getattr(config, 'model_type', None)!r}")
        model = cls(config)
        incompatible = model.load_state_dict(hf_model.state_dict(), strict=False)
        missing = [key for key in incompatible.missing_keys if "rotary_emb" not in key]
        unexpected = [key for key in incompatible.unexpected_keys if "rotary_emb" not in key]
        if missing or unexpected:
            raise RuntimeError(
                f"Incompatible Hugging Face weights; missing={missing}, unexpected={unexpected}"
            )
        return model


def cache_size_bytes(cache: ContiguousKVCache) -> int:
    """Return storage occupied by key/value tensors (without Python overhead)."""

    total = 0
    for layer in cache.layers:
        if layer is not None:
            total += sum(tensor.numel() * tensor.element_size() for tensor in layer)
    return total
