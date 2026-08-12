"""Correctness tests against Hugging Face's Llama implementation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from minivllm.config import ModelConfig
from minivllm.model_executor import MiniLlamaForCausalLM, ModelRunner
from minivllm.model_executor.llama import cache_size_bytes


@pytest.fixture
def tiny_hf_llama() -> LlamaForCausalLM:
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        attention_bias=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def test_full_forward_logits_match_huggingface(tiny_hf_llama: LlamaForCausalLM) -> None:
    model = MiniLlamaForCausalLM.from_huggingface(tiny_hf_llama).eval()
    input_ids = torch.tensor([[1, 8, 15, 16, 23, 42], [1, 4, 9, 25, 36, 49]])

    with torch.inference_mode():
        expected = tiny_hf_llama(input_ids, use_cache=False).logits
        actual = model(input_ids, use_cache=False).logits

    torch.testing.assert_close(actual, expected.float(), rtol=1e-4, atol=1e-5)


def test_prefill_and_cached_decode_match_huggingface(
    tiny_hf_llama: LlamaForCausalLM,
) -> None:
    model = MiniLlamaForCausalLM.from_huggingface(tiny_hf_llama).eval()
    prompt = torch.tensor([[1, 11, 12, 13]])
    next_token = torch.tensor([[14]])

    with torch.inference_mode():
        hf_prefill = tiny_hf_llama(prompt, use_cache=True)
        hf_decode = tiny_hf_llama(
            next_token,
            past_key_values=hf_prefill.past_key_values,
            use_cache=True,
        )
        prefill = model(prompt, use_cache=True)
        decode = model(next_token, cache=prefill.cache, use_cache=True)
        full = model(torch.cat((prompt, next_token), dim=1), use_cache=False)

    assert prefill.cache is not None
    assert decode.cache is prefill.cache
    assert decode.cache.num_tokens == 5
    torch.testing.assert_close(prefill.logits, hf_prefill.logits.float(), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(decode.logits, hf_decode.logits.float(), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(decode.logits[:, -1], full.logits[:, -1], rtol=1e-4, atol=1e-5)


def test_runner_greedy_generation_matches_reference(
    tiny_hf_llama: LlamaForCausalLM,
) -> None:
    runner = ModelRunner(
        ModelConfig(model="unused", dtype="float32", device="cpu", max_model_len=32)
    )
    runner.load_huggingface_model(tiny_hf_llama)
    prompt = [1, 20, 30]

    actual = runner.generate_token_ids(prompt, max_new_tokens=5)
    with torch.inference_mode():
        expected = tiny_hf_llama.generate(
            torch.tensor([prompt]),
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tiny_hf_llama.config.pad_token_id,
            eos_token_id=None,
        )[0, len(prompt) :].tolist()

    assert actual == expected


def test_cache_layout_and_size(tiny_hf_llama: LlamaForCausalLM) -> None:
    runner = ModelRunner(
        ModelConfig(model="unused", dtype="float32", device="cpu", max_model_len=32)
    )
    runner.load_huggingface_model(tiny_hf_llama)

    output = runner.prefill([1, 2, 3])

    assert output.cache is not None
    assert output.cache.num_tokens == 3
    assert len(output.cache.layers) == tiny_hf_llama.config.num_hidden_layers
    key, value = output.cache.layers[0]
    expected_shape = (
        1,
        tiny_hf_llama.config.num_key_value_heads,
        3,
        tiny_hf_llama.config.hidden_size // tiny_hf_llama.config.num_attention_heads,
    )
    assert key.shape == expected_shape
    assert value.shape == expected_shape
    assert cache_size_bytes(output.cache) > 0


def test_runner_loads_huggingface_model_and_tokenizer(
    tiny_hf_llama: LlamaForCausalLM,
    tmp_path: Path,
) -> None:
    tiny_hf_llama.save_pretrained(tmp_path)
    backend = Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "<unk>": 3,
                "hello": 4,
            },
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(tmp_path)

    runner = ModelRunner(
        ModelConfig(model=str(tmp_path), dtype="float32", device="cpu", max_model_len=16)
    )
    runner.load_model()

    assert runner.model is not None
    assert runner.tokenizer is not None
    assert runner.tokenizer.encode("hello", add_special_tokens=False) == [4]
