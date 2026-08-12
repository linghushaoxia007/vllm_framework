"""Command-line entrypoint for MiniVLLM."""

from __future__ import annotations

import argparse

from minivllm import __version__
from minivllm.config import EngineConfig, ModelConfig
from minivllm.engine import LLMEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minivllm",
        description="MiniVLLM: lightweight LLM inference framework",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"minivllm {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Start the OpenAI-compatible API server")
    serve.add_argument("--host", default="0.0.0.0", help="Bind host")
    serve.add_argument("--port", type=int, default=8000, help="Bind port")
    serve.add_argument("--model", required=True, help="Hugging Face model id or local path")

    generate = subparsers.add_parser("generate", help="Run correctness-first greedy generation")
    generate.add_argument("--model", required=True, help="Hugging Face Llama model id or path")
    generate.add_argument("--prompt", required=True, help="Input prompt")
    generate.add_argument("--max-new-tokens", type=int, default=16)
    generate.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device")
    generate.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "serve":
        print(
            "API server is not implemented yet. "
            f"Requested model={args.model!r}, host={args.host!r}, port={args.port}."
        )
        return 0

    if args.command == "generate":
        config = EngineConfig(
            model=ModelConfig(model=args.model, device=args.device, dtype=args.dtype)
        )
        engine = LLMEngine(config)
        engine.load_model()
        print(engine.generate(args.prompt, max_new_tokens=args.max_new_tokens))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
