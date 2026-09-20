#!/usr/bin/env python3
"""Extract checkpoint-local PhaseSwap backing files without loading all weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_vlm.utils import get_model_path, load_model


def save_component(module, path: Path) -> int:
    weights = dict(tree_flatten(module.parameters()))
    if not weights:
        raise ValueError(f"component at {path.name} has no parameters")
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), weights, metadata={"format": "mlx"})
    return sum(array.nbytes for array in weights.values())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract JustFit language-head, input-embedding, and optional "
            "vision-tower backing files."
        )
    )
    parser.add_argument("--model", required=True, help="Local path or Hugging Face id")
    parser.add_argument("--revision", help="Optional immutable Hugging Face revision")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--language-only",
        "--head-only",
        dest="language_only",
        action="store_true",
        help=(
            "Extract the language head and input embedding, but not the vision "
            "tower. --head-only is retained as a compatibility alias."
        ),
    )
    args = parser.parse_args()

    model_path = get_model_path(args.model, revision=args.revision)
    model = load_model(model_path, lazy=True)
    language_model = getattr(model, "language_model", model)
    head = getattr(language_model, "lm_head", None)
    if head is None:
        raise ValueError("checkpoint does not expose an untied language_model.lm_head")

    head_path = args.output / "language-head.safetensors"
    head_bytes = save_component(head, head_path)
    print(f"wrote {head_path} ({head_bytes} parameter bytes)")

    text_model = getattr(language_model, "model", language_model)
    input_embedding = getattr(text_model, "embed_tokens", None)
    if input_embedding is None:
        raise ValueError("checkpoint does not expose a text input embedding table")
    embedding_path = args.output / "input-embedding.safetensors"
    embedding_bytes = save_component(input_embedding, embedding_path)
    print(f"wrote {embedding_path} ({embedding_bytes} parameter bytes)")

    if not args.language_only:
        vision = getattr(model, "vision_tower", None)
        if vision is None:
            raise ValueError("checkpoint does not expose model.vision_tower")
        vision_path = args.output / "vision-tower.safetensors"
        vision_bytes = save_component(vision, vision_path)
        print(f"wrote {vision_path} ({vision_bytes} parameter bytes)")


if __name__ == "__main__":
    main()
