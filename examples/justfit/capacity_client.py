#!/usr/bin/env python3
"""Submit the deterministic JustFit full-execution memory-stress workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


def build_messages(tokenizer, prompt_tokens: int, tokenizer_offset: int) -> list[dict]:
    target = prompt_tokens + tokenizer_offset
    repetitions = target
    for _ in range(10):
        messages = [
            {"role": "system", "content": "final qualification lane 0."},
            {
                "role": "user",
                "content": " measurement" * repetitions
                + "\nOutput the word measurement repeatedly, separated by spaces. "
                "Continue until the output limit. Never conclude or explain.",
            },
        ]
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        ids = encoded["input_ids"] if hasattr(encoded, "get") else encoded
        actual = len(ids)
        if actual == target:
            return messages
        repetitions += target - actual
    raise RuntimeError("could not construct the requested tokenizer-side length")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--model", required=True, help="Server-visible model id")
    parser.add_argument("--tokenizer", help="Checkpoint path; defaults to --model")
    parser.add_argument("--prompt-tokens", required=True, type=int)
    parser.add_argument("--output-tokens", required=True, type=int)
    parser.add_argument("--tokenizer-offset", type=int, default=36)
    parser.add_argument("--timeout", type=float, default=14400)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, local_files_only=True
    )
    messages = build_messages(tokenizer, args.prompt_tokens, args.tokenizer_offset)
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": args.output_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "logit_bias": {"248044": -10000.0, "248046": -10000.0},
    }
    request = urllib.request.Request(
        args.endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    started = time.perf_counter()
    first_text = None
    text_parts: list[str] = []
    usage = None
    finish_reason = None
    done = False
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for raw_line in response:
            line = raw_line.strip()
            if not line.startswith(b"data:"):
                continue
            body = line[5:].strip()
            if body == b"[DONE]":
                done = True
                continue
            if not body:
                continue
            event = json.loads(body)
            if event.get("usage") is not None:
                usage = event["usage"]
            for choice in event.get("choices", []):
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta", {})
                text = (delta.get("reasoning_content") or "") + (
                    delta.get("content") or ""
                )
                if text:
                    first_text = first_text or time.perf_counter()
                    text_parts.append(text)
    ended = time.perf_counter()

    result = {
        "schema_version": 1,
        "endpoint": args.endpoint,
        "model": args.model,
        "requested_prompt_tokens": args.prompt_tokens,
        "requested_output_tokens": args.output_tokens,
        "tokenizer_offset": args.tokenizer_offset,
        "done": done,
        "finish_reason": finish_reason,
        "usage": usage,
        "ttft_seconds": first_text - started if first_text else None,
        "wall_seconds": ended - started,
        "text_sha256": hashlib.sha256("".join(text_parts).encode()).hexdigest(),
    }
    if usage and usage.get("prompt_tokens") != args.prompt_tokens:
        result["validation_error"] = (
            f"server reported {usage.get('prompt_tokens')} prompt tokens, "
            f"expected {args.prompt_tokens}"
        )
    if usage and usage.get("completion_tokens") != args.output_tokens:
        result["validation_error"] = (
            result.get("validation_error", "")
            + f" server reported {usage.get('completion_tokens')} completion tokens, "
            f"expected {args.output_tokens}"
        ).strip()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not done or "validation_error" in result:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
