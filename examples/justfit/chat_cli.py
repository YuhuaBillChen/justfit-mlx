#!/usr/bin/env python3
"""Small interactive client for a JustFit OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def request_json(url: str, *, api_key: str) -> dict:
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def discover_model(base_url: str, *, api_key: str) -> str:
    payload = request_json(f"{base_url.rstrip('/')}/models", api_key=api_key)
    models = [entry.get("id") for entry in payload.get("data", []) if entry.get("id")]
    if len(models) != 1:
        raise RuntimeError(
            f"expected exactly one served model, found {models!r}; pass --model"
        )
    return models[0]


def stream_chat(
    base_url: str,
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")

    parts: list[str] = []
    saw_done = False
    with urllib.request.urlopen(request, timeout=None) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                saw_done = True
                break
            event = json.loads(data)
            choices = event.get("choices") or []
            if not choices:
                continue
            content = (choices[0].get("delta") or {}).get("content")
            if content:
                print(content, end="", flush=True)
                parts.append(content)
    if not saw_done:
        raise RuntimeError("SSE stream ended without [DONE]")
    print()
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--api-key", default=os.environ.get("JUSTFIT_API_KEY", ""))
    parser.add_argument("--model", help="Defaults to the sole entry from /v1/models")
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")

    try:
        model = args.model or discover_model(args.base_url, api_key=args.api_key)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"failed to discover model: {exc}", file=sys.stderr)
        return 1

    print(f"Connected to {model}. Commands: /reset, /exit")
    messages: list[dict[str, str]] = []
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            continue
        if prompt == "/exit":
            return 0
        if prompt == "/reset":
            messages.clear()
            print("Conversation reset.")
            continue

        messages.append({"role": "user", "content": prompt})
        print("assistant> ", end="", flush=True)
        try:
            answer = stream_chat(
                args.base_url,
                api_key=args.api_key,
                model=model,
                messages=messages,
                max_tokens=args.max_tokens,
            )
        except KeyboardInterrupt:
            print("\nRequest cancelled.")
            messages.pop()
            continue
        except (urllib.error.URLError, RuntimeError, ValueError) as exc:
            print(f"\nrequest failed: {exc}", file=sys.stderr)
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    raise SystemExit(main())
