#!/usr/bin/env python3
"""Read-only, stdlib audit of the supplied M3 core-v1 evidence directory.

Usage: python3 verify_evidence.py /path/to/paper-repro/core-v1
Prints a reproducible report; never runs a model or modifies the evidence.
This checks raw measurement evidence, not checkpoint contents or macOS state.
"""

import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

NAMES = [
    "fixed-32k-6k-r1",
    "fixed-32k-6k-r2",
    "fixed-32k-6k-r3",
    "b1-cold-240k-16k",
    "b2-cold-128k-16k",
    "b4-incumbent-128k-3x8k-12k",
]
SHAPES = [([32768], 6144)] * 3 + [
    ([245760], 16384),
    ([131072, 131072], 16384),
    ([131072, 8192, 8192, 8192], 12288),
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def jsonlines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(root):
    rows, hashes, pids = [], {}, []
    for name, (prompts, output) in zip(NAMES, SHAPES):
        folder = root / name
        metrics = load(folder / "metrics.json")
        clients = load(folder / "clients.json")
        events = jsonlines(folder / "events.jsonl")
        samples = jsonlines(folder / "footprint.jsonl")
        require(
            sorted(c["requested_prompt_tokens"] for c in clients) == sorted(prompts),
            name,
        )
        require(all(c["requested_output_tokens"] == output for c in clients), name)
        require(len(metrics["lanes"]) == len(clients), name)
        startup = [e for e in events if e["event"] == "instrumentation_start"]
        require(len(startup) == 1, name + ": instrumentation start")
        pids.append(startup[0]["pid"])
        points, prefill_times, token_hashes, text_hashes = [], [], [], []
        for client in clients:
            lane = next(l for l in metrics["lanes"] if l["lane"] == client["lane"])
            stream = folder / (client["lane"] + ".sse")
            done, finishes, usage, text = 0, [], None, []
            for line in stream.read_text().splitlines():
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    done += 1
                    continue
                if not body:
                    continue
                event = json.loads(body)
                require(not event.get("error"), name + ": SSE error")
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    if choice.get("finish_reason"):
                        finishes.append(choice["finish_reason"])
                    delta = choice.get("delta") or {}
                    text.append(
                        (delta.get("reasoning_content") or "")
                        + (delta.get("content") or "")
                    )
            require(done == 1 and finishes == ["length"], name + ": SSE termination")
            require(usage == client["usage"], name + ": SSE usage")
            require(
                usage["prompt_tokens"] == client["requested_prompt_tokens"]
                and usage["completion_tokens"] == output
                and usage["total_tokens"] == client["requested_prompt_tokens"] + output
                and usage["prompt_tokens_details"]["cached_tokens"] == 0,
                name + ": token totals",
            )
            digest = hashlib.sha256("".join(text).encode()).hexdigest()
            require(
                digest == client["text_sha256"] == lane["text_sha256"],
                name + ": text hash",
            )
            text_hashes.append(digest)
            history = [e for e in events if e.get("request_id") == lane["request_id"]]
            start = [e for e in history if e["event"] == "request_start"]
            require(
                len(start) == 1
                and start[0]["prompt_tokens"] == client["requested_prompt_tokens"],
                name,
            )
            pf = [e for e in history if e["event"] == "prefill_completed"]
            require(len(pf) == 1 and pf[0]["cached_tokens"] == 0, name)
            prefill_times.append(pf[0]["prompt_time"])
            decode = [e for e in history if e["event"] == "decode"]
            require(decode[-1]["finish_reason"] == "length", name)
            total, seq, token_ids = 0, [], bytearray()
            for e in decode:
                count = e["token_count"]
                require(count in (0, 1), name + ": individual token IDs required")
                total += count
                require(e["generated_tokens"] == total, name + ": token sequence")
                if count:
                    seq.append(e["timestamp"])
                    token_ids.extend(e["token"].to_bytes(8, "little"))
            require(
                total == output and seq == sorted(seq), name + ": decode count/timing"
            )
            digest = hashlib.sha256(token_ids).hexdigest()
            require(digest == lane["token_ids_sha256"], name + ": token hash")
            token_hashes.append(digest)
            points.append(seq)
        require(len({l["request_id"] for l in metrics["lanes"]}) == len(clients), name)
        begin, end = max(p[0] for p in points), min(p[-1] for p in points)
        require(end > begin, name + ": no common decode interval")
        count = (
            output - 1
            if len(points) == 1
            else sum(begin < t < end for p in points for t in p)
        )
        peak = max(s["phys_footprint_bytes"] for s in samples) / 2**20
        require(peak < 21000, name + ": sampled guard")
        pool = re.findall(
            r"high_water_pages=(\d+) capacity_pages=(\d+)",
            (folder / "server.log").read_text(),
        )
        require(bool(pool), name + ": pool telemetry")
        calculated = {
            "prefill_work_tps": sum(prompts) / sum(prefill_times),
            "decode_tps": count / (end - begin),
            "decode_interval_tokens": count,
            "decode_interval_seconds": end - begin,
            "sampled_peak_phys_footprint_mib": peak,
            "paged_high_water_position_slots": max(int(a) for a, _ in pool) * 256 // 16,
            "completed_input_output_positions": sum(prompts) + output * len(prompts),
            "wall_seconds": max(c["ended_at"] for c in clients)
            - min(c["started_at"] for c in clients),
        }
        for key, value in calculated.items():
            require(
                math.isclose(value, metrics[key], rel_tol=1e-10, abs_tol=1e-8),
                name + ": " + key,
            )
        require(
            len(set(token_hashes)) == len(set(text_hashes)) == 1,
            name + ": lane output agreement",
        )
        rows.append(
            dict(
                name=name,
                **calculated,
                token_ids_sha256=token_hashes[0],
                text_sha256=text_hashes[0],
            )
        )
        for file in sorted(folder.iterdir()):
            if (
                file.name
                in {
                    "metrics.json",
                    "clients.json",
                    "events.jsonl",
                    "footprint.jsonl",
                    "server.log",
                }
                or file.name.startswith("lane-")
                and file.suffix == ".sse"
            ):
                hashes[str(file.relative_to(root))] = hashlib.sha256(
                    file.read_bytes()
                ).hexdigest()
    require(len(set(pids)) == 6, "Expected six fresh server processes")
    require(
        len({r["token_ids_sha256"] for r in rows[:3]}) == 1,
        "Fixed repeat hash mismatch",
    )
    require(
        rows[3]["token_ids_sha256"] == rows[4]["token_ids_sha256"],
        "B1/B2 hash mismatch",
    )
    return {
        "status": "passed",
        "scope": "Independent raw SSE, token-event, timing, sampled footprint and pool audit; no model rerun or weight rehash",
        "rows": rows,
        "fixed_n3_medians": {
            k: statistics.median(r[k] for r in rows[:3])
            for k in (
                "prefill_work_tps",
                "decode_tps",
                "sampled_peak_phys_footprint_mib",
            )
        },
        "evidence_sha256": hashes,
    }


if __name__ == "__main__":
    print(json.dumps(audit(Path(sys.argv[1])), indent=2, allow_nan=False))
