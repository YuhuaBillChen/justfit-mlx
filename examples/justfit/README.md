# JustFit reproduction kit

This kit accompanies the
[JustFit paper](https://arxiv.org/abs/2609.17475) and its
[project page](https://yuhuabillchen.github.io/mlx-vlm/).

## Start here

> ### [Run JustFit: beginner Quick Start →](quickstart.md)
>
> Download the public model, start the tested 64K-input + 8K-output profile,
> send the first message, and connect Open WebUI or Hermes Agent. Every command
> is included; no knowledge of the paper or benchmark harness is required.

This README is the **research reproduction guide**. Use it when you want to
reproduce the paper's smoke checks, 192K+16K boundary run, or validation suite.

This directory makes the public JustFit branch runnable without copying the
author's machine-specific launcher. It supports two useful levels of evidence:

1. **Smoke / lifecycle check** — a short request or several staggered requests
   to verify server startup, paged Q4 execution, streaming completion and
   singleton-to-batch transitions.
2. **Capacity protocol** — the deterministic repetitive-text workload used to
   force allocation and execution through a requested input/output boundary.

The second is intentionally expensive and is not a quality benchmark.

For normal use instead of a benchmark, see the
[serving guide](../../docs/justfit-serving.md). It includes a secured local
server plus Open WebUI, terminal chat, and Hermes Agent setup. The Hermes
profile is 64K input + 8K output (73,728 total positions).

## 1. Install the pinned source

On an Apple-silicon Mac with a working MLX environment:

```bash
git clone https://github.com/YuhuaBillChen/mlx-vlm.git
cd mlx-vlm
git checkout justfit-repro-v3
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

This tag contains the tested reproduction kit. The three repeated limit runs use
`143967472416aece96eeae392eaf931426acad3c` as their measured runtime
provenance; the `mlx_vlm/` tree is identical between that commit and this tag.

The evolving branch is useful for daily use, but an experiment should always
record an immutable commit:

```bash
git checkout production/qwen-paged-continuous-batching
git rev-parse HEAD
```

## 2. Download the pinned public artifacts

The public reproduction path uses immutable Hugging Face revisions. Download
the target checkpoint and the exact PhaseSwap/MTP artifacts:

```bash
python -m pip install -U huggingface_hub

hf download mlx-community/Qwen3.8-27B-mxfp4 \
  --revision 97ab0819817ab1c61d7d39f9169fc71999915641 \
  --local-dir ./justfit-models/qwen38-mxfp4
hf download billchen42/JustFit-Qwen3.8-27B-components \
  --revision 13e0462fa911a1bcbdccbaa759120500c0f90582 \
  --local-dir ./justfit-components

export MODEL_PATH="$PWD/justfit-models/qwen38-mxfp4"
export MTP_PATH="$PWD/justfit-components/mtp"
export VISION_PATH="$PWD/justfit-components/vision-bf16.safetensors"
```

The paper's local target artifact was repacked under the label
`Qwen3.8-27B-mxfp4-mtp-vq8`. A post-publication tensor-level audit found that
all 1,349 non-vision tensors (14,292,384,768 bytes) match the pinned public
MXFP4 checkpoint. The published BF16 vision backing matches the production
PhaseSwap file for all 333 tensors, and the published MTP weight file has the
same SHA-256 as the measured artifact. The local target container itself is not
claimed to be byte-identical because its inactive checkpoint-side vision tower
used a different quantized representation.

Verify the published component files before running:

```bash
shasum -a 256 \
  "$VISION_PATH" \
  "$MTP_PATH/model.safetensors" \
  "$MTP_PATH/config.json"
```

Expected digests are recorded in
[`components-manifest.json`](components-manifest.json).

Extract the untied output head and input embedding from the pinned target
checkpoint. The input embedding is already part of the target download; it is
not duplicated in the component repository.

```bash
python examples/justfit/prepare_components.py \
  --model "$MODEL_PATH" \
  --output ./justfit-extracted \
  --language-only
export LM_HEAD_PATH="$PWD/justfit-extracted/language-head.safetensors"
export INPUT_EMBEDDING_PATH="$PWD/justfit-extracted/input-embedding.safetensors"
```

The pinned checkpoint produces a 675,430,400-byte input-embedding tensor
payload. The production backing's provenance is recorded under
`derived_components` in
[`components-manifest.json`](components-manifest.json). Do not require a
locally generated container file to have the same SHA-256: safetensors header
metadata can vary by MLX version while the tensors remain identical.

If a different target checkpoint does contain top-level `mtp.*` tensors, create
the standalone directory with mlx-vlm's family splitter instead:

```bash
python -m mlx_vlm.speculative.drafters.qwen3_5_mtp.split \
  --model "$MODEL_PATH" \
  --output ./justfit-components/mtp \
  --block-size 3
export MTP_PATH="$PWD/justfit-components/mtp"
```

The component files are backing artifacts. PhaseSwap reconstructs the matching
runtime module from them; it does not imply a discrete-VRAM transfer. The
separate input-embedding backing lets an admitted image request release the
text embedding table before loading the vision tower, then restore it before
text prefill resumes.

## 3. Start the server

```bash
# Start with the safer smoke profile.
LANES=1 KV_CAPACITY=32768 MAX_TOKENS=512 \
  examples/justfit/run_server.sh
```

The launcher defaults to port 8080 and binds localhost. Override `PORT`,
`HOST`, `LANES`, `KV_CAPACITY`, `MAX_TOKENS`, or `OUTPUT_GUARANTEE` explicitly.
For a normal server, leave `CAPACITY_MODE=0`; EOS suppression is only enabled
for the forced-length capacity protocol.

## 4. Smoke test

In another terminal:

```bash
python examples/justfit/capacity_client.py \
  --model "$MODEL_PATH" \
  --tokenizer "$MODEL_PATH" \
  --prompt-tokens 8192 \
  --output-tokens 64 \
  --output smoke-result.json
```

Confirm that the server reports `Paged TurboQuant enabled`, the client sees
`[DONE]`, reported prompt/completion lengths match, and a second request can run
after the first releases its pages.

## 5. Reproduce the B1 limit protocol

This profile is a boundary test, not a production default. The reported runs
took roughly 103 minutes each and had only 23–40 MiB of sampled margin below a
21,000-MiB guard. Close competing GPU workloads and monitor the whole process.

```bash
# Terminal 1
CAPACITY_MODE=1 LANES=1 KV_CAPACITY=229376 MAX_TOKENS=16384 \
  OUTPUT_GUARANTEE=16384 examples/justfit/run_server.sh

# Terminal 2
python examples/justfit/capacity_client.py \
  --model "$MODEL_PATH" \
  --tokenizer "$MODEL_PATH" \
  --prompt-tokens 196608 \
  --output-tokens 16384 \
  --tokenizer-offset 36 \
  --timeout 14400 \
  --output b1-192k-16k.json
```

The `36`-token offset is specific to the recorded checkpoint/template pair:
the standalone tokenizer inserted a prefix that the server-side text processor
did not. The client validates the final server-reported prompt length. If your
checkpoint reports a different value, do not edit the result—record the
template revision and calibrate the offset for a new cohort.

The checked-in [`results/limit-runs.json`](results/limit-runs.json) is the
path-sanitized record used in the arXiv paper. The authoritative definitions
are:

- PP = uncached prefill tokens / summed prefill work seconds;
- B1 TG = 16,383 tokens after the first / first-to-last-token interval;
- peak = sampled whole-process footprint, not MLX active bytes and not a
  per-component attribution.

## Concurrent profiles

Set `LANES=2` or `LANES=4`, keep the pool total at 229,376 positions, and submit
several clients concurrently. Admission reserves page-rounded prompt state plus
the configured output guarantee and safety allowance. B2/B4 totals are shared
pool occupancy; aggregate TG must be calculated only over the common active
interval if it is to be compared with the paper.

Do not treat two sequential requests as B2, or add per-request decode rates to
manufacture an aggregate result.

## Reproduction-kit smoke evidence

On 2026-09-15, the tagged source flow was exercised on the paper's M4 Pro / 24
GiB host with APC disabled. The local measured-checkpoint flow produced a
675,430,400-byte language head and a 617,072,992-byte quantized vision tower.
Two consecutive requests on
one server each reported exactly 8,192 prompt tokens, 64 completion tokens,
`finish_reason=length`, `[DONE]`, and no stream error. Their TTFT / wall times
were 65.64 / 68.23 seconds and 64.25 / 66.84 seconds. The second request reused
the persistent paged pool after the first request released its page ownership.

This smoke validates the launcher, component extraction, streaming client,
phase-swapped head, paged Q4 prefill/decode, and page reuse against the local
measured-checkpoint layout. The public component repository removes the former
weight-availability gap; its BF16 vision file is the exact production
PhaseSwap backing, not the smaller quantized tower reported by that smoke. The
smoke remains a lifecycle check rather than a quality benchmark.

## Minimal validation suite

The backend-independent policy tests can run anywhere the package imports. The
Metal numerical tests require an Apple-silicon Mac:

```bash
python -m pytest -q \
  mlx_vlm/tests/test_component_residency.py \
  mlx_vlm/tests/test_residency_integration.py \
  mlx_vlm/tests/test_paged_turboquant_policy_lifecycle.py \
  mlx_vlm/tests/test_paged_turboquant_cache.py \
  mlx_vlm/tests/test_paged_turboquant_pool.py \
  mlx_vlm/tests/test_paged_turboquant_storage.py \
  mlx_vlm/tests/test_paged_turboquant_kernel.py
```

Archive the exact commit, checkpoint hashes, effective environment, full
successes and failures, stream trace, process-footprint samples, and postflight
reuse result for every reported cohort.
