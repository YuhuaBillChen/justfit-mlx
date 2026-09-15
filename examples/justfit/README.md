# JustFit reproduction kit

This directory makes the public JustFit branch runnable without copying the
author's machine-specific launcher. It supports two useful levels of evidence:

1. **Smoke / lifecycle check** — a short request or several staggered requests
   to verify server startup, paged Q4 execution, streaming completion and
   singleton-to-batch transitions.
2. **Capacity protocol** — the deterministic repetitive-text workload used to
   force allocation and execution through a requested input/output boundary.

The second is intentionally expensive and is not a quality benchmark.

## 1. Install the pinned source

On an Apple-silicon Mac with a working MLX environment:

```bash
git clone https://github.com/YuhuaBillChen/mlx-vlm.git
cd mlx-vlm
git checkout a30c0cf39fd4c3367f3bf381df192851cfaa1801
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

This commit contains the reproduction kit. The three repeated limit runs use
`143967472416aece96eeae392eaf931426acad3c` as their measured runtime
provenance; the `mlx_vlm/` tree is identical between that commit and the
reproduction-kit commit above.

The evolving branch is useful for daily use, but an experiment should always
record an immutable commit:

```bash
git checkout production/qwen-paged-continuous-batching
git rev-parse HEAD
```

## 2. Prepare checkpoint-side components

Set `MODEL_PATH` to a compatible converted Qwen3.8-27B MXFP4 VLM checkpoint.
The paper's local artifact label was `Qwen3.8-27B-mxfp4-mtp-vq8`; this repository
does not redistribute those weights or claim a retrospectively verified public
model revision.

Extract the output head and vision tower from the same checkpoint:

```bash
python examples/justfit/prepare_components.py \
  --model "$MODEL_PATH" \
  --output ./justfit-components
```

Split the checkpoint's native MTP tensors with mlx-vlm's family splitter:

```bash
python -m mlx_vlm.speculative.drafters.qwen3_5_mtp.split \
  --model "$MODEL_PATH" \
  --output ./justfit-components/mtp \
  --block-size 3
```

The component files are backing artifacts. PhaseSwap reconstructs the matching
runtime module from them; it does not imply a discrete-VRAM transfer.

## 3. Start the server

```bash
export MODEL_PATH=/absolute/path/to/Qwen3.8-27B-mxfp4-mtp-vq8
export MTP_PATH="$PWD/justfit-components/mtp"
export LM_HEAD_PATH="$PWD/justfit-components/language-head.safetensors"
export VISION_PATH="$PWD/justfit-components/vision-tower.safetensors"

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
  --model "$(basename "$MODEL_PATH")" \
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
  --model "$(basename "$MODEL_PATH")" \
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
path-sanitized record used in the final preprint. The authoritative definitions
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
