# JustFit

[Paper](https://arxiv.org/abs/2609.17475) ·
[Project page](https://yuhuabillchen.github.io/mlx-vlm/) ·
[Reproduction kit](../examples/justfit/README.md) ·
[Serving guide](justfit-serving.md) ·
[Published components](https://huggingface.co/billchen42/JustFit-Qwen3.8-27B-components)

**JustFit** is an experimental mlx-vlm runtime for executing and serving long
contexts under a fixed unified-memory budget. It coordinates three mechanisms:

- **KVExec** keeps TurboQuant Q4 KV state in fixed-size pages and consumes those
  pages directly for decode and short MTP verification. Eligible prefill uses a
  direct-inverse reconstruction path and evaluates each layer before returning,
  bounding the lifetime of floating-point attention operands.
- **PhaseSwap** assigns explicit leases to components whose useful lifetimes do
  not cover the whole request: the output head, vision tower, input embedding
  adapter when configured, and deferred MTP drafter.
- **StateTrans** changes between singleton MTP and multi-request AR at safe
  cohort boundaries while retaining surviving target KV and recurrent state.
  Completed or cancelled requests return page IDs to the resident pool only
  after pending state has been evaluated.

The production entry point is the
[`production/qwen-paged-continuous-batching`](https://github.com/YuhuaBillChen/mlx-vlm/tree/production/qwen-paged-continuous-batching)
branch. Commit
[`143967472416aece96eeae392eaf931426acad3c`](https://github.com/YuhuaBillChen/mlx-vlm/commit/143967472416aece96eeae392eaf931426acad3c)
is the pinned unified-residency revision used for the three repeated
single-request limit runs.

## Measured results

Platform: Apple M4 Pro MacBook Pro, 24 GiB unified memory; Qwen3.8-27B with
fixed MXFP4 weights; 21,000 MiB sampled process-footprint guard. `K = 1,024`.

| Workload | Retained positions | PP | TG | Peak footprint |
| --- | ---: | ---: | ---: | ---: |
| mlx-vlm baseline, B1 24K input + 6K output | 30,720 | — | did not complete the next fixed step | below guard at this point |
| JustFit, B1 192K input + 16K output, median of 3 | 212,992 | 68.311 tok/s | 4.9853 tok/s | 20,975 MiB |
| JustFit, B2 2×(96K input + 16K output) | 229,376 aggregate | 90.44 tok/s | 12.55 aggregate tok/s | 20,310 MiB |
| JustFit, B4 128K+12K and 3×(8K+12K) | 204,800 aggregate | 83.16 tok/s | 20.78 aggregate tok/s | 20,623 MiB |

All three B1 limit attempts completed 196,608 uncached input positions and
16,384 generated tokens. Their sampled peaks were 20,977, 20,975 and
20,960 MiB; all output-token sequences agreed, and all postflight/page-reuse
checks passed. The minimum sampled guard margin was only 23 MiB, so this is a
capacity boundary rather than a recommended service profile.

In a separate capability protocol, paged TQ4 answered 29/30 AIME 2026 problems
correctly and generated 696,834 tokens at 15.04 token-weighted tok/s. That
result is an integrated runtime check, not a claim that Q4 is universally more
accurate than INT8, and not a 200K long-context comprehension evaluation.

## What the numbers do and do not mean

- “200K” means retained input **plus generated output** positions. The B1 result
  is 196,608 input + 16,384 output, not 200K input.
- The B2 and B4 totals are aggregate state across concurrent requests. Their TG
  values are aggregate common-active-interval rates, not per-request rates.
- The capacity corpus is deterministic repetitive text with EOS suppression.
  It measures complete allocation and execution, not coding-agent task quality.
- The B1 cold request takes about 6,165 seconds end to end. Prefix reuse is a
  separate path and should not be conflated with cold ingestion.
- Returning pages makes resident pool slots reusable; it does not return the
  pool's backing allocation to macOS.
- “Loaded” in the residency manager means attached and usable. It is not a
  physical-memory counter and does not imply CPU↔GPU transfer on unified memory.

## Run it

The [reproduction kit](../examples/justfit/README.md) contains:

- a component-preparation helper;
- immutable public target and component revisions with SHA-256 manifests;
- a server launcher with the effective JustFit controls made explicit;
- a deterministic streaming capacity client;
- the public, path-sanitized JSON for the three repeated limit runs.

Start with the 8K+64 smoke profile. Only attempt 192K+16K after verifying the
checkpoint, component files, thermal conditions, free disk space and a process
footprint monitor. The reported throughput and final few MiB of headroom are
hardware-, checkpoint- and software-revision-specific.

For daily use, the [serving guide](justfit-serving.md) documents one shared
OpenAI-compatible endpoint for Open WebUI, the included terminal client, and
Hermes Agent. Its Hermes profile uses 65,536 input positions plus an 8,192-token
output ceiling (73,728 total), rather than the paper's boundary workload.

## Source map

| Concern | Primary implementation |
| --- | --- |
| Paged ownership and refcounts | `mlx_vlm/paged_turboquant.py` |
| Per-request cache facade | `mlx_vlm/paged_turboquant_cache.py` |
| Pool-backed storage | `mlx_vlm/paged_turboquant_storage.py` |
| Metal decode / verify / direct-inverse prefill | `mlx_vlm/paged_turboquant_kernel.py` |
| Pool registry and APC restoration | `mlx_vlm/paged_turboquant_pool.py` |
| Admission, mixed prefill/decode, MTP↔AR transitions | `mlx_vlm/server/generation.py` |
| Component leases and dependencies | `mlx_vlm/server/component_residency.py` |
| Head / embedding adapters | `mlx_vlm/server/language_lifecycle.py` |
| Vision and drafter adapters | `mlx_vlm/server/vision_lifecycle.py`, `draft_lifecycle.py` |

For implementation contracts and version-scoped validation, also see
[`paged-runtime-quality.md`](paged-runtime-quality.md) and
[`component-residency-quality.md`](component-residency-quality.md).

## Status

This is research software. The page geometry currently targets the evaluated
Qwen3.8 configuration (head dimension 256, page size 256, Q4 TurboQuant) and
does not claim general paged-attention support for every mlx-vlm model. The
converted checkpoint label used in the experiments remains a local artifact
label. A post-publication tensor audit matched all 1,349 non-vision tensors to
the pinned public MXFP4 target, all 333 production BF16 vision-backing tensors
to the published component, and the MTP weights by SHA-256. The local target
container is not claimed to be byte-identical because its inactive
checkpoint-side vision representation was quantized differently.

## Citation

```bibtex
@misc{chen2026justfit,
  title         = {JustFit: 200K-Token LLM Serving on a 24 GiB Laptop with Just-in-Time State Management},
  author        = {Yuhua Chen},
  year          = {2026},
  eprint        = {2609.17475},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2609.17475}
}
```
