# JustFit

**Run a 27B model over its full 256K-token context — and share that context
across concurrent requests — inside 24 GiB of unified memory.**

> Built on [**MLX**](https://github.com/ml-explore/mlx), Apple's array framework
> for Apple silicon, and [**mlx-vlm**](https://github.com/Blaizzy/mlx-vlm) by
> Prince Canuma, which this project extends and ships in full. JustFit adds a
> serving runtime on top of their work; the models, kernels, and loading paths
> underneath are theirs. See [Built on mlx-vlm](#built-on-mlx-vlm).

[Paper](https://arxiv.org/abs/2609.17475) ·
[Project page](https://yuhuabillchen.github.io/mlx-vlm/) ·
[Quick Start](examples/justfit/quickstart.md) ·
[Reproduction kit](examples/justfit/README.md) ·
[Serving guide](docs/justfit-serving.md) ·
[Pinned components](https://huggingface.co/billchen42/JustFit-Qwen3.8-27B-components)

JustFit is an MLX inference runtime for long-context local serving. Quantizing
the KV cache is only the first step: usable context is set by every allocation
that overlaps it — weights, resident pages, attention operands, prefill
workspace, speculative state, component lifetimes. JustFit treats their overlap
as the actual problem, and materializes each one only while it is needed.

On an Apple M4 Pro with 24 GiB, that moves Qwen3.8-27B from 30,720 completed
positions to the complete 262,144-position native window — **8.53×** — with a
measured peak of 20,357 MiB under a 21,000 MiB fail-closed guard.

## Install

JustFit is not on PyPI yet — install from source:

```sh
git clone https://github.com/YuhuaBillChen/justfit-mlx
cd justfit-mlx && pip install -e .
```

The import package is `mlx_vlm`, so existing mlx-vlm code keeps working.

To reproduce the paper's runs, use the pinned tree instead:

```sh
git clone --branch justfit-repro-v4 https://github.com/YuhuaBillChen/mlx-vlm
cd mlx-vlm && examples/justfit/run_server.sh
```

Start with the 8K+64 smoke profile in the
[Quick Start](examples/justfit/quickstart.md). Only attempt 240K+16K after
verifying the checkpoint, component files, thermal conditions, free disk space,
and a process-footprint monitor.

## How it fits

- **KVExec** keeps TurboQuant Q4 KV state in fixed-size pages and consumes those
  pages directly for decode and short MTP verification. Eligible prefill uses a
  direct-inverse reconstruction path and evaluates each layer before returning,
  bounding the lifetime of floating-point attention operands.
- **PhaseSwap** assigns explicit leases to components whose useful lifetimes do
  not cover the whole request: the output head, vision tower, input embedding
  table when configured, and the deferred MTP drafter.
- **StateTrans** changes between singleton MTP and multi-request AR at safe
  cohort boundaries while retaining surviving target KV and recurrent state.
  Completed or cancelled requests return page IDs to the resident pool only
  after pending state has been evaluated.

## Measured results

Apple M4 Pro MacBook Pro, 24 GiB unified memory; Qwen3.8-27B with fixed MXFP4
weights; 21,000 MiB sampled process-footprint guard. `K = 1,024`.

| Workload | Retained positions | PP | TG | Peak footprint |
| --- | ---: | ---: | ---: | ---: |
| mlx-vlm baseline, B1 24K input + 6K output | 30,720 | — | did not complete the next fixed step | below guard at this point |
| JustFit, B1 240K input + 16K output, fully cold | 262,144 | 60.546 tok/s | 5.936 tok/s | 20,357 MiB |
| JustFit, B2 2×(128K input + 16K output), fully cold | 294,912 aggregate | 81.399 tok/s | 10.341 aggregate tok/s | 20,858 MiB |
| JustFit, B2 160K+16K and 128K+16K, ordered warm-prefix restore | 327,680 aggregate | protocol-dependent incremental PP | 9.735 aggregate tok/s median | 20,432–20,953 MiB |
| JustFit, B4 128K+12K incumbent then 3×(8K+12K) | 204,800 aggregate | 85.452 tok/s | 22.570 common-active; 12.147 wall-output tok/s | 18,915 MiB |

In a separate capability protocol, paged TQ4 answered 29/30 AIME 2026 problems
correctly and generated 696,834 tokens at 15.04 token-weighted tok/s.

Full cohort detail, protocol boundaries, and the source map are in
[`docs/justfit.md`](docs/justfit.md).

## What these numbers do not claim

- Retained positions mean input **plus generated output**. The full-window B1
  result is 245,760 input + 16,384 output, not 262K input.
- 327,680 is aggregate across two requests, not one 320K conversation. The
  strongest single-request shape is the native 256K window.
- The B2 and B4 totals are aggregate state across concurrent requests. Their TG
  values are aggregate common-active-interval rates, not per-request rates.
- The capacity corpus is deterministic repetitive text with EOS suppression. It
  measures complete allocation and execution, not coding-agent task quality.
- The fully cold 240K+16K run takes about 6,820 seconds end to end. It is a
  memory stress test, not a daily preset.
- 29/30 on AIME 2026 is an integrated runtime check, not a 256K-context
  comprehension test.
- Returning pages makes resident pool slots reusable; it does not return the
  pool's backing allocation to macOS.

## Status

This is research software. The page geometry currently targets the evaluated
Qwen3.8 configuration (head dimension 256, page size 256, Q4 TurboQuant) and
does not claim general paged-attention support for every mlx-vlm model.

## Built on mlx-vlm

JustFit runs on [MLX](https://github.com/ml-explore/mlx), Apple's array
framework for Apple silicon. Its unified-memory model, lazy evaluation, and
quantization primitives are what make a memory budget something a runtime can
schedule against at all.

The project is a derivative of [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) by
Prince Canuma, and ships the whole library — every VLM, omni-model, and
fine-tuning capability upstream provides still works here. Model
implementations, kernels, and loading paths are upstream's work; JustFit adds
paged KV execution, component residency, and the serving layer above them. That
documentation is preserved at [`docs/mlx-vlm-usage.md`](docs/mlx-vlm-usage.md).

Please report bugs that reproduce on plain mlx-vlm, without the JustFit runtime,
to [upstream](https://github.com/Blaizzy/mlx-vlm/issues) rather than here.

## Citation

The paper's reproduction URLs resolve against the `justfit-repro-v4` tag in the
[research fork](https://github.com/YuhuaBillChen/mlx-vlm/tree/justfit-repro-v4/examples/justfit),
which stays available permanently so published links keep working.

```bibtex
@misc{chen2026justfit,
  title         = {JustFit: Just-in-Time State Management for Local LLM Serving},
  author        = {Yuhua Chen},
  year          = {2026},
  eprint        = {2609.17475},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2609.17475}
}
```

## License

MIT. Copyright © 2026 Yuhua Chen, and © 2025 Prince Canuma for the upstream
mlx-vlm portions. See [LICENSE](LICENSE).
