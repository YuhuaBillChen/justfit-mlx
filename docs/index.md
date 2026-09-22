# JustFit

JustFit is an MLX inference runtime for long-context local LLM serving on Apple
silicon, built on [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) and
[MLX](https://github.com/ml-explore/mlx).

It runs Qwen3.8-27B over its complete 262,144-position native window — and
shares context across concurrent requests — inside 24 GiB of unified memory, by
coordinating compressed KV execution, phase-aware component residency, and
state-preserving serving transitions.

- [Overview and measured results](justfit.md)
- [Serving guide](justfit-serving.md)
- [Installation](installation.md)
- [mlx-vlm usage](mlx-vlm-usage.md) — the underlying library's own documentation
- [Paper](https://arxiv.org/abs/2609.17475)
