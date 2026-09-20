---
license: apache-2.0
library_name: mlx
base_model:
- Qwen/Qwen3.8-27B
- mlx-community/Qwen3.8-27B-mxfp4
tags:
- mlx
- mlx-vlm
- speculative-decoding
- qwen3.8
- justfit
---

# JustFit Qwen3.8-27B components

Published checkpoint-side components for the
[JustFit reproduction kit](https://github.com/YuhuaBillChen/mlx-vlm/tree/production/qwen-paged-continuous-batching/examples/justfit)
and [paper](https://arxiv.org/abs/2609.17475).

This repository does not duplicate the target model. Download the pinned
`mlx-community/Qwen3.8-27B-mxfp4` target separately. The JustFit preparation
script extracts the input-embedding backing from that target locally, avoiding
another 644 MiB network download.

## Files

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `vision-bf16.safetensors` | 921,492,862 | `b27ad7963b3752b5cab9d225f9c0b033d0d59066e7ff357dac17d690a6ff7994` |
| `mtp/model.safetensors` | 225,662,258 | `9b1d99f402f98940e08891d04815271e3fa94d6bc762f042fd311d3aa0d8bd1d` |
| `mtp/config.json` | 3,777 | `e0d0d5dc68f59559940e1b2dafc40a34700b9ae7ff963de268225a54550af563` |

- `vision-bf16.safetensors` is the exact BF16 backing used by the production
  PhaseSwap vision adapter. Its 333 tensors were matched tensor-for-tensor to
  the pinned public MXFP4 checkpoint.
- `mtp/` is the standalone block-size-3 MXFP4 MTP drafter. Its weight file is
  byte-identical to the measured artifact.

## Download

```bash
hf download billchen42/JustFit-Qwen3.8-27B-components \
  --revision 13e0462fa911a1bcbdccbaa759120500c0f90582 \
  --local-dir ./justfit-components
```

For the full launcher, capacity client, pinned target revision, limitations,
and result definitions, use the linked reproduction kit rather than treating
these files as a standalone model.
