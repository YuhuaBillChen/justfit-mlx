# Paged runtime quality work

Initial configuration patch: `d98e883b`, isolated branch
`refactor/paged-runtime-config`. Production integration also preserves the
existing prompt-logits fix and adds idempotent generator cleanup.

The acceptance target is explicit interfaces, predictable ownership, reproducible
configuration, tested failure behavior, and measured numerical/performance
compatibility. This is an engineering target, not a claim of Apple review or
acceptance. Production features and long-context results alone do not establish
these properties.

## Implemented: construction-time execution policy

`PagedTurboQuantConfig` is an immutable, MLX-independent value object. A pool
registry resolves its policy before allocating tensors. All independently
created request caches and `new_empty()` caches inherit that same object.
Joining caches with different policies raises before transferring ownership.

Explicit use:

```python
from mlx_vlm.paged_turboquant_config import PagedTurboQuantConfig
from mlx_vlm.paged_turboquant_pool import PagedTurboQuantPoolRegistry

config = PagedTurboQuantConfig(
    prefill_impl="direct_inverse",
    prefill_eager_release=True,
    mtp_qtile=True,
)
registry = PagedTurboQuantPoolRegistry(layer_specs, config=config)
experiment_record["paged_execution"] = registry.config.to_dict()
```

Environment compatibility:

| Environment variable | Explicit field | Unset default |
| --- | --- | --- |
| `MLX_VLM_PAGED_PREFILL_IMPL` | `prefill_impl` | `compatibility` |
| `MLX_VLM_PAGED_PREFILL_EAGER_RELEASE` | `prefill_eager_release` | `False` |
| `MLX_VLM_TQ_MTP_QTILE` | `mtp_qtile` | `False` |

Omitting `config` snapshots these variables at registry construction, or at cache
construction for standalone caches. Explicit configuration ignores them.
Changing the environment after construction no longer changes these dispatch
choices. Invalid values now fail early rather than silently selecting a fallback.
Boolean environment values are `0`, `1`, or empty (disabled).

Dispatch order remains decode, eligible direct-inverse prefill, eligible MTP
qtile, compatibility prefill. Eager release preserves the existing `mx.eval`
boundary. No kernel arithmetic, layout, capacity, or scheduler policy is changed.

This conversion covers the paged facade's three choices. Legacy TurboQuant
helpers still have their own environment-based kernel controls. Therefore this
configuration record is not yet a complete reproduction manifest for all
TurboQuant behavior. Experiment collection is not automatically wired to it.

## Validation status

- On MBP: 69 paged tests and 9 subtests passed, including Metal numerical tests;
  the final generation/service/cleanup suite passed 745 tests with one skip.
- B1 24K+512 and B4 4*(2K+512) paired measurements showed no observed regression
  and matching output token hashes. This is one measurement per arm, not a
  statistically established speedup or a maximum-context qualification.
- The subsequent generator cleanup fix passed the final 745 tests and live
  broker checks for thinking, APC (2327 cached tokens), B4 and postflight.
  The performance pairs were not repeated after that cleanup-only change.
- Deployed to `/Users/bill/services/mlx-vlm-qwen-cb-quality-20260913`.
  The localllm release report is
  `docs/research/qwen38/paged_quality_production_release_20260913.md`;
  JSON and raw measurements are under `run/paged-quality-release-20260913/`.

CPU command (no MLX import required):

```sh
python3 mlx_vlm/tests/test_paged_turboquant_config.py
```

Mac verification command, from this checkout in an environment with project
dependencies installed:

```sh
python -m pytest -q \
  mlx_vlm/tests/test_paged_turboquant_policy_lifecycle.py \
  mlx_vlm/tests/test_paged_turboquant_cache.py \
  mlx_vlm/tests/test_paged_turboquant_pool.py \
  mlx_vlm/tests/test_paged_turboquant_storage.py \
  mlx_vlm/tests/test_paged_turboquant_kernel.py
```

## Remaining stages and acceptance criteria

1. **Ownership contract.** Introduce a cache-owned append/attention interface only
   after documenting RoPE offsets, causal masks, lazy array lifetimes, and
   reservation rollback. Reject unsupported inputs before mutation. Exercise
   append failure, attention failure, cancellation, and B1/B4/B1 transitions.
   Adding an `update_and_attend` wrapper alone does not make the operation atomic.
2. **Backend configuration.** Enumerate environment reads in legacy helpers and
   separate numerical policy from performance tuning. Freeze them at the owning
   model/generator boundary and test two differently configured instances in one
   process. Preserve current batch-invariance and exact-verifier guarantees.
3. **Numerical evidence.** Record max absolute error, relative error with an
   explicit near-zero policy, and RMS error against the same dequantized reference
   across B=1/2/4, T=1/2/3/4, page boundaries and fragmented layouts. Distinguish
   quantization error from kernel arithmetic error. Choose thresholds from this
   evidence; do not tighten tolerances merely to resemble another PR.
4. **Resource evidence.** Run matched baseline/refactor workloads with fixed
   weights, prompt/output lengths, seed, effective configuration and warmup.
   Record prefill/decode TPS separately, peak process footprint and MLX peak,
   allocation/page counts, cancellation and reuse. Include the standard 24K+6K
   workload and an established long-context workload before adopting a broader
   attention/storage redesign. This remains beyond the smaller configuration
   and cleanup release validated above.
5. **Reviewable extraction.** Separate cache storage, attention execution and
   serving integration by their dependencies. Keep APC and scheduler ownership
   explicit. Document unsupported geometry/masks and reproducible commands.
   Avoid splitting files without reducing coupling or clarifying responsibility.

These broader stages remain open. The configuration/cleanup release is deployed;
it is not a completed runtime redesign.
