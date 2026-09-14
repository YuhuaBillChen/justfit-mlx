# Unified component residency

## Scope

The residency manager now lives in `mlx_vlm/server/component_residency.py`,
independently of MLX and the language-head adapter. `language_lifecycle.py`
re-exports the original name for compatibility. A structural
`ResidencyComponent` protocol specifies the existing `load`/`unload` contract;
it adds no runtime type checks or GPU synchronization in the manager.

The server registers configured vision, language-head and input-embedding
adapters, plus the deferred MTP drafter. Drafter leases pin the configured head
and embedding dependencies. Dependencies must be registered first and cannot
be changed after registration, preventing cycles and runtime policy drift.

The embedding adapter is retained on last-owner release; only the existing
warm-spill boundary explicitly requests idle eviction. Its B1-only gate remains.
Production B4 does not enable embedding spill: that table remains resident,
not newly offloaded by this change. The decode lease keeps the head resident
during mixed prefill/decode.

This is a lease/dependency coordinator, not a predictive eviction scheduler.
Numerical kernels, sampling, pool capacity and admission limits are unchanged.
At retirement, the speculative iterator is closed, target cache state evaluated,
and the batch's loaded-drafter alias dropped before component release. Each
speculative cohort uses a unique lease object. Repeated cleanup cannot release
a successor cohort's drafter. Non-managed callers retain the legacy adapter path.
The lazy wrapper retains three cumulative host-scalar counters when weights are
unloaded. Final-response statistics and snapshot/diff accounting therefore
survive early retirement and subsequent MTP re-promotion without retaining tensors.

## Ownership contract

- An owner name represents an idempotent lease, not a nesting reference count.
- A failed load does not publish a new owner.
- Failed dependency acquisition rolls back newly acquired leases only.
- Releasing one owner cannot unload a component held by another owner.
- Last-owner release consumes the lease before calling unload. If unload raises,
  the error propagates; retry via `unload_if_idle`, which checks for new owners.
- Failed component unload retains dependency leases until retry. A retained
  component does not unload just because its logical owner count reached zero.
- The serialized execution worker supplies the safety boundary. The manager
  does not synchronize GPU work or prove all external tensor aliases are gone.
- `loaded` on a language adapter means its model field is attached. It is not
  a measurement of physical residency. Unload need not return memory to the OS.

## Verification (2026-09-14)

Run without importing the MLX-dependent package initializer:

```sh
python mlx_vlm/tests/test_component_residency.py
```

Seventeen backend-independent tests cover registration, repeated acquire/release, multiple owners,
failed first/second load, failed unload and safe retry, successor ownership,
unknown components, immutable owner snapshots and independent components.
The original eleven contracts also passed against the pre-refactor manager.
Additional tests cover dependency ordering, rollback, failed unload and retention.

On M4 Pro with MLX, the final integrated suite passed 1,611 tests and nine
subtests, with one skip and two explicit expected failures. This includes warm
embedding eviction guards, restore after cancellation and cumulative drafter
statistics. The statistics regression fails before the scalar-retention fix.
The deployment report records paired performance measurements and live checks.

The expected failures are pre-existing tiny-Qwen dynamic multi-row MTP token
divergences. The former unseeded test could fail depending on random weights.
Baseline seed sweep 0..127 failed at 56 and 96; candidate sweep 0..63 failed
at 56, and both cases reproduce in the candidate's parameterized test. Fixed
seeds 0 and 1 remain ordinary passing tests. The failing cases are strict xfails,
not deleted assertions. This does not claim the underlying numerical issue is
fixed; production uses singleton-only MTP and multi-row AR, not multi-row MTP.

## Remaining limits

Input providers can retain embedding aliases. The existing B1 spill path delays
detach until its constructor's temporary providers have retired. General B4
embedding offload, provider-wide leases, per-page GPU events and physical-byte
attribution are not introduced. No per-chunk load/unload cycling is enabled.

Short paired B1/B4/staggered measurements are regression qualification, not a new
192K+16K capacity experiment or proof of no slowdown for every workload.

Native C++/Metal work is a separate decision after profiling load, materialize,
synchronization and reclamation costs. A language change alone does not remove
disk IO or GPU synchronization costs.
