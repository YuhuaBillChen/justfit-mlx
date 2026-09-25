# JustFit on M3 Pro — completed core paper-workload comparison

## Outcome

**All six selected runs completed and passed validation** on this Mac: three fresh-process32K+6K repeats, fully cold240K+16K B1, cold2×(128K+16K) B2, and incumbent128K+3×8K B4 with12K outputs per stream. No guard trip, relaxed-setting retry, or runtime-source repair was needed.

This is a **public-kit reconstruction with disclosed differences**, not a claim that every historical paper environment was reproduced exactly or that GPU generation is the only cause of the performance differences.

The benchmark server is stopped and port8080 was verified closed. The six server lifetimes totaled about **7h35m**; the10 main requests processed761856 input and116736 output tokens. Health checks are additional and excluded from main-request timing.

Reference: [JustFit paper v2](https://arxiv.org/html/2609.17475v2), especially Tables4–5 and12 and AppendicesB.4/E, plus the pinned `examples/justfit/results/` manifests. `K=1024`.

## Throughput comparison

All rates are tokens/s. Fixed-workload values are independent medians of three fresh processes. Each long workload has one local run, compared with its corresponding published cohort. B2/B4 TG is **aggregate common-active throughput**, not per-request speed.

| Workload | M3 PP | Published M4 PP | M3 TG | Published M4 TG | M3/M4 TG |
|---|---:|---:|---:|---:|---:|
| B1 32K input +6K output, n=3 median |87.109|114.370|12.603|18.284|68.9%|
| B1 fully cold240K+16K |44.403|60.546|4.944|5.936|83.3%|
| B2 cold2×(128K+16K) |57.262|81.399|7.101|10.341|68.7%|
| B4 128K incumbent then3×8K,12K output each |62.888|85.452|16.800|22.570|74.4%|

Do not turn these four workload-specific ratios into a universal M3/M4 speed factor. In particular, B2 has material scheduling/occupancy and host-pressure differences described below.

### Main-request latency

- **Full B1:**8849.779s (**147.50min**) here versus6820.029s (**113.67min**) published; about1.298× the wall time.
- **Cold B2:**9619.614s (**160.33min**) here. Wall-output rate is3.406tok/s versus5.113tok/s published. Its exact original preparation cadence is not fully archived.
- **B4:**5554.808s (**92.58min**) here versus4046.521s (**67.44min**) published. Wall-output rate is8.849tok/s versus12.147tok/s.
- Fixed32K+6K main wall times:906.363,858.839,863.255s. The median is863.255s; the paper's separate PP/TG medians are not substituted for an unreported measured wall-time median.

These timings include the main requests' prefill and generation, but not startup, preflight, or postflight. The sampled process peak covers the whole measured server lifetime.

## Memory and occupied state

| Workload | M3 sampled peak MiB | Published M4 peak MiB | M3 sampled margin below21000MiB |
|---|---:|---:|---:|
|32K+6K median|15650.43|15626|5349.57|
|Full cold B1|20904.995|20357|95.005|
|Cold B2|20749.089|20858|250.911|
|Incumbent B4|19285.073|18915|1714.927|

**The full B1 run completed the complete262144-position input-plus-output window.** Its narrow95MiB sampled margin is not a continuous-memory guarantee or a recommended daily-serving headroom target.

| Long workload | Cumulative completed input/output positions | M3 simultaneous page high-water | Published M4 high-water |
|---|---:|---:|---:|
|Full B1|262144|262144|262144|
|Cold B2|294912|286720|294400|
|Incumbent B4|204800|201728|201728|

High-water counts **page-capacity slots**, not a per-token census of initialized KV entries. Completed cumulative work, allocated pool capacity, and simultaneous occupied state are different quantities.

### Why B2 needs an explicit qualification

Both128K requests were submitted concurrently, but the public runtime prepared them using N4/PF64: the first stream generated8197 tokens before the second's first token. That changes preparation overlap, the common interval, and the state present before the first stream finishes.

The measured common interval contains **16372 tokens over2305.433671583s**. Both requests still complete all16384 outputs, but the observed simultaneous occupancy is286720—not the historical294400-slot result. The original cold-B2 cadence/arrival environment is not completely specified in the public archive. This row therefore does **not** establish an identical historical capacity/latency protocol or isolate a chip-only difference. No settings were retuned to force the reference occupancy.

B4 follows its separately documented incumbent-first arrival: the three peers were submitted5.532–5.629ms after the incumbent's first received text; actual server prefill admission followed its first generated token by2.553–2.636s. Its four-stream common interval contains42944 tokens over2556.158116583s. The reference counts42945 tokens over its own interval; timing boundaries are observed rather than manufactured to match a literal count.

## Validation and artifact integrity

- All main requests reported exact input/output totals, cached0, `finish_reason=length`, one `[DONE]`, and no stream error.
- All six fresh servers passed one-token preflight and same-process8192+64 postflight reuse.
- Fixed repeats have identical token-ID and text hashes. Both B2 outputs match full B1's16384-token sequence; all four B4 outputs match each other at12288 tokens.
- Independent read-only audits recomputed the first fixed result and all three long workloads from raw SSE, token events, footprint samples, and logs. They confirmed the aggregate results and lifecycle transitions.
- Final verification re-ran the frozen analyzer, checked every expected lane shape and stream, confirmed six distinct server PIDs, and verified all19 archived measurement-source hashes and unchanged dependency versions.
- Model/tokenizer/component files were rehashed **after timed inference stopped** and matched the initial manifest. Extracted head and embedding tensors were compared byte-for-byte with the pinned target tensors; both matched, with two tensors and675430400 payload bytes each.
- Final regression run: **115 tests passed, plus20 subtests, in7.13s**, including the documented Metal numerical/lifecycle suite and measurement-harness controls.
- No tracked upstream runtime files were edited.

Authoritative final integrity result: `core-v1/final-verification.json`; test log: `core-v1/final-tests.log`.

## Sampling and host conditions

The watchdog targeted250ms sampling and stopped at integer-MiB footprint>=21000. It sampled macOS physical process footprint, not RSS or MLX-active memory. Actual timing was recorded:

| Cohort | Maximum lifetime sampling gap | Maximum gap overlapping main request |
|---|---:|---:|
|Fixed repeat1|1.5242s|0.2634s|
|Fixed repeat2|0.2700s|0.2700s|
|Fixed repeat3|0.2696s|0.2696s|
|Full B1|0.2702s|0.2702s|
|Cold B2|0.3939s|0.3939s|
|B4|0.5174s|0.2697s|

The two lifetime gaps above0.5s occurred during loading, before the main requests. **No main-request sampling gap exceeded0.5s**, but no sampled watchdog proves absence of intersample spikes.

All recorded host samples used AC power, and monitoring commands reported no errors. `pmset` reported no recorded thermal/performance warnings; that is not a direct GPU-temperature/clock measurement or proof of no throttling. An AC-policy spot check showed low-power mode off.

**B2 experienced changing host pressure:** swap usage went from2857.06M to6594.94M, reaching7019.00M in `vm.swapusage`. Those are usage counters, not swap-I/O rates. The Mac was not a controlled no-swap host. Other applications, storage/cache state, OS behavior, and thermal policy remain confounders.

## Configuration and fidelity boundaries

- Local hardware: **M3 Pro,36GiB,12 CPU cores (6P+6E),18 GPU cores**; macOS27.0/build26A428.
- Reference hardware: M4 Pro/24GiB. Exact CPU/GPU bin is not given in the inspected paper/manifests.
- Source: `justfit-repro-v4`, commit `3cc6e0cf41f1d0ed939a9702926bd62642a3fef8`.
- Python3.12.12, MLX/mlx-metal0.32.2, Transformers5.17.0; full dependency snapshot in the suite.
- Target: `mlx-community/Qwen3.8-27B-mxfp4` at `97ab0819817ab1c61d7d39f9169fc71999915641`.
- Components: `billchen42/JustFit-Qwen3.8-27B-components` at `13e0462fa911a1bcbdccbaa759120500c0f90582`.
- Retained paged Q4, two-KV-head-group direct-inverse prefill, eager release, PhaseSwap, deferred MTP width3, singleton MTP/multi-request AR, and N4/PF64 scheduling.
- APC enabled, fresh disk directory per process, explicitly disk-only with borrowed synchronous writes. Hash-only tenant separation prevents health checks/peers from creating main prefix hits. Complete historical APC settings are unavailable, so these are declared reconstruction choices.
- The32K pool/admission settings and some cold-B2 controls were reconstructed explicitly, not recovered from a complete original launch manifest.
- The system's `iogpu.wired_limit_mb` stayed at0/default; the paper's public setup specifies22016. No global memory settings were changed. Equal process guards do not emulate a24GiB host.
- The original EVO10 dirty-source fingerprint and original instrumentation/full environment are not completely public. Current B4 generation source matches the published fingerprint, but that does not make every historical cohort an identical v4 run.
- Peer labels in `metrics.json` use canonical same-shape mapping. Do not pair a peer's client latency with its canonically assigned server ID as an exact identity; aggregates are independently verified. Raw client results and event traces remain available.

Inputs are deterministic repeated words with EOS suppression. These runs establish execution/lifecycle behavior and workload-specific speed—not coding-agent quality, full-window comprehension, or an AIME score. The selected scope **excludes** the warm320K B2 headline, warm B1 repetitions, historical baseline/ablation stages, images and AIME. No claim is made to reproduce the entire paper.

For hardware-spec context and why it does not establish causality, see `HARDWARE-CONTEXT.md` with official Apple sources.

## Deliverables and rerunning

- `core-v1/COMPARISON.md` and `comparison.json`: full comparison, occupancy and host tables.
- `core-v1/final-verification.json` / `.log`: final source, dependency, artifact, tensor and evidence checks.
- `core-v1/final-tests.log`: final115-test result.
- Per-run directories: frozen spec/environment, raw SSE, client timestamps, server token telemetry, logs, physical-footprint samples, host snapshots, metrics and audits.
- `core-v1/harness-source/`: copies of the helpers actually used; `source-hashes.json` records fingerprints.
- `README.md`, `protocol.json`, and helper scripts: setup, rerun and measurement instructions.
- Downloadable bundle: `justfit-paper-core-evidence.tar.gz` in the session workspace, excluding large model files and APC cache directories. Those remain locally.

The checkout is `/Users/chyuhua/workspace/temp/justfit-mlx`. To repeat the selected batch after reviewing the caveats:

```bash
cd /Users/chyuhua/workspace/temp/justfit-mlx
.venv/bin/python paper-repro/run_suite.py --mode core
```

This creates a new timestamped suite rather than overwriting `core-v1`. Keep the Mac plugged in and idle; use one server only. See the README for qualification and post-suite verification commands.

The complete checkout now occupies about **49GiB**, including roughly **32GiB of paper-run artifacts, mostly generated APC caches**. These were retained for inspection; the compact evidence bundle does not include those caches. Removing only run-generated APC cache directories later can recover substantial space without deleting the downloaded model, scripts or logs.
