# M3 Pro vs published M4 Pro — completed core comparison

**Public-kit reconstruction, not a controlled chip-only comparison.** The M3 has36GiB, the reference M4 has24GiB; both use a21,000MiB process guard. APC/arrival/instrumentation and host-setting differences are recorded in the frozen protocol. Pending/stopped rows are not completed capacity results.

| Cohort | Status | M3 PP tok/s | M3 TG tok/s | M4 TG tok/s | M3 / M4 TG | M3 sampled peak MiB |
|---|---|---:|---:|---:|---:|---:|
| fixed-32k-6k-r1 | validated | 83.723 | 11.935 | 18.284 | 0.653x | 15650.43 |
| fixed-32k-6k-r2 | validated | 87.109 | 12.736 | 18.284 | 0.697x | 15650.87 |
| fixed-32k-6k-r3 | validated | 87.241 | 12.603 | 18.284 | 0.689x | 15643.25 |
| b1-cold-240k-16k | validated | 44.403 | 4.944 | 5.936 | 0.833x | 20905.00 |
| b2-cold-128k-16k | validated | 57.262 | 7.101 | 10.341 | 0.687x | 20749.09 |
| b4-incumbent-128k-3x8k-12k | validated | 62.888 | 16.800 | 22.570 | 0.744x | 19285.07 |

For 32K+6K, the M4 reference is a three-process median; individual M3 rows are not matched medians. For B2/B4, TG is aggregate common-active throughput, not per-request throughput.

## Three-process fixed-workload summary

M3 median PP **87.109 tok/s**; median TG **12.603 tok/s** (0.689x published M4). Median sampled peak **15650.43 MiB**. Token-ID hashes agree across repeats: **True**.

| Three-process median | M3 Pro | Published M4 Pro | M3 / M4 |
|---|---:|---:|---:|
| Prefill tok/s | 87.109 | 114.370 | 0.762x |
| Decode tok/s | 12.603 | 18.284 | 0.689x |
| Sampled peak MiB | 15650.43 | 15626 | 1.002x |

## b1-cold-240k-16k — completed workload

| Metric | M3 Pro | Published M4 Pro | M3 / M4 |
|---|---:|---:|---:|
| Prefill tok/s | 44.403 | 60.546 | 0.733x |
| Decode tok/s | 4.944 | 5.936 | 0.833x |
| Sampled peak MiB | 20905.00 | 20357 | 1.027x |
| Main wall seconds | 8849.779 | 6820.029 | 1.298x |

Completed cumulative input/output work: **262,144 positions**. Observed simultaneous page high-water: **262,144 slots** in a **262,144-slot pool**.

## b2-cold-128k-16k — completed workload

| Metric | M3 Pro | Published M4 Pro | M3 / M4 |
|---|---:|---:|---:|
| Prefill tok/s | 57.262 | 81.399 | 0.703x |
| Decode tok/s | 7.101 | 10.341 | 0.687x |
| Sampled peak MiB | 20749.09 | 20858 | 0.995x |
| Wall-output tok/s | 3.406 | 5.113 | 0.666x |

Completed cumulative input/output work: **294,912 positions**. Observed simultaneous page high-water: **286,720 slots** in a **294,912-slot pool**.
Published M4 page high-water: **294,400 slots**. Do not equate cumulative work, configured pool capacity, and measured occupied state.
The observed public N4/PF64 preparation schedule differs in stream progression from the historical record; see SCHEDULING-NOTE.md in this run directory. This is not an isolated chip-only latency/capacity comparison.

## b4-incumbent-128k-3x8k-12k — completed workload

| Metric | M3 Pro | Published M4 Pro | M3 / M4 |
|---|---:|---:|---:|
| Prefill tok/s | 62.888 | 85.452 | 0.736x |
| Decode tok/s | 16.800 | 22.570 | 0.744x |
| Sampled peak MiB | 19285.07 | 18915 | 1.020x |
| Main wall seconds | 5554.808 | 4046.521 | 1.373x |
| Wall-output tok/s | 8.849 | 12.147 | 0.728x |

Completed cumulative input/output work: **204,800 positions**. Observed simultaneous page high-water: **201,728 slots** in a **204,800-slot pool**.
Published M4 page high-water: **201,728 slots**. Do not equate cumulative work, configured pool capacity, and measured occupied state.

## Observed sampling cadence

Target250ms is not a hard real-time guarantee. The table preserves lifetime stalls rather than deleting them from peak-memory evidence.

| Cohort | Lifetime maximum gap | Main-window maximum gap | Main gaps >0.5s |
|---|---:|---:|---:|
| fixed-32k-6k-r1 | 1.5242s | 0.2634s | 0 |
| fixed-32k-6k-r2 | 0.2700s | 0.2700s | 0 |
| fixed-32k-6k-r3 | 0.2696s | 0.2696s | 0 |
| b1-cold-240k-16k | 0.2702s | 0.2702s | 0 |
| b2-cold-128k-16k | 0.3939s | 0.3939s | 0 |
| b4-incumbent-128k-3x8k-12k | 0.5174s | 0.2697s | 0 |

## Recorded M3 host conditions

| Cohort | Swap used before / after (sysctl M) | Maximum observed swap used (M) | AC in all samples |
|---|---:|---:|---|
| fixed-32k-6k-r1 | 3857.00 / 3721.00 | 3857.00 | True |
| fixed-32k-6k-r2 | 3721.00 / 3585.00 | 3721.00 | True |
| fixed-32k-6k-r3 | 3585.00 / 3545.00 | 3585.00 | True |
| b1-cold-240k-16k | 3545.00 / 2857.06 | 3545.00 | True |
| b2-cold-128k-16k | 2857.06 / 6594.94 | 7019.00 | True |
| b4-incumbent-128k-3x8k-12k | 6594.94 / 5674.31 | 6594.94 | True |

Swap usage is not a swap-I/O rate; changing host pressure is a comparison confound. No recorded pmset warning is not proof of constant GPU clocks or no throttling. Original M4 host telemetry is not available here.

## Interpretation and evidence

- PP uses summed uncached prefill work; TG uses actual server token counts and timestamps. Wall-output throughput is separately recorded in each metrics.json.
- Physical-footprint sampling includes server lifetime; page high-water is separate from cumulative completed input/output positions. B4's204800 completed positions must not be called204800 simultaneous occupied positions.
- Each run directory retains raw streams, server events, usage, health samples, footprint, and pre/postflight results. Guard trips and validation failures are preserved, not treated as completed shapes.
- No original paper token digest is published. Local repeat/hash agreement is not a verified match to unavailable original traces.
- Reference source: https://arxiv.org/html/2609.17475v2 and examples/justfit/results/{evo10-capacity,b4-throughput}.json.
