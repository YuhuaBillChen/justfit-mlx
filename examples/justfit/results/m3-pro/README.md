# Supplementary M3 Pro reconstruction

The primary JustFit evaluation remains **M4 Pro / 24 GiB**. This additional
device run exercises selected workloads using the public `justfit-repro-v4`
kit; it is not a controlled chip-only comparison or a full paper reproduction.

All six selected runs completed: three fresh-process fixed-workload repeats
and one run each of the three long workloads. The physical process-footprint
watchdog was **21,000 MiB**. The additional host was an M3 Pro (12 CPU / 18 GPU
cores, 36 GiB unified memory); the same process guard does not reproduce the
M4 host's total-memory constraints.

| Workload | n | PP tok/s | TG tok/s | Sampled peak MiB |
|---|---:|---:|---:|---:|
| B1 32K input + 6K output | 3 | 87.109 | 12.603 | 15,650.43 |
| B1 fully cold 240K + 16K | 1 | 44.403 | 4.944 | 20,905.00 |
| B2 cold 2 × (128K + 16K) | 1 | 57.262 | 7.101 | 20,749.09 |
| B4 128K incumbent + 3 × 8K; 12K output each | 1 | 62.888 | 16.800 | 19,285.07 |

K = 1,024. The fixed row uses independent medians of three runs. PP is input
tokens / summed uncached prefill work seconds, not input tokens / TTFT.
B1 TG excludes the first output token from its numerator. B2/B4 TG counts
real server token events strictly inside the common first-to-last-token
interval; it is aggregate throughput, not per-lane speed or wall throughput.

## Evidence and verification

- [results.json](results.json): configuration, full-precision results, provenance,
  original evidence-bundle SHA-256 and limitations.
- [independent-audit.json](independent-audit.json): independently recomputed
  metrics and hashes of the raw files consumed by the audit.
- [verify_evidence.py](verify_evidence.py): read-only standard-library audit of
  raw SSE, token events, sampled footprint and pool logs. It does not execute
  code from the evidence bundle or load a model.
- [REPORT.md](REPORT.md) and [COMPARISON.md](COMPARISON.md): original contributor
  reports, retained with their environment and historical-comparison caveats.

The separately supplied `justfit-paper-core-evidence.tar.gz` contains the raw
traces, frozen harness, launch protocol and original final-verification files;
it excludes model weights and APC cache blobs. Its SHA-256 is
`99b0029577f04a15a1642a8aab381aacaf54e2379643343b66f344d56f3b1bdc`.
No public asset URL is assigned by this source change. After obtaining and
extracting that bundle, reproduce the independent audit with:

```sh
python3 verify_evidence.py /path/to/paper-repro/core-v1 > audit-local.json
diff -u independent-audit.json audit-local.json
```

All six rows match the archived metrics. Main streams have exact input/output
counts, zero prefix hits, one DONE marker and length termination. Fixed-repeat
token/text hashes agree; the B2 outputs match the long B1 output. These are
local consistency checks, not matches to unavailable historical M4 traces.
The supplied original verifier also reports source/model integrity and
postflight success; our independent audit does not repeat model hashing or
the original Metal tests on the measurement host.

## Interpretation limits

B2 completed 294,912 input/output positions cumulatively, but its observed
simultaneous high-water was **286,720 page-capacity positions**. The first
stream generated 8,197 tokens before the second stream's first token. This
preparation cadence differs from the historical M4 run. B4's simultaneous
high-water was 201,728, versus 204,800 cumulatively completed positions.

The B1 peak leaves only about 95 MiB below the sampled guard; this is not a
continuous-memory guarantee or a daily-serving headroom recommendation.
B2 system swap usage rose from about 2,857 to 6,595 M (peak 7,019 M); usage
counters alone do not measure swap-I/O rates. The wired-memory setting stayed
at the host default, unlike the published M4 setup. Hardware, host pressure,
APC choices and historical scheduling therefore remain confounders.

Warm B1 repeats, warm 320K B2, vision, AIME, historical EVO ablations and Dash
were outside this suite. Long-workload results remain n=1 observations.
