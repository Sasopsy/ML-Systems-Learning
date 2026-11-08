# 15 — Blog case selection: five connected shapes

- **Working title:** “Faster Softmax Than PyTorch Eager (for Certain Shapes)”.
- **Recommendation:** five core case studies within the plotted width limit 16384,
  comparing **our retained Online** against PyTorch. Keep the full-range plots
  visible so the selected wins and losses have context.
- **Selection rule:** connected changes in width or row count, clear median gaps,
  coverage of all three PyTorch families, and paired profiles already available.
  These are questions for profiling, not causal conclusions.

## Primary comparison: CUDA Graph replay

- H100 80GB HBM3, contiguous FP32, preallocated outputs, softmax over columns;
  PyTorch 2.11.0+cu130. Same deterministic input fixture as note09.
- Times are medians of nine 500-call batch averages in microseconds/call.
  PyTorch chooses its kernel through eager dispatch; timing is graph replay,
  **not ordinary eager Python-call latency**. Ratio > 1 favors our retained kernel.

| Shape | PyTorch family | PyTorch µs | Retained Online µs | PyTorch / ours |
|---|---|---:|---:|---:|
| 37 × 2048 | Warp | 6.633 | 3.988 | **1.66×** |
| 37 × 4096 | Reg | 3.017 | 6.114 | 0.49× |
| 4096 × 4096 | Reg | 58.885 | 49.372 | **1.19×** |
| 4096 × 8192 | Reg | 145.427 | 100.415 | **1.45×** |
| 4096 × 16384 | General | 214.255 | 261.213 | 0.82× |

## Why these five?

1. **37×2048 → 37×4096:** our strongest retained win in the plotted range turns
   into a loss. PyTorch changes Warp→Reg and its measured time decreases even
   though the row gets wider. Investigate mapping, work per thread and reduction
   overhead; the family switch alone does not prove the cause.
2. **37×4096 → 4096×4096:** change only row count. The same Reg specialization
   changes from beating ours to losing. This is a useful question about scaling
   across the GPU, not evidence yet for a specific occupancy/latency explanation.
3. **4096×4096 → 4096×8192 → 4096×16384:** our advantage grows to 1.45×, then becomes
   a loss as PyTorch changes Reg→General. This covers the retained-data versus
   repeated-load tradeoff without assuming which resource limits either kernel.

**Start the profiling discussion with the first pair.** Record a hypothesis and
an alternative, then inspect only those four reports before moving on.

## Direct-launch cross-check

The same winner appears at all five shapes in direct mode. These intervals also
include GPU idle gaps from host submission; PyTorch eager and the custom ctypes
bridge are different host paths. Do not use them alone to explain kernel speed.

| Shape | PyTorch family | PyTorch µs | Retained Online µs | PyTorch / ours |
|---|---|---:|---:|---:|
| 37 × 2048 | Warp | 7.633 | 4.873 | **1.57×** |
| 37 × 4096 | Reg | 5.065 | 6.982 | 0.73× |
| 4096 × 4096 | Reg | 60.683 | 50.681 | **1.20×** |
| 4096 × 8192 | Reg | 146.931 | 102.817 | **1.43×** |
| 4096 × 16384 | General | 215.990 | 262.812 | 0.82× |

## Selection limits

- Recomputed 20 selected medians from 180 raw batch samples. For each selected
  shape/mode, all nine faster-variant batch averages are below all nine slower
  ones in this saved run. This avoids selecting a near tie; it is not a confidence
  interval or evidence of reproducibility across independent runs/devices.
- Widths≤8192 come from the Oct27 run; width 16384 comes from the Oct31 extension.
  Each within-shape speedup is matched, but the 8192→16384 progression crosses runs.
  The user confirmed the same H100/environment and repeated measurements; no rerun
  is a prerequisite. Keep session provenance and distinguish profile-supported
  explanations from isolated causal experiments.
- Width 512's small wins are less useful opening cases. Rows 256 largely repeat the
  small-row pattern. Widths>16384 extend beyond the current plot scope and can be
  a later large-width case study. Keep non-retained Online separate from this story.
- Exact-width specialization, input-distribution sensitivity and cache/clock
  conditions remain part of the experiment's limits; see note09. No new profiles
  or causal diagnoses were produced during selection.

## Selected profiles for the next step

Only report existence was checked here; profiler counters were not analyzed for
this selection. These downloaded artifacts are not tracked in Git.

| Shape | PyTorch report | Our report |
|---|---|---|
| 37 × 2048 | [PyTorch](../../results/2026-10-27-offline-profiles/37x2048/pytorch.ncu-rep) | [Retained Online](../../results/2026-10-27-offline-profiles/37x2048/online_registers.ncu-rep) |
| 37 × 4096 | [PyTorch](../../results/2026-10-27-offline-profiles/37x4096/pytorch.ncu-rep) | [Retained Online](../../results/2026-10-27-offline-profiles/37x4096/online_registers.ncu-rep) |
| 4096 × 4096 | [PyTorch](../../results/2026-10-27-offline-profiles/4096x4096/pytorch.ncu-rep) | [Retained Online](../../results/2026-10-27-offline-profiles/4096x4096/online_registers.ncu-rep) |
| 4096 × 8192 | [PyTorch](../../results/2026-10-27-offline-profiles/4096x8192/pytorch.ncu-rep) | [Retained Online](../../results/2026-10-27-offline-profiles/4096x8192/online_registers.ncu-rep) |
| 4096 × 16384 | [PyTorch](../../results/2026-10-31-offline-profiles-128k/4096x16384/pytorch.ncu-rep) | [Retained Online](../../results/2026-10-31-offline-profiles-128k/4096x16384/online_registers.ncu-rep) |

## Evidence

- [Combined timing summary](../../results/2026-10-31-softmax-128k/combined_summary.csv).
- Raw batch measurements: [Oct27](../../results/2026-10-27-softmax-sweep/rounds.csv),
  [Oct31](../../results/2026-10-31-softmax-128k/rounds.csv).
- [Full inventory and timing scope](09-results-and-kernel-inventory.md) ·
  [Full-range plots](10-runtime-plots.md) · [Section index](README.md).
