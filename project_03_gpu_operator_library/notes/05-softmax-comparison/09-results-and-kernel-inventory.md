# 09 — Softmax comparison: kernel inventory and measured speedups

## Scope and reading the tables

- **Purpose:** evidence for a blog about where our custom softmax kernels outperform
  PyTorch's selected implementation and where they become slower. This first step
  inventories kernels and measurements; algorithm/profiling explanations follow.
- **Workload:** contiguous FP32 input/output, softmax over each row, preallocated
  outputs, NVIDIA H100 80GB HBM3, PyTorch 2.11.0+cu130. All three variants use the
  same deterministic synthetic input: `((flat_index % 41) - 20) * 0.25`.
- **Coverage:** 30 shapes × three variants × two timing modes. This is the matched
  sweep, not every earlier standalone correctness or microbenchmark experiment.
- **Speedup = PyTorch median / custom median**, within the same shape, run and mode.
  `1.50×` means 1.5 times as fast (one-third less time); `0.50×` means twice as slow.
- **Bold custom cells:** lower observed median than PyTorch, not necessarily the
  fastest of all three. Unbolded custom cells are slower. PyTorch's reference ratio
  is always `1.00×`; ratios use unrounded medians, then round to two decimals.
- Timings are **microseconds per call**, estimated as the median of nine batch
  averages, each batch containing 500 calls. These are not per-call latency
  distributions or statistical significance tests.

## Kernel inventory

| Table label | Widths in this sweep | PyTorch kernel and specialization |
|---|---|---|
| Warp | 128 / 512 / 1024 / 2048 | `softmax_warp_forward<float,float,float,L,false,false>`; `L` = 7 / 9 / 10 / 11 respectively |
| Reg | 4096 / 8192 | `cunn_SoftMaxForwardReg`; FP32 input/accumulator/output, `SoftMaxForwardEpilogue`, final template parameter 4 / 8 respectively |
| General | 16384 / 32768 / 65536 / 131072 | `cunn_SoftMaxForward<4,float,float,float,SoftMaxForwardEpilogue>` |

- **PyTorch:** ordinary `torch.softmax(x, dim=-1, out=y)`, automatic eager operator
  dispatch. Graph mode captures those calls; no `torch.compile`/Inductor variant.
  Labels above omit namespace and pointer-signature noise from the profiler.
- **Online:** `softmax_online<128>(x, y, dims)`; runtime width, conditional online
  update, rereads inputs during output. One block of 128 threads per row.
- **Retained online:** `softmax_online_registers<128, dims>(x, y)`; exact compile-time
  width, retains assigned inputs in a private array. One block of 128 threads per
  row. The name does not guarantee register residency: large widths spill.
- Both custom variants are timed at every shape. Paired full profiles contain
  **PyTorch and retained online only**: 60 reports over 30 shapes. Do not use the
  retained variant's counters to explain the non-retained variant's runtime.

## Graph replay: primary kernel comparison

- Separate CUDA Graphs capture 500 calls for each variant, followed by a warm
  replay and timed replays. CUDA events measure GPU elapsed time divided by 500.
- The underlying PyTorch kernel comes from eager dispatch, but these timings are
  **CUDA Graph replay**, not ordinary eager Python invocation latency.
- Retained online beats PyTorch on **9/30** shapes; online beats it on **7/30**.

| Shape [rows × columns] | PyTorch family | PyTorch µs | Online µs (speedup) | Retained online µs (speedup) |
|---|---|---:|---:|---:|
| 37 × 128 | Warp | 1.743 | 1.948 (0.89×) | 1.832 (0.95×) |
| 37 × 512 | Warp | 2.590 | 2.988 (0.87×) | **2.435 (1.06×)** |
| 37 × 1024 | Warp | 3.940 | **3.609 (1.09×)** | **2.975 (1.32×)** |
| 37 × 2048 | Warp | 6.633 | **4.795 (1.38×)** | **3.988 (1.66×)** |
| 37 × 4096 | Reg | 3.017 | 7.180 (0.42×) | 6.114 (0.49×) |
| 37 × 8192 | Reg | 4.361 | 11.777 (0.37×) | 9.739 (0.45×) |
| 37 × 16384 | General | 5.962 | 20.862 (0.29×) | 16.014 (0.37×) |
| 37 × 32768 | General | 9.560 | 44.667 (0.21×) | 38.088 (0.25×) |
| 37 × 65536 | General | 18.139 | 146.429 (0.12×) | 119.129 (0.15×) |
| 37 × 131072 | General | 38.858 | 393.754 (0.10×) | 469.588 (0.08×) |
| 256 × 128 | Warp | 1.997 | 2.273 (0.88×) | 2.151 (0.93×) |
| 256 × 512 | Warp | 2.701 | 3.164 (0.85×) | **2.604 (1.04×)** |
| 256 × 1024 | Warp | 4.051 | **3.787 (1.07×)** | **3.175 (1.28×)** |
| 256 × 2048 | Warp | 6.859 | **5.041 (1.36×)** | **4.384 (1.56×)** |
| 256 × 4096 | Reg | 4.506 | 7.543 (0.60×) | 6.889 (0.65×) |
| 256 × 8192 | Reg | 8.483 | 12.519 (0.68×) | 11.082 (0.77×) |
| 256 × 16384 | General | 11.625 | 33.177 (0.35×) | 18.955 (0.61×) |
| 256 × 32768 | General | 34.211 | 158.444 (0.22×) | 85.591 (0.40×) |
| 256 × 65536 | General | 93.806 | 315.142 (0.30×) | 307.061 (0.31×) |
| 256 × 131072 | General | 182.902 | 620.982 (0.29×) | 803.831 (0.23×) |
| 4096 × 128 | Warp | 2.549 | 6.638 (0.38×) | 5.591 (0.46×) |
| 4096 × 512 | Warp | 5.055 | 10.645 (0.47×) | 8.642 (0.58×) |
| 4096 × 1024 | Warp | 9.207 | 15.817 (0.58×) | 12.679 (0.73×) |
| 4096 × 2048 | Warp | 28.460 | 32.341 (0.88×) | **25.487 (1.12×)** |
| 4096 × 4096 | Reg | 58.885 | 81.342 (0.72×) | **49.372 (1.19×)** |
| 4096 × 8192 | Reg | 145.427 | 160.694 (0.90×) | **100.415 (1.45×)** |
| 4096 × 16384 | General | 214.255 | 317.428 (0.67×) | 261.213 (0.82×) |
| 4096 × 32768 | General | 654.465 | **624.104 (1.05×)** | 1,646.811 (0.40×) |
| 4096 × 65536 | General | 1,400.800 | **1,241.854 (1.13×)** | 4,481.342 (0.31×) |
| 4096 × 131072 | General | 2,805.656 | **2,534.497 (1.11×)** | 12,867.443 (0.22×) |

## Direct launches: eager PyTorch versus custom host submission

- CUDA events surround 500 Python-loop calls. PyTorch uses its eager operator path;
  custom kernels use a ctypes bridge. The GPU event interval can include idle gaps
  waiting for host submissions; it does not measure full end-to-end CPU/API latency.
- These comparisons describe the measured launch paths. A lower direct median
  alone does not establish a faster GPU kernel. Use the graph table alongside it.
- Both custom variants beat PyTorch on **11/30** shapes in this mode.

| Shape [rows × columns] | PyTorch family | PyTorch µs | Online µs (speedup) | Retained online µs (speedup) |
|---|---|---:|---:|---:|
| 37 × 128 | Warp | 4.633 | **3.498 (1.32×)** | **3.499 (1.32×)** |
| 37 × 512 | Warp | 4.484 | **3.725 (1.20×)** | **3.432 (1.31×)** |
| 37 × 1024 | Warp | 4.878 | **4.496 (1.08×)** | **3.852 (1.27×)** |
| 37 × 2048 | Warp | 7.633 | **5.691 (1.34×)** | **4.873 (1.57×)** |
| 37 × 4096 | Reg | 5.065 | 8.115 (0.62×) | 6.982 (0.73×) |
| 37 × 8192 | Reg | 5.341 | 12.659 (0.42×) | 10.591 (0.50×) |
| 37 × 16384 | General | 6.915 | 21.793 (0.32×) | 16.891 (0.41×) |
| 37 × 32768 | General | 10.342 | 45.406 (0.23×) | 38.872 (0.27×) |
| 37 × 65536 | General | 19.029 | 147.231 (0.13×) | 119.813 (0.16×) |
| 37 × 131072 | General | 40.290 | 395.154 (0.10×) | 471.884 (0.09×) |
| 256 × 128 | Warp | 4.856 | **3.682 (1.32×)** | **3.680 (1.32×)** |
| 256 × 512 | Warp | 4.950 | **4.056 (1.22×)** | **3.636 (1.36×)** |
| 256 × 1024 | Warp | 4.971 | **4.653 (1.07×)** | **4.070 (1.22×)** |
| 256 × 2048 | Warp | 7.868 | **5.914 (1.33×)** | **5.262 (1.50×)** |
| 256 × 4096 | Reg | 5.470 | 8.469 (0.65×) | 7.764 (0.70×) |
| 256 × 8192 | Reg | 9.319 | 13.250 (0.70×) | 11.895 (0.78×) |
| 256 × 16384 | General | 12.214 | 33.929 (0.36×) | 19.707 (0.62×) |
| 256 × 32768 | General | 35.940 | 160.073 (0.22×) | 87.206 (0.41×) |
| 256 × 65536 | General | 95.489 | 316.802 (0.30×) | 308.798 (0.31×) |
| 256 × 131072 | General | 184.568 | 622.689 (0.30×) | 805.483 (0.23×) |
| 4096 × 128 | Warp | 5.288 | 7.534 (0.70×) | 6.466 (0.82×) |
| 4096 × 512 | Warp | 5.973 | 11.461 (0.52×) | 9.401 (0.64×) |
| 4096 × 1024 | Warp | 10.248 | 17.216 (0.60×) | 13.916 (0.74×) |
| 4096 × 2048 | Warp | 30.197 | 33.928 (0.89×) | **27.150 (1.11×)** |
| 4096 × 4096 | Reg | 60.683 | 82.964 (0.73×) | **50.681 (1.20×)** |
| 4096 × 8192 | Reg | 146.931 | 162.336 (0.91×) | **102.817 (1.43×)** |
| 4096 × 16384 | General | 215.990 | 319.018 (0.68×) | 262.812 (0.82×) |
| 4096 × 32768 | General | 656.221 | **625.776 (1.05×)** | 1,648.589 (0.40×) |
| 4096 × 65536 | General | 1,402.304 | **1,243.532 (1.13×)** | 4,482.198 (0.31×) |
| 4096 × 131072 | General | 2,807.167 | **2,536.348 (1.11×)** | 12,868.926 (0.22×) |

## Claim boundaries for the blog

- **Supported:** these custom variants have lower observed medians than standard
  PyTorch softmax for the highlighted shapes under the stated measurement mode.
- **Not established:** online softmax is inherently faster, a general replacement
  for PyTorch, a model-level speedup, or a statistically robust win across devices,
  input distributions, PyTorch versions and execution settings.
- PyTorch uses compile-time specializations too. Our retained kernel additionally
  assumes an exact width. A runtime-width/bucket-specialization control experiment
  has not been performed; the present numbers do not isolate this difference.
- The deterministic input matters for the conditional online update. Additional
  distributions would be needed to generalize its branch behavior and timings.
- Observed large-width spills motivate a hypothesis, not a quantified causal
  explanation. Interpret algorithms and profiles before attributing speed changes.
- Caches/clocks were uncontrolled; buffers reused. Direct mode precedes graph mode
  in each shape. Small differences are observations, not established significance.
- Allocation, H2D/D2H transfers, reference computation and validation are excluded.
  Profiler replay durations are not used in either timing table.

## Evidence and reproducibility

1. Widths128–8192 come from `2026-10-27-softmax-sweep`; widths16384–131072 from
   `2026-10-31-softmax-128k`. These are **two runs**, not one controlled sweep of
   all widths. Every speedup compares variants within its own run.
2. Tables use the downloaded [combined summary](../../results/2026-10-31-softmax-128k/combined_summary.csv),
   with explicit per-row experiment provenance. All180 medians were checked
   against each source summary and recomputed from all1620 raw timing samples.
3. Sources: [earlier summary](../../results/2026-10-27-softmax-sweep/summary.csv),
   [earlier samples](../../results/2026-10-27-softmax-sweep/rounds.csv),
   [extension summary](../../results/2026-10-31-softmax-128k/summary.csv), and
   [extension samples](../../results/2026-10-31-softmax-128k/rounds.csv).
4. Names were checked against the30 captured kernel-pair entries in the
   [earlier profile inventory](../../results/2026-10-27-offline-profiles/comparison.csv)
   and [extension profile inventory](../../results/2026-10-31-offline-profiles-128k/comparison.csv).
5. Read the [saved harness](../../results/2026-10-31-offline-profiles-128k/source/sweep_softmax.py),
   [bridge](../../results/2026-10-31-offline-profiles-128k/source/softmax_sweep_bridge.cu),
   and [kernel snapshot](../../results/2026-10-31-offline-profiles-128k/source/softmax.cu)
   for measured behavior. Local working source is older; do not silently substitute it.
6. Recorded correctness uses independent NumPy FP64 references, fixed elementwise
   `atol=1e-7`, `rtol=1e-5`, row-sum tolerance `1e-5`, and finite/range checks.
   Initial and post-batch checks passed for this fixture. Sanitizer coverage is
   narrower than all shapes; see the source experiment READMEs.

- These numbers were compiled from the downloaded files on **2026-11-02**. Earlier
  prose may quote different saved medians; this table uses the current CSV files
  consistently and does not mix those older quotations into its ratios.
- Results/source snapshots are ignored by Git and require separate download;
  this note preserves the compiled tables for the tracked notes collection.
- **Next step:** explain the algorithms of Warp, Reg, General, Online, and Retained
  online; then form per-shape hypotheses and test them against profile evidence.
- [Section index](README.md) · [All Project 3 notes](../README.md).
