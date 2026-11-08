# Softmax comparison — blog working notes

- **Goal:** document shape-dependent wins and losses against PyTorch softmax,
  explain the algorithms, then build evidence-based performance hypotheses.
- [09 — Kernel inventory, timings and speedups](09-results-and-kernel-inventory.md):
  all30 shapes, both custom variants, PyTorch kernel names, graph/direct tables,
  measurement boundaries and provenance.
- [10 — PyTorch versus Online variants: runtime plots](10-runtime-plots.md): three row-count
  plots per variant (non-retained and retained) through width16384, PNG/SVG exports and a reproducible plotting script.
- [11 — PyTorch warp softmax algorithm](11-pytorch-warp-softmax.md): one paper-style algorithm box
  covering host/device steps, parameters, retained state and reductions, with
  explanatory notes and concrete shape instantiations underneath.
- [12 — PyTorch register softmax algorithm](12-pytorch-register-softmax.md): one block
  per row, retained logits, explicit local reductions and shared-memory block reductions.
- [13 — PyTorch general softmax algorithm](13-pytorch-general-softmax.md): three input
  passes, vector loads/stores, small temporary buffers and block reductions.
- [14 — Our retained online softmax](14-retained-online-softmax.md): retained logits,
  conditional online state updates, pair merging and four-warps-per-row mapping.
- [15 — Blog case selection](15-blog-case-selection.md): five recommended shapes
  covering wins, losses and connected crossovers; graph timings and direct cross-check.
- [16 — First profile crossover:37×2048 versus37×4096](16-small-row-count-profile-crossover.md):
  mapping, scheduler readiness, instruction work and memory evidence behind the winner change.
- [17 — Row-count crossover:37×4096 versus4096×4096](17-row-count-profile-crossover.md):
  independent row-blocks, per-row work and the change in memory traffic.
- [18 — Wide-row crossover:4096×8192 versus4096×16384](18-wide-row-profile-crossover.md):
  retained-register pressure, vectorized General, cache reuse and the winner flip.
- **Next:** discuss the wide-row crossover, then synthesize the five-case blog narrative.
  Causal performance explanations remain hypotheses until supported by evidence.
- Keep observations, hypotheses, and controlled test results distinct. These are
  working notes for a blog; no publication or mastery assessment has occurred.
- [Reading profiler metrics](../03-validation-and-measurement/02-reading-softmax-profiles.md).
- [All Project 3 notes](../README.md).
