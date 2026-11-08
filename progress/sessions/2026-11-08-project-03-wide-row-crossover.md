# 2026-11-08 — Width8192 to16384 softmax crossover

## Context

- Project3 blog, TUTOR profiling explanation. User asks why PyTorch wins again
  at4096×16384 after losing at4096×8192; local Mac and existing downloaded reports.
- Initial hypothesis: retained-state resource pressure reduces concurrency while
  General uses smaller temporary state/vector accesses; check spills, cache and
  instruction traffic rather than assuming extra passes triple DRAM traffic.

## Work and evidence

- Read current instructions/status/project/session; four raw/detail/README exports,
  timing summary and recorded source. Delegated read-only scheduler/resource and
  memory/SASS audits; no delegated edits.
- Added note18 and index/progress entries. Generated metric table from exact raw
  fields; all four captures use4096 blocks, PyTorch1024 threads and ours128.
- Registers/limits: Torch37/1→32/2, ours80/6→168/3 (registers per thread / blocks
  per SM). No spilling or local sectors at either width. Ours conditional update
  count63→127 after initialization; private input slots64→128.
- At16384 General vector load/store counts1,572,864/524,288 match three vec4 read
  passes and one vec4 output pass; ours2,097,152 each. Input sectors3× for General,
  yet DRAM reads274.981MB versus268.563MB. Direct cache hit counters support reuse.
  SASS corroborates vector main path; scalar prefix/tail/fallback sites also exist.
- Compared instruction advantage and scheduler readiness without treating higher
  occupancy or bandwidth as stand-alone causes. Branch uniformity improves for
  ours; worsening divergence is not supported. Kept profile timings distinct.
- Checked metric values, note links and `git diff --check`. No visual rendering,
  CUDA execution or new benchmark, and no source implementation changes.

## Learning and handoff

- Combined evidence supports retained-register cost plus General's vectorization,
  cache reuse and greater residency; exact contributions are not isolated.
- Worked explanation supplied; independent transfer/learner conclusion pending.
- Next: discuss wide-row crossover, then synthesize all five selected cases.
- Changes local/uncommitted; no commit/push. Earlier work and GEMM resume preserved;
  results/source snapshots require separate downloads.

## Follow-up — retained-online tradeoff and checkpoint

- Learner questioned whether online normalization is redundant when inputs are
  retained. Explained Warp's retained-exponential reuse versus our recurrence,
  rescaling and output exponentials; added this comparison to note 14.
- Proposed a conventional retained softmax with the same 128-thread mapping as
  a future control. It has not been implemented or benchmarked; independent
  understanding has not been reassessed.
- User then authorized committing and pushing accumulated work from starting
  revision `b664fcb`. This includes the DSA roadmap migration, GEMM notes and
  softmax study. Git history records the resulting checkpoint; earlier dated
  "local/uncommitted" statements describe their original sessions.
- Next: synthesize the five-case blog narrative. Raw profile/source downloads
  remain outside Git; no GPU run or kernel implementation in this follow-up.
