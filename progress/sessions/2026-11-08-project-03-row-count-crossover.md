# 2026-11-08 — Softmax crossover from37 to4096 rows

## Context

- Project3 softmax blog; TUTOR profiling explanation. User requests algorithm-linked
  explanation at fixed width4096 when row count grows37→4096.
- Local Mac, existing source/profile downloads. Initial hypothesis: independent
  row supply makes smaller blocks' lower aggregate work useful; alternatives
  include cache/memory regime, spilling, changed kernel or branch behavior.

## Work and evidence

- Read learning instructions/status/project/latest relevant session; four profile
  README/raw/detail exports, timing summary, algorithms and source/SASS evidence.
- Delegated independent read-only scheduler/resource and memory/SASS audits.
  No delegated edits. Added note17 and index/current progress updates.
- Raw counters show resident-block limits2(Reg) and9(ours), our active scheduler
  warps1→8.11, issue16.66→58.57%, Reg issue59.72→68.36%. Reg retains higher occupancy.
- Instructions per row remain9661 Reg/~6774.6 ours (~30% less). Shared LD/ST counts
  are66 each per row versus5. Source/kernel specializations and per-thread work
  unchanged;32 versus4 warps per row and6 versus2 block barriers.
- Equal global sectors; no spilling. Verified device L2=52,428,800 bytes. Input
  grows606,208→67,108,864 bytes. L2 read hit~99.9%→~0.0067%; DRAM traffic largecase
  ~133.83MB Reg/~132.27MB ours, throughput2.23/2.67TB/s. Kept units/normalizations.
- Used exact named raw metrics and per-row normalization; source/SASS static
  sites not interpreted as dynamic instruction totals. Waves not treated as
  measured serial rounds. Graph medians3.017/6.114 and58.885/49.372µs remain
  separate from profiler durations4.192/7.328 and60.000/49.632µs.
- Checked note metric table, local links, and `git diff --check`. No CUDA run,
  performance experiment or code modifications.

## Learning and handoff

- Evidence supports inter-row concurrency plus less work, with co-varying memory
  behavior; no isolated percentage attribution. Assisted explanation, learner
  conclusion/independent transfer pending.
- Next: discuss this case, then4096×8192 versus4096×16384.
- Changes local/uncommitted; no commit/push; previous work/GEMM resume preserved.
  Downloaded report/source artifacts are not tracked in Git.
