# 2026-11-06 — First softmax profiling comparison

## Context

- Project3 softmax blog, TUTOR analysis; user asks why retained Online wins at
  37×2048 but PyTorch wins at37×4096. Starting commit `b664fcb`, local Mac.
- Initial hypothesis: mapping/per-thread work and latency hiding dominate the
  reversal. Alternatives: fewer instructions, less traffic, spilling, bandwidth.
- User confirmed same H100/environment and repeated timing; do not make a rerun
  prerequisite solely because earlier selected widths span separate sessions.

## Work and evidence

- Read four detail/raw/SASS exports in `results/2026-10-27-offline-profiles`,
  recorded source and NVIDIA scheduler/metric definitions. Delegated independent
  read-only scheduler and memory/SASS reviews; no delegated file edits.
- Added `notes/05-softmax-comparison/16-small-row-count-profile-crossover.md` under
  Project3 with mapping, counter tables, explanation, limits and source/report links.
- Raw CSV parsing used the units row before data. Built the counter table from
  selected exact metric names, kept benchmark timing separate from replay duration.
- Profiles:2048 Torch10 blocks/one warp per row versus ours37/four;4096 both37
  blocks but Torch32 warps per row versus ours4. Active-cycle issue20.69/15.77%
  at2048;59.72/16.66% at4096. Higher total instructions do not determine the loser.
- Same global input/store sectors within shape, zero spills/local sectors, DRAM
  throughput<0.31%, L2 read hit>99.75%. Supports no bandwidth/spill explanation;
  does not exclude memory latency or establish benchmark cache state.
- SASS review supports per-thread input counts64/16/4/32 and online recurrence
  structure. Static code sites are not dynamic counts. Raw SM stats are rollups,
  not explicit per-instance placement counts.
- Checked note local links, generated metric table and `git diff --check`.
  No new benchmark, CUDA compilation, GPU run or kernel implementation changes.

## Learning and handoff

- Evidence supports mapping + dependency hiding + local work; no isolated
  attribution of speedup fractions. Explanation assisted, independent mastery
  unassessed; learner conclusion pending.
- Next: discuss this case, then37×4096 versus4096×4096 (fixed width, more rows).
- No commit/push; existing work and GEMM resume preserved. Large reports remain
  separate downloaded artifacts.

## Follow-up — independent audit and algorithm linkage

- Existing analysis was present at turn start; preserved it and independently
  rechecked the four raw exports and static SASS site counts. Requested a read-only
  agent audit of causal interpretation, normalization and alternative causes.
- Confirmed all main scheduler, instruction, memory-sector and spill counters.
  Added matched global-load instruction counts (2368/4736) and their distribution
  across valid row-serving warps; expanded the explicit stable-versus-online
  computation explanation at both widths.
- Avoided treating static EX2 site counts as dynamic exponentials, active-cycle
  issue as whole-GPU coverage, or grid counts as observed per-SM placement.
- Checked local links and `git diff --check`; no GPU execution, new benchmark,
  kernel changes, commit or push. User-facing explanation remains assisted.
