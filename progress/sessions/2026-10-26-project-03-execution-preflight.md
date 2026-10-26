# 2026-10-26 — Project 3 execution preflight

## Context

- Project3; TUTOR with user-authorized execution of existing kernels.
- Starting commit e6def5a; clean working tree. User reports substantial new work and requests GPU runs.
- Learning target: validate the learner's actual implementations before measuring performance; not infer mastery from compilation or passing checks.

## Work and evidence

- Read repository instructions, updated progress/STATUS.md, Project3 record and latest relevant 2026-10-20workflow session.
- `git fetch origin` discovered newer main; `git merge --ff-only origin/main` succeeded at f6d6002. No local changes discarded or conflicts resolved.
- `git ls-remote --heads origin` lists only main at f6d6002. `rg --files --hidden --no-ignore project_03_gpu_operator_library` lists only notes/README.md. The fetched revision contains workflow planning but no kernel source/harness.
- Asked user whether sources remain on Mac or are in another location/repository.
- `nvidia-smi` reports H10080GBHBM3, driver580.173.02 and no listed compute processes; `nvcc --version` reports13.0.88.
- Checked installed package metadata in kvforge: torch2.11.0+cu130, triton3.6.0, tilelang0.1.9, ninja1.13.2. PyTorch CUDA availability is true and device name matches H100. This is environment availability only.
- No kernel launch experiment, build, benchmark, installation or source edit performed.

## Learning and handoff

- Source location is the outstanding prerequisite. User's implementations must be obtained; do not replace them with invented kernels.
- Next: synchronize actual kernel files, read operation contracts/reference checks, run bounded correctness cases and then appropriate benchmarks.
- Progress records local/uncommitted. No commit or push authorized this turn; no rental shutdown action.

## Follow-up — source arrival and H100 execution

- User requested retry after pushing code. Fetched and fast-forwarded to `9882f73ba53f7f074d425b167047ebbcea4e5ef9`. Preserved this preflight history; only assistant-created preliminary progress edits were set aside before the pull.
- Read updated instructions, project record and online-merge review. Kernel bodies, fixtures and tolerances were not modified.
- Built `affine`, `bias_relu`, `row_sum`, `softmax` with `nvcc -std=c++17 -O3 -lineinfo -arch=sm_90`; built `softmax_reference_test.cpp` with `g++ -std=c++17 -O2 -Wall -Wextra -Werror`. All builds exited 0. No fast-math flag.
- Executed CPU reference tests and all four GPU harnesses; all exited 0. Affine: 1,000 exact outputs. Bias/ReLU: 4,810 exact outputs. Row sum: 20 variant/case runs, no mismatched rows. Softmax: 30 variant/case runs, no bad elements or rows under original atol=1e-7, rtol=1e-5, row-sum tolerance=1e-5.
- Largest reported softmax absolute error across these runs: 3.8147563e-8; largest row-sum error: 8.72603323e-8.
- Ran Compute Sanitizer 2025.3.1.0 with `--tool TOOL --error-exitcode 99 EXECUTABLE` for memcheck/racecheck/synccheck/initcheck on each of the four GPU binaries. All 16 runs exited 0 and reported zero errors; racecheck also reported zero warnings/hazards. Each subprocess had a 180-second timeout.
- Exact build/run commands, outputs and exit codes are preserved in `progress/evidence/2026-10-26-project-03-validation/`. Compiler/GPU version logs included. Local executables are ignored under `artifacts/project03-2026-10-26/bin/`; rebuild them on another machine.

## Experiment boundaries and handoff

- Correctness only: deterministic existing FP32 fixtures, row-sum widths 1–1025 and softmax widths 1–4097, 128 threads per reduction block. No timings, performance ranking, profiling capture or end-to-end inference claim.
- Sanitizer success covers exercised paths; it is not a proof for arbitrary shapes/inputs. Guided learner implementations now have assisted runtime validation; independent transfer remains unassessed. Learner conclusion not yet provided.
- Next concrete task: a matched CUDA-event timing harness for the three softmax variants, with warmup and repeated measurements and allocation/transfers outside the timing boundary.
- Updated status, project record and quick notes. Evidence and notes are local/uncommitted and not yet available across machines via Git; no commit/push or rental shutdown performed.

## Follow-up — user requested timings

- User explicitly requested timing existing kernels; added only a standalone benchmark harness, no learner kernel changes. Reuses sources with standalone host-symbol renaming.
- Hypothesis: warp reductions may reduce synchronization cost; online softmax may reduce input traversals. Alternatives include extra arithmetic/dependency cost, warm caches and launch-submission limits. Learner timing prediction not solicited: this was delegated execution.
- Compiled with the same nvcc C++17/O3/lineinfo/sm_90 flags; compilation and bounded (180-second timeout) benchmark process exited 0. No competing compute processes listed before execution.
- Timed all seven kernel variants: initial elementwise shapes, 37×130 reductions, and 4096×1024 reductions. Each benchmark fixture independently validated before timing and after each batch; fixed softmax tolerances retained.
- 50 warmup launches/variant, nine rotating-order rounds, 500 launches per sample. Recorded/synchronized CUDA events around each batch. Direct launches include GPU idle submission gaps; allocations, host/device copies and validation outside timing. Buffers reused, warm-cache conditions; clocks not locked.
- Verified all 108 samples (12 groups × 9 rounds). Raw CSV, summary CSV, environment snapshots and detailed interpretation/reproduction instructions live under `project_03_gpu_operator_library/results/2026-10-26-cuda-timing/`.
- Large-shape median time reductions: warp row sum ~4.9%, warp softmax ~6.4%; online softmax ~35.6% slower than baseline. Small row-sum ranges overlap. Observed differences are not explanations of hardware bottlenecks or cross-session confidence intervals.
- Updated concise measurement notes and progress. Learner conclusion not yet provided; assisted benchmark execution does not establish independent understanding.
- Next: interpret online softmax result and select one controlled follow-up measurement. All changes/results local and uncommitted; no commit/push performed.
