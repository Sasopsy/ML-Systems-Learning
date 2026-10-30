# Project 3 — offline handoff

## Download and start

- **128K dims extension:** [combined timing tables](project_03_gpu_operator_library/results/2026-10-31-softmax-128k/README.md) and [24 new paired profiles](project_03_gpu_operator_library/results/2026-10-31-offline-profiles-128k/README.md) cover added dims 16384/32768/65536/131072. Download this new profile folder alongside the earlier package below to inspect the full range. Both have verified manifests.

- **Refreshed and verified — 2026-10-31:** the final API is `softmax_online_registers<N, dims>`, with complete-warp and divisibility checks. All benchmark results and reports were replaced again from this final version, using N = 128 for comparison. Folder names remain unchanged for existing links.

- **Results are excluded from Git.** Download them separately and restore `project_03_gpu_operator_library/results/` inside your local checkout so the links below resolve. Code, scripts, notes and progress records are included in the Git handoff.

- Download [the self-contained full-sweep profile folder](project_03_gpu_operator_library/results/2026-10-27-offline-profiles/README.md), preserving its subfolders. It includes all36 reports, comparison pages, text/CSV/SASS exports, source snapshots, sweep evidence, validation logs and a checksum manifest.
- For the complete experiment history, download the entire [Project3 results folder](project_03_gpu_operator_library/results/README.md) instead.
- Open matching ours/PyTorch `.ncu-rep` files in a compatible Nsight Compute viewer on the Mac; no GPU execution is needed. Start with4096×1024 (PyTorch wins), then4096×2048 (ours wins); continue with4096×4096 and4096×8192.
- If GUI report compatibility is a problem, read the exported tables/text/CSV/SASS. PyTorch source is bundled separately because its installed binary lacks CUDA line mappings; our source is embedded and supplied separately.
- Capture version: Nsight Compute2025.3.1.0, H100, CUDA13.0.88, PyTorch2.11.0+cu130.

## Current implementation and tests

- Existing online kernel uses specialized one-exponential private updates; new `softmax_online_registers<N, dims>` retains inputs. Same128-thread/row mapping; supported sweep dims128/512/1024/2048/4096/8192/16384/32768/65536/131072, default1024 preserved.
- Configurable N validation: 64 checks across all 32 complete-warp block sizes, each at widths N and 8*N, passed independent references and four sanitizer tools. This is correctness coverage, not a block-size timing sweep.
- Original CUDA operators passed correctness/sanitizer runs. Expanded softmax suite:62 variant/case runs passed; four sanitizer tools passed, with new retained variant tested at width1024.
- Matched sweep:18 shapes ×3 variants; direct/graph timing modes,972 samples; all independent-reference checks passed. Generalized widths did not receive a new sanitizer sweep.
- Full-sweep profile batches:18 shapes ×2 kernels, all36 pre/post independent reference checks passed. Full profiler replay is diagnostic and distinct from the sweep timing statistics.
- Our variant won nine shapes in both modes; PyTorch won seven in both. Our37×128 and256×128 were direct-only wins. Do not assume those indicate faster GPU kernels.

## Continue learning

- Read [current status](progress/STATUS.md), [Project3 record](progress/projects/03-gpu-operator-library.md), [latest session](progress/sessions/2026-10-31-project-03-softmax-128k.md) and [revision-note index](project_03_gpu_operator_library/notes/README.md).
- Next task: compare the4096×1024 and4096×2048 pairs' instruction work, branch efficiency, memory traffic and stalls; identify a testable explanation for the crossover. No additional GPU run is required to study the saved reports.
- This is an assisted implementation/measurement checkpoint, not completion of Project3 or evidence of independent mastery.
- The earlier checkpoint was pushed as `b664fcb`. This refresh is included in the user-authorized Git handoff; earlier session entries preserve its pre-commit state. Results remain ignored by Git; verify your separate download before rental shutdown. No shutdown action was performed.
