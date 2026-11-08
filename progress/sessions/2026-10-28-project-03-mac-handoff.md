# 2026-10-28 — Project 3 Mac handoff verification

## Context

- Project 3, read-only results review plus authorized progress maintenance on local macOS.
- User requested Git pull and inspection of new results. Starting HEAD 9882f73; tracked working tree clean, separately downloaded results present.

## Work and evidence

- Read learning instructions, current status/project record, prior session, then incoming handoff and latest profiling session.
- Executed `git pull --ff-only origin main`: fast-forward to b664fcb. Results became ignored under incoming `.gitignore`; no result files removed or overwritten.
- Read `OFFLINE_PROJECT03.md`, results indexes, sweep methodology and paired 4096×1024/2048 comparison pages.
- Python verification of every `results/2026-10-27-offline-profiles/manifest.json` entry checked file existence, byte count and SHA256: 227 files / 186223642 bytes / zero errors. Counted 36 Nsight reports. All paths are relative to Project 3.
- Inspected actual correctness/sanitizer logs: 62 runs pass, memcheck/initcheck/synccheck zero errors, racecheck zero hazards. Original retained sanitizer coverage is width1024; no extension inferred to all sweep widths.
- Confirmed graph median crossover in saved tables: 4096×1024 retained12.691µs versus PyTorch9.305µs; 4096×2048 retained25.466µs versus PyTorch28.484µs. These are sweep batch-average medians, not profiler durations or new measurements.

## Learning and handoff

- No learner interpretation assessed yet. Next task: compare instruction work, mapping, memory traffic and stalls across the two paired shapes and propose a testable explanation.
- No CUDA compilation, GPU execution, remote compute contact or rental shutdown. Verified only the self-contained full-sweep package, not the entire historical results tree.
- Code pull complete; this progress update remains local/uncommitted. No new commit/push requested or performed.
