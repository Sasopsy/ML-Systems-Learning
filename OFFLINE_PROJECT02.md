# Project 2 — Start here offline

## What is ready

- The GPU-dependent runs for the **isolated forward convolution milestone** are complete. The full Project 2 roadmap is not complete.
- Read reports on your Mac with **Nsight Compute**. Use **Perfetto** for `.trace.json` timelines. Neither requires a rented GPU to view saved data.
- This snapshot includes current source, learning instructions, revision notes, progress records, and ignored Project 2 artifacts. Source, notes and small evidence files are version-controlled; the results folder is downloaded separately. The user authorized a local commit, not a push.
- Results are organized under `project_02_performance_laboratory/results/`. Download the whole project folder for runnable scripts plus notes/results, or download `results/` alone for reports and bundled study context.

## Open these reports first

| Purpose | Report |
| --- | --- |
| Original small search-off report | [Off 007](project_02_performance_laboratory/results/conv_fprop/nsight/basic/search_off/conv_fprop_ncu_off_007.ncu-rep) |
| Original small search-on report | [On 008](project_02_performance_laboratory/results/conv_fprop/nsight/basic/search_on/conv_fprop_ncu_on_008.ncu-rep) |
| Expanded original search-off kernel, tile256x64x8 | [Off 030](project_02_performance_laboratory/results/conv_fprop/nsight/expanded/search_off/tile_256x64x8/run_030/ncu_off_030.ncu-rep) |
| Expanded original search-on kernel, tile128x32x8 | [On 035](project_02_performance_laboratory/results/conv_fprop/nsight/expanded/search_on/tile_128x32x8/run_035/ncu_on_035.ncu-rep) |
| Off repeat reports | [033](project_02_performance_laboratory/results/conv_fprop/nsight/expanded/search_off/tile_256x64x8/run_033/ncu_off_033.ncu-rep), [034](project_02_performance_laboratory/results/conv_fprop/nsight/expanded/search_off/tile_256x64x8/run_034/ncu_off_034.ncu-rep) |
| On repeat reports for tile128x32x8 | [036](project_02_performance_laboratory/results/conv_fprop/nsight/expanded/search_on/tile_128x32x8/run_036/ncu_on_036.ncu-rep), [037](project_02_performance_laboratory/results/conv_fprop/nsight/expanded/search_on/tile_128x32x8/run_037/ncu_on_037.ncu-rep) |
| Alternate search-on kernel, tile32x32x8 | [031](project_02_performance_laboratory/results/conv_fprop/nsight/expanded/search_on/tile_32x32x8/run_031/ncu_on_031.ncu-rep), [032](project_02_performance_laboratory/results/conv_fprop/nsight/expanded/search_on/tile_32x32x8/run_032/ncu_on_032.ncu-rep) |

- **Do not merge different kernel variants into one implementation comparison.** Search on selected different kernels in fresh processes. The cause of selection variability is unproven.
- Expanded captures have 12 sections and 36 replay passes each. They use kernel replay, cache flushing, and unfixed clocks. Profiled duration is not the unprofiled benchmark estimate.
- cuDNN source code may not be available in the Source view; the reports still contain counters and machine-code information collected by Nsight.
- Text reports and CSV counter exports are in [the batch evidence folder](progress/sessions/2026-10-01-project-02-offline-batch/), so study can continue without the GUI too.

## Read in this order

1. **Launch Statistics and Occupancy:** connect grid size, resource allocation, theoretical residency, and achieved residency. Review [note13](project_02_performance_laboratory/notes/13-nsight-introduction.md).
2. **Scheduler Statistics:** distinguish resident, eligible, and issued warps. Compare the same kernel across its three reports before explaining an off/on difference.
3. **Warp State Statistics:** identify measured waiting conditions. A large stall category alone is not proof that removing it would improve elapsed time.
4. **Memory Workload Analysis:** inspect cache traffic, hit rates, bytes, and throughput. Separate issuing a load from completing it; low HBM bandwidth alone does not prove that memory latency is irrelevant.
5. **Compute Workload Analysis and Instruction Statistics:** relate pipeline activity to instruction types. Percentages have different denominators and need not sum to100%.
6. **Workload Distribution:** inspect differences across hardware instances. Distinguish SM coverage from occupancy during active cycles.
7. Use the tile32x32x8 reports as a transfer exercise after explaining the original pair. Do not treat its higher occupancy or any rule's estimated speedup as automatic evidence of a better implementation.

## Measurements and limits

- Six fresh unprofiled processes, order off/on/on/off/off/on; each uses10 warmup calls and5x100 timed calls, with synchronization at block boundaries.
- Median of process medians: off **67.76275us**, on **27.65771us**, **59.1845% reduction** for this synthetic convolution call. This is not the ResNet training-step reduction.
- Timing processes did not record kernel identities. Capture identities do not prove which kernel ran in each separate unprofiled process.
- Original kernels reproduced in three expanded captures each; two additional on captures selected a different implementation.
- Numerical criterion remains unresolved: atol1e-5/rtol1.3e-6, **4736/524288 failures**, maximum absolute error about **0.0001594768301**. Relative L2 error in the saved benchmark pair is **4.1820152e-7**.
- Saved benchmark snapshots have identical input, weights, and output across the off/on pair. This is evidence for that pair only; it does not establish downstream training accuracy.
- [Machine-readable commands and results](progress/sessions/2026-10-01-project-02-offline-batch/summary.json).
- [Handoff and learning status](progress/sessions/2026-10-01-project-02-offline-handoff.md).

## Continue in a new local session

Open the downloaded folder as the workspace and say:

> Continue Project2 in TUTOR mode from OFFLINE_PROJECT02.md and the latest offline handoff. No GPU is available. We have completed the captures; teach me to read Scheduler Statistics in off030 and on035, one concept at a time. Keep short revision notes. Do not infer mastery from the completed runs.

## Transfer before stopping the rental

1. Download `project_02_performance_laboratory/` recursively to your Mac. It includes scripts, notes, and the organized `results/` folder. No archive is required.
2. Open `results/README.md` for report links, copied session context and local verification instructions.
3. In `results/`, run `python3 verify_download.py`, then open an expanded report in Nsight Compute.
4. Stop the rented instance through your provider after confirming the copy. Transfer and shutdown have not been performed by the assistant.

The old tar archive remains an earlier backup; it does not reflect this layout. The folder manifest verifies current payload files. Historical session commands retain their original paths; [the relocation map](progress/sessions/2026-10-01-project-02-result-layout.json) maps those artifacts to current locations.
