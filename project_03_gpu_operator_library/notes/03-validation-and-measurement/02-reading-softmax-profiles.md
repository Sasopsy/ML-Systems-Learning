# Reading softmax profiles

- **Executed warp instructions:** how much instruction work ran; not a FLOP count. More merge arithmetic and control flow can outweigh a saved input pass.
- **Global-load requests ≠ DRAM traffic:** caches may serve requests. Check both before claiming a memory saving.
- **Long scoreboard:** a warp waits for a dependency on an L1TEX operation, including global loads. This is exposed latency, not automatically saturated memory bandwidth.
- **Math-pipe throttle:** waiting for an execution pipeline to accept work. More exponentials alone do not prove that this pipeline is saturated.
- **Stall fractions ≠ wall-time fractions:** another ready warp may execute while one waits. Check scheduler eligibility/issue activity too.
- **Registers and occupancy:** compare actual counts, theoretical limits, achieved occupancy and spills; don't assume a slower kernel uses more registers.
- **Profiler replay:** metric collection can rerun a kernel. Keep diagnostic durations separate from normal benchmark statistics, and record cache/clock settings.
- **Next controlled experiment:** change one mechanism, preserve correctness, predict which counters should improve, then time without the profiler.

Definitions: [NVIDIA Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference).

## Branch divergence versus branch work

- **Divergence:** active lanes take different paths at a branch; not every branch is divergent.
- **Branch efficiency:** uniformity of branch targets, not percentage of kernel time used efficiently.
- **Check lane participation too:** average active threads per warp instruction and average active, predicated-on threads. Predication can disable lanes without a divergent branch.
- **Source attribution:** use divergent-branch metrics in CUDA/SASS view to locate the condition instead of guessing from source code alone.
- More branch instructions can add work even when almost all branches are uniform. A proposed conditional optimization needs its own measurements.

## Comparing occupancy across mappings

- **Theoretical occupancy:** resource-limited resident warps/SM divided by hardware maximum; does not guarantee enough launched work to fill that capacity.
- **Achieved occupancy:** measured average active resident warps relative to the maximum; not eligible-warps count or compute throughput.
- Different rows/block changes the total block count for the same tensor. Compare grid size as well as registers and per-block residency limits.
- A faster kernel can have lower occupancy if it performs less work or communicates less. Occupancy alone cannot rank implementations or explain a speed difference.
