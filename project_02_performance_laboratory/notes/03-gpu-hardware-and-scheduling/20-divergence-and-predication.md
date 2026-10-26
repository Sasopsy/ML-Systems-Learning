# Branch divergence and predication

## Four metrics to distinguish

| Metric | Meaning | Nsight Compute location |
| --- | --- | --- |
| **Branch Efficiency** | Uniformity of branch targets across active lanes; not percentage of runtime used efficiently | Details → Source Counters |
| **Avg. Active Threads Per Warp** | Average active lanes per executed warp instruction, before checking instruction predicates; maximum 32 | Details → Warp State Statistics |
| **Avg. Not Predicated Off Threads Per Warp** | Average lanes that are both active and enabled by the instruction predicate | Details → Warp State Statistics |
| **Divergent branch-target count** | Counts divergent branch targets, including fallthrough; not unique source `if` statements | Raw → filter `smsp__sass_branch_targets_threads_divergent.sum` |

## Active versus predicated-on

```text
@P0 ADD ...
```

- **Active lane:** participates in the current execution path.
- **Predicated-on lane:** active and its instruction predicate (`P0` here) is true, so it performs the addition.
- Predication can disable lanes without taking divergent branch paths.
- Lane averages are weighted by executed instructions, not elapsed time. High branch efficiency alone does not guarantee full lane participation or good performance.

## Find the metrics and their source locations

1. Open the `.ncu-rep` and select the desired kernel.
2. In **Details**, expand **Source Counters** and **Warp State Statistics** for the summary metrics above.
3. In **Raw**, filter for exact metric names. Branch efficiency is `smsp__sass_average_branch_targets_threads_uniform.pct`.
4. Select a metric and open **Metric Details** for its definition, units and value.
5. In **Source → CUDA/SASS**, show **Divergent Branches** to find nonzero counts at particular instructions/source lines. UI selectors vary by version; source correlation requires line information.

- A missing metric is not zero: it may not have been collected.
- **More branch instructions ≠ more divergence:** all lanes can agree on a branch and still execute extra control-flow work.
- A high or low branch-uniformity percentage does not directly quantify time lost; check lane participation, instruction work and scheduler behavior too.
- Source attribution identifies the condition associated with divergent targets, not necessarily which dynamic caller or input pattern caused them.

## Example from our online softmax profile

- Online: **99.01%** branch efficiency, **31.38/32** active lanes, **29.91/32** active predicated-on lanes, **8,192** divergent branch targets.
- Nonzero source counts mapped to the `if (b.s == 0)` merge guard. Divergence existed, but aggregate lane participation remained high; this did not establish divergence as the dominant bottleneck.
- Full experiment evidence stays in [Project 3's report](../../../project_03_gpu_operator_library/results/2026-10-27-softmax-ncu/README.md).

Sources: [NVIDIA source metrics](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#source-page), [Raw page](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#raw-page).
