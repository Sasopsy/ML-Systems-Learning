# Ranking backward kernels

## Compare total contribution

- **Total kernel time** = sum of durations for all calls with the same exact
  kernel name; **mean time** = total / call count.
- A single long call and a frequently repeated short call can rank differently.
- **Share** = group total / sum of all selected backward kernel durations.
  This is not a fraction of backward wall time or whole-step latency.

## Verified second-step results

- Captured backward: **184 kernels**, **1598.380 us** summed kernel duration.
- Assigned actual kernel events through correlation to runtime **or driver**
  launch calls, then to the CPU phase containing the launch. For backward,
  checked the containing CPU operator using External id on the launch thread.
- Main-thread backward scope includes the worker's launch/operator intervals in
  this eager workload. This temporal classification is workload-specific, not a
  universal cross-thread attribution algorithm.
- Excluded GPU annotation ranges, memsets, and gaps. All **975** captured kernels
  across three steps assigned to exactly one phase; no unmatched or duplicate ones.

| Exact kernel-name group (abbreviated) | Calls | Total (us) | Mean (us) | Backward kernel share |
| --- | ---: | ---: | ---: | ---: |
| dgrad_engine | 4 | 449.821 | 112.455 | 28.14% |
| dgrad2d_alg1_1 | 1 | 158.079 | 158.079 | 9.89% |
| wgrad_alg1_engine | 5 | 145.407 | 29.081 | 9.10% |
| bn_bw_1C11_singleread | 19 | 115.100 | 6.058 | 7.20% |
| nchwToNhwcKernel (one exact specialization) | 34 | 96.223 | 2.830 | 6.02% |

- Candidate for closer study: **dgrad_engine**, largest exact-name group in
  all three captured steps (452.476 / 449.821 / 451.196 us).
- Step 2 group is also **19.81%** of all step-2 kernel duration (2270.570 us).
  This does not predict an equal end-to-end speedup if optimized.
- Four durations: **142.015, 141.311, 141.631, 24.864 us**. Three use grid
  `[72, 1, 32]`, one `[9, 2, 32]`; all block `[8, 8, 1]`.
- **Same name does not mean same workload.** Do not treat the 112.455 us mean
  as a fixed-shape isolated baseline. Start with the previously inspected
  142.015 us call, then recover tensor shapes and convolution parameters.
- Evidence: [derived report with full names and launch links](../../progress/sessions/2026-09-27-project-02-backward-kernel-summary.json).

## Recovering the workload shapes

- Current phase trace did **not record shapes**: selected CPU operator 1208 has
  only External id, Record function id, and Ev Idx metadata.
- Add **record_shapes=True** to the active torch.profiler.profile context.
- Export a new `resnet18_shapes.trace.json` to preserve earlier captures.
- Inspect the associated **CPU aten::convolution_backward** event for input
  tensor dimensions; GPU grid/block dimensions are not tensor shapes.
- Re-identify the target using the new trace's operator/launch links; IDs and
  timings can change between captures.
- Verify stride, padding, dilation, and groups from recorded operator arguments
  where available or from the matching model layer. Shape collection alone does
  not guarantee every parameter needed for an isolated workload is captured.
- Shape recording adds profiling overhead; use this capture to identify the
  workload, not as an unprofiled performance comparison.
- [PyTorch 2.11 profiler shape recording](https://docs.pytorch.org/docs/2.11/profiler.html)

## Recovered candidate workload

- Learner generated `resnet18_shapes.trace.json`; corresponding second-step
  candidate still has External id **1208**, launch correlation **7174**, and
  now lasts **142.302 us**. Same IDs this time were verified, not assumed.
- Input argument order checked against installed PyTorch 2.11 operator schema:
  **grad_output, input, weight**, then scalar/list arguments.

| Tensor | Shape | Recorded element strides |
| --- | --- | --- |
| grad_output (dY) | `[32, 512, 2, 2]` | `[2048, 4, 2, 1]` |
| input (X) | `[32, 512, 2, 2]` | `[2048, 4, 2, 1]` |
| weight (W) | `[512, 512, 3, 3]` | `[4608, 9, 3, 1]` |

- **dtype:** float32 for all three; contiguous NCHW activations and standard
  `[out_channels, in_channels/groups, kernel_h, kernel_w]` weight layout.
- **Convolution:** stride `[1,1]`, padding `[1,1]`, dilation `[1,1]`, groups `1`,
  transposed `False`, output_padding `[0,0]`.
- output_mask `[True,True,False]`: compute input and weight gradients, not bias
  gradient. Original operation therefore includes both dgrad and wgrad work.
- To study **dgrad alone**, request only the input gradient; then verify which
  kernel/cuDNN path is selected rather than assuming the same implementation.
- Shapes/settings are recovered; tensor values and exact module identity are not.
  A random-tensor reproducer would be a stated synthetic workload.
- [Recorded candidate evidence](../../progress/sessions/2026-09-27-project-02-selected-convolution.json).

## Why tensor counts alone are insufficient

- For the recovered layer: **X = 32 x 512 x 2 x 2 = 65,536 elements**;
  **W = 512 x 512 x 3 x 3 = 2,359,296 elements**. W has **36x** as many.
- Spatial activations are only 2x2, while weights connect every input channel
  to every output channel with a 3x3 filter. A batch dimension alone does not
  establish that activations contain more elements than weights.
- Despite this, the selected dgrad kernel is longer than its paired wgrad kernel;
  output element count alone does not explain either duration.
- **Work per output matters:** each dX element accumulates contributions across
  output channels and relevant filter positions; each dW element accumulates
  across batch items and relevant output positions.
- Fewer outputs with longer sums can require comparable arithmetic to more
  outputs with shorter sums. Convolution padding affects valid contributions.
- **Execution efficiency matters too:** different algorithms organize threads,
  memory reuse, and accumulation differently. Similar arithmetic counts need
  not have similar runtimes.
- Shape capture confirms dgrad **142.302 us** versus wgrad **37.248 us** (~3.82x).
  The exact cause of this difference remains unmeasured. Small 2x2 maps and high
  channel counts motivate studying algorithm efficiency; they are not a proven
  memory/compute bottleneck diagnosis. Isolate workload before controlled tests.

- Learner predicted dgrad because input X has more elements / a batch dimension,
  while weights W are shared across batch items.
- **Input gradients:** compute a gradient for each input element.
- **Weight gradients:** each shared weight accumulates contributions from many
  batch items and spatial positions. Both calculations depend on batch size.
- Small illustrative example: `y_i = w * x_i`, with incoming gradients `g_i`.
  Then `dx_i = w * g_i`, but `dw = sum_i(x_i * g_i)`. A single weight gradient
  still requires work across the batch.
- Tensor element count alone does not establish operation count or execution
  time. Shape, reductions, memory access, and implementation matter.
- Measured name groups containing dgrad total 794.746 us; wgrad-named groups
  total 169.759 us. Name-based families do not classify unnamed GEMM helpers;
  do not claim these totals cover every input/weight-gradient calculation.
- [NVIDIA convolution computation dimensions](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html)
