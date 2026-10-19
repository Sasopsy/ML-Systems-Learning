# Constants and instruction reissue

## What is a constant here?

- **Constant value:** a value read without modification during a kernel execution. It may change between launches.
- Example: `output[i] = input[i] * scale`, with `scale = 0.5` for one launch and `0.8` for the next.

| Form | Example | Important distinction |
| --- | --- | --- |
| Literal in the program | `input[i] * 0.5` | The compiler may encode it directly in an instruction |
| Kernel argument | Scale, dimensions, pointers | Argument values can use the constant-memory path; a pointer's target data is separate |
| CUDA constant memory | Small read-only lookup table | An explicit CUDA memory space |

- **Constant cache:** keeps recently accessed data from the constant-memory path close to execution hardware.
- **Cache miss:** the requested value is not currently available in that cache and must be fetched.
- Neural-network weights are **not automatically in CUDA constant memory** because a forward pass only reads them.
- A source-level `const` qualifier alone does not place an object in CUDA constant memory.

- **Stall IMC Miss:** waiting for an Immediate Constant Cache miss to complete. This concerns a constant operand; **No Instructions** concerns instruction fetching. See [the comparison in note 16](16-latency-hiding.md#no-instructions-versus-imc-miss).

## Why issued and executed counts can differ

- **Issue is not a guarantee of successful dispatch on that attempt.** An instruction can be reissued before executing.
- NVIDIA documents this sequence for an **immediate constant-cache (IMC) miss**:
  1. An instruction issues and needs a constant that misses in the cache.
  2. The instruction is not dispatched; the missing data is fetched.
  3. The instruction is reissued after the miss completes and can execute.
- **Illustrative count:** two issue attempts, one execution. No required computation was dropped.
- Different issued/executed denominators can produce slightly different warp-cycles-per-instruction ratios, even when both round to `1.53`.

## Pitfalls

- This is a possible mechanism, **not a confirmed explanation** for the small counter difference in our capture.
- Hardware instruction reissue is separate from **Nsight kernel replay**, which reruns a kernel to collect profiling counters.
- Neither cycles-per-issued-instruction nor cycles-per-executed-instruction directly measures an instruction's completion latency.

## References

- [NVIDIA constant-cache behavior: IMC](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#units).
- [Nsight throughput and counters](13-nsight-introduction.md).
- [Dense-layer dependency example](14-compute-throughput-dense-layer-example.md).
