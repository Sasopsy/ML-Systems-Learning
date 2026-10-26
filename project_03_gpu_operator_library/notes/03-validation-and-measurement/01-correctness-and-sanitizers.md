# Correctness before timing

- **Reference checks:** compare outputs against an independent CPU calculation; softmax also checks finite probabilities and row sums near 1.
- **memcheck:** detects invalid or misaligned device-memory accesses.
- **racecheck:** detects shared-memory access hazards.
- **synccheck:** detects invalid synchronization usage.
- **initcheck:** checks uninitialized device global-memory reads with the default configuration used here.
- **Passing means:** no failures detected for the exercised inputs and paths. It does not prove correctness for every input or measure speed.
- **CUDA-event batches:** record start, launch the kernel repeatedly, record stop, synchronize stop, then divide elapsed time by the number of launches.
- **Timing boundary:** leave allocation, copies, warmup and validation outside; direct-launch GPU idle gaps still count inside the event interval.
- **Compare fairly:** same inputs and shape, repeated rounds, rotate variant order. Median/min/max of batch averages are not individual-call latency percentiles.
- **Warm buffers:** repeated access may reuse cache; logical input traversals are not the same as DRAM traffic.
- **Optimization pitfall:** fewer traversals can still lose to extra arithmetic or dependencies. A timing result alone does not identify the cause.
