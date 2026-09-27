# Warmup and timing a block of steps

## Warmup and clock choice

- Early steps can include **one-time initialization and allocation**.
- Run warmup before measuring steady-state execution.
- Warmup alone does **not prove timing stability**; inspect repeated measurements.
- Use **`time.perf_counter()`** for both timestamps: it is monotonic.
- `time.time()` can jump if the system clock is adjusted.
- Both clocks count waiting time; synchronization is still necessary.

## Our measurement procedure

1. Run **10 untimed warmup steps** once.
2. Measure **five blocks of 50 training steps**.
3. Synchronize before starting and before stopping each block's timer.
4. Store durations; print and extract scalars **after all blocks finish**.

| Inside the timed block | Outside the timed block |
| --- | --- |
| Gradient clearing, forward, loss, backward, optimizer update | Model/optimizer setup and warmup |
| Python loop and CPU submission overhead | Input creation, transfers, and logging |

- Inputs are already on GPU; the same synthetic batch is reused.
- Synchronizing every step would change this execution protocol.

## Formulas and units

Let **`N` = measured steps**, **`T` = block seconds**, **`B` = batch size**.

| Metric | Formula |
| --- | --- |
| Milliseconds per block | `1000 * T` |
| Average milliseconds per step | `1000 * T / N` |
| Images per second | `B * N / T` |

- Apply the **same unit conversion** to summaries as to individual blocks.
- Forgetting `/ N` labels block duration incorrectly as step duration.
- Report **median, minimum, and maximum** of the five block-average step times.
- Mode is not useful when all durations differ: every sample ties for frequency.
- These are **block averages**, not individual-step or tail-latency measurements.

## State and reporting pitfalls

- Warmup updates **weights and BatchNorm statistics**, even though it is untimed.
- State continues across blocks: these are consecutive training blocks, not
  repeated identical-state trials.
- Record warmup and state handling when comparing implementations.
- A loop variable retains its **last assigned value** afterward.
- Saving every duration does not save every loss: print final loss once, or
  explicitly collect per-block results before using per-block labels.
- Synthetic repeated inputs measure computation, not model quality or a real pipeline.

## Difference versus improvement

- Unchanged code can produce different times.
- Example: **4.51 → 4.47 ms** is roughly a **1% measured difference**.
- One pair does **not establish that an optimization caused the difference**.
- Use repeated, controlled baseline/candidate comparisons to assess an effect.
- Five block means do not define a universal noise threshold.
- Small real improvements are not automatically impossible to measure.
- Timing variation alone does **not identify its cause**.
