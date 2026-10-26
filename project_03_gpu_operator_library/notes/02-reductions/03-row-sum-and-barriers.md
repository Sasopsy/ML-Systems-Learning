# 03 — Row sums and block synchronization

## Work assignment

- **Operation:** Y[r] = sum of X[r,c] over columns c. Input [37,130] produces output [37].
- **First launch:** one block per row, exactly 128 threads per block.
- Thread t visits columns t, t+128, t+256, ... while the column is less than cols.
- For 130 columns: thread 0 reads 0/128; thread 1 reads 1/129; threads 2–127 read one element each.
- Each thread first accumulates a **private sum**, then writes it to its own shared-memory slot.
- `__shared__ float partials[128];` provides an array shared by threads in that block; each block has its own array.

## Tree reduction

1. Store private sums in shared memory; all threads reach a block barrier.
2. For offsets 64,32,16,8,4,2,1: threads t < offset add slot t+offset into slot t.
3. All threads reach a barrier after each round, outside the conditional addition.
4. Thread 0 writes slot 0 to the output for its row.

- **Example:** [1,2,3,4,5,6,7,8] → [6,8,10,12] → [16,20] → [36], showing only the remaining partial sums.
- The first algorithm assumes a power-of-two block size (128), not a power-of-two row length.
- Threads that stop adding still participate in barriers. Do not put the barrier inside `if (t < offset)` or return early just because a thread has no input element.
- Implementation and GPU validation are pending; this note records the taught algorithm.
