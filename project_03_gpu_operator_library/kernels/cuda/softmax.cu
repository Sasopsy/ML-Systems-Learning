#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "softmax_test_support.h"

// Summary of a group of finite logits: maximum and sum shifted by that maximum.
struct OnlineState {
    float m;
    float s;
};

// Merge finite-logit summaries, treating the empty summary as an identity.
__device__ __forceinline__
OnlineState merge_online(OnlineState a, OnlineState b) {
    // TODO M1: Handle empty groups (s == 0) before evaluating exponentials.
    // Returning from this helper is fine; callers must still reach all shuffles
    // and block barriers. Two empty groups should produce an empty group.
    if (b.s == 0) {
        return a;
    }
    if (a.s == 0) {
        return b;
    }

    // TODO M2: Find the combined maximum and rescale both sums to that maximum.
    // Keep the old a.m and b.m available when calculating the rescaling factors.
    OnlineState result{-INFINITY, 0.0f};
    // Set result.m and result.s here.
    result.m = fmaxf(a.m, b.m);
    result.s = a.s * expf(a.m - result.m) + b.s * expf(b.m - result.m);
    return result;
}

// X/Y: separate contiguous row-major FP32 [rows, cols], finite inputs, cols > 0.
// Launch exactly 'rows' blocks and N threads per block; N must be a power of two.
// First workload: rows=37, cols=130, N=128; host harness also checks edge cases.
// Baseline: recompute exponentials for output rather than retain a row buffer.
template <int N>
__global__ void softmax(const float* x, float* y, int cols) {
    __shared__ float partials[N];

    const int row = blockIdx.x;
    const int t = threadIdx.x;

    // TODO 1: Find this thread's private maximum over its assigned columns.
    // Start at -INFINITY; use fmaxf for FP32 comparisons.
    float max = -INFINITY;
    for (int i = t; i < cols; i+=N) {
        max = fmaxf(max, x[row * cols + i]);
    }
    partials[t] = max;
    __syncthreads();

    // TODO 2: Store the private maximum in partials[t], then synchronize.
    // Reduce the shared maxima using halving offsets and unconditional barriers.
    for (int d = N / 2; d > 0; d /= 2) {
        if (t < d) {
            partials[t] = fmaxf(partials[t], partials[t + d]);
        }
        __syncthreads();
    }

    // TODO 3: Copy the row maximum into a private variable.
    // Synchronize before any thread overwrites partials for the next phase.
    float row_max = partials[0];
    __syncthreads();

    // TODO 4: Sum expf(x[row * cols + col] - row_max) over assigned columns.
    float sum = 0.0f;
    for (int i = t; i < cols; i+=N) {
        sum += expf(x[row * cols + i] - row_max);
    }
    partials[t] = sum;
    __syncthreads();
    // TODO 5: Store the private sum in partials[t], then synchronize.
    // Reduce the shared sums using halving offsets and unconditional barriers.
    for (int d = N / 2; d > 0; d /= 2) {
        if (t < d) {
            partials[t] += partials[t + d];
        }
        __syncthreads();
    }

    // TODO 6: Read the row denominator, then revisit assigned columns.
    // Recompute each shifted exponential, normalize it, and write y.
    float row_sum = partials[0];
    for (int i = t; i < cols; i+=N) {
        y[row * cols + i] = expf(x[row * cols + i] - row_max) / row_sum;
    }
}

// Warp-reduction exercise: same finite FP32 input/output contract as softmax.
// Launch exactly 'rows' blocks and N threads per block; first variant N=128.
// The host harness below validates both baseline and warp versions.
// Keep exponential recomputation for output to isolate the reduction change.
template <int N>
__global__ void softmax_warp(const float* x, float* y, int cols) {
    static_assert(N >= 32 && N <= 1024 && N % 32 == 0,
                  "Use complete warps and at most 1024 threads per block");
    constexpr int num_warps = N / 32;
    __shared__ float warp_partials[num_warps];

    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;

    float max_value = -INFINITY;
    // TODO 1: Find this thread's private maximum over its strided columns.
    for (int i = t; i < cols; i += N) {
        max_value = fmaxf(max_value, x[row * cols + i]);
    }

    // TODO 2: Reduce maxima within each warp using fmaxf and shuffle-down.
    // All 32 lanes participate at offsets 16, 8, 4, 2, 1.
    // Pause here for the first review.
    for (int d = 16; d > 0; d /= 2) {
        max_value = fmaxf(max_value, __shfl_down_sync(0xffffffff, max_value, d));
    }

    // TODO 3: Each warp's lane 0 stores its maximum in warp_partials[warp].
    // All threads reach a block barrier before warp 0 reads these partials.
    if (lane == 0) {
        warp_partials[warp] = max_value;
    }
    __syncthreads();

    // TODO 4: Warp 0 loads the partial maxima, padding unused lanes with -INFINITY.
    // Reduce within warp 0; its lane 0 publishes the row maximum in warp_partials[0].
    if (warp == 0) {
        if (lane < num_warps) {
            max_value = warp_partials[lane];
        }
        else {
            max_value = -INFINITY;
        }
        for (int d = 16; d > 0; d /= 2) max_value = fmaxf(max_value, __shfl_down_sync(0xffffffff, max_value, d));
        if (lane == 0) warp_partials[0] = max_value;
    }
    __syncthreads();
    // TODO 5: Make that maximum visible to every thread with a block barrier.
    // Each thread copies it into a private row_max variable.
    // Protect those reads with another block barrier before shared-buffer reuse.
    // Both barriers must be reached by every thread, not just warp 0.

    float sum = 0.0f;
    float row_max = warp_partials[0];
    __syncthreads();
    // TODO 6: Accumulate shifted exponentials over this thread's columns.
    // Then reduce the private sums within each warp using shuffle-down.
    for (int i = t; i < cols; i += N) {
        sum += expf(x[row * cols + i] - row_max);
    }
    for (int d = 16; d > 0; d /= 2) sum += __shfl_down_sync(0xffffffff, sum, d);
    if (lane == 0) warp_partials[warp] = sum;
    __syncthreads();

    // TODO 7: Each warp's lane 0 stores its sum; synchronize the whole block.
    // Warp 0 loads these sums with zero padding and performs the final reduction.
    // Its lane 0 publishes the denominator in warp_partials[0].
    if (warp == 0) {
        if (lane < num_warps) {
            sum = warp_partials[lane];
        }
        else {
            sum = 0.0f;
        }
        for (int d = 16; d > 0; d /= 2) sum += __shfl_down_sync(0xffffffff, sum, d);
        if (lane == 0) warp_partials[0] = sum;
    }
    __syncthreads();
    // TODO 8: Synchronize the whole block before all threads read the denominator.
    // Each thread recomputes and normalizes exponentials for its assigned columns.
    // Shared storage is not overwritten after this phase.
    for (int i = t; i < cols; i += N) {
        y[row * cols + i] = expf(x[row * cols + i] - row_max) / warp_partials[0];
    }

}

// Online-normalizer exercise: same finite FP32 input/output contract.
// Launch exactly 'rows' blocks and N threads/block; first variant N=128.
// The host harness validates baseline, warp and online versions on common cases.
template <int N>
__global__ void softmax_online(const float* x, float* y, int cols) {
    static_assert(N >= 32 && N <= 1024 && N % 32 == 0,
                  "Use complete warps and at most 1024 threads per block");
    constexpr int num_warps = N / 32;
    __shared__ OnlineState warp_partials[num_warps];

    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;
    OnlineState state{-INFINITY, 0.0f};

    // TODO 1: Traverse this thread's strided columns once to build its pair.
    // A single finite input v has the summary {v, 1.0f}; use merge_online.
    // Threads with no input keep the empty state and still participate below.
    for (int i = t; i < cols; i += N) {
        state = merge_online(state, OnlineState{x[row * cols + i], 1.0f});
    }

    // TODO 2: Reduce pairs within each warp at offsets 16, 8, 4, 2, 1.
    // Every lane executes BOTH full-mask shuffles (one for m and one for s).
    // Fetch both partner fields before changing state.
    // After the shuffles, merge only if lane + offset < 32.
    // Pause here for review after implementing the helper and TODOs 1-2.
    for (int d = 16; d > 0; d /= 2) {
        state = merge_online(state, OnlineState{__shfl_down_sync(0xffffffff, state.m, d), __shfl_down_sync(0xffffffff, state.s, d)});
    }
    if (lane == 0) {
        warp_partials[warp] = state;
    }
    __syncthreads();

    // TODO 3: Each warp's lane 0 stores its pair in warp_partials[warp].
    // Synchronize the whole block before warp 0 loads these pairs.
    // TODO 4: Warp 0 loads one pair per lane, padding with empty pairs.
    // Reduce those pairs with the same two-shuffle/merge pattern.
    // Warp 0's lane 0 publishes the final row pair in warp_partials[0].
    if (warp == 0) {
        if (lane < num_warps) {
            state = warp_partials[lane];
        }
        else {
            state = OnlineState{-INFINITY, 0.0f};
        }
        for (int d = 16; d > 0; d /= 2) {
            state = merge_online(state, OnlineState{__shfl_down_sync(0xffffffff, state.m, d), __shfl_down_sync(0xffffffff, state.s, d)});
        }
        if (lane == 0) warp_partials[0] = state;
    }
    __syncthreads();
    // TODO 6: Traverse assigned columns again and write normalized outputs.
    // Use the final row maximum and sum, not this thread's private summary.
    state = warp_partials[0];
    for (int i = t; i < cols; i += N) {
        y[row * cols + i] = expf(x[row * cols + i] - state.m) / state.s;
    }
}

// Correctness only: allocation, reference work, and checks are not benchmarks.
void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

int main() {
    constexpr int threads_per_block = 128;
    int failed_runs = 0;
    int total_runs = 0;
    const char* variants[] = {"baseline", "warp", "online"};
    constexpr int num_variants = sizeof(variants) / sizeof(variants[0]);
    const auto tests = softmax_test::cases();
    printf("FP32 criteria: atol=%g, rtol=%g, row_sum_atol=%g\n",
           softmax_test::atol, softmax_test::rtol, softmax_test::row_sum_atol);

    for (const auto& test : tests) {
        const auto x = softmax_test::make_input(test);
        const auto expected = softmax_test::reference(x, test.rows, test.cols);
        std::vector<float> y(x.size());
        const size_t bytes = x.size() * sizeof(float);
        float* x_gpu = nullptr;
        float* y_gpu = nullptr;
        check_cuda(cudaMalloc(&x_gpu, bytes), "allocate x");
        check_cuda(cudaMalloc(&y_gpu, bytes), "allocate y");
        check_cuda(cudaMemcpy(x_gpu, x.data(), bytes, cudaMemcpyHostToDevice), "upload x");
        for (int variant = 0; variant < num_variants; ++variant) {
            printf("Running softmax/%s %s [%d,%d]\n",
                   variants[variant], test.name, test.rows, test.cols);
            fflush(stdout);
            // Reset for EACH variant so prior valid output cannot hide missing stores.
            check_cuda(cudaMemset(y_gpu, 0xff, bytes), "initialize NaN output sentinel");
            if (variant == 0) {
                softmax<threads_per_block><<<test.rows, threads_per_block>>>(x_gpu, y_gpu, test.cols);
            } else if (variant == 1) {
                softmax_warp<threads_per_block><<<test.rows, threads_per_block>>>(x_gpu, y_gpu, test.cols);
            } else {
                softmax_online<threads_per_block><<<test.rows, threads_per_block>>>(x_gpu, y_gpu, test.cols);
            }
            check_cuda(cudaGetLastError(), "launch softmax variant");
            check_cuda(cudaDeviceSynchronize(), "execute softmax variant");
            check_cuda(cudaMemcpy(y.data(), y_gpu, bytes, cudaMemcpyDeviceToHost), "download y");

            const auto result = softmax_test::validate(y, expected, test.rows, test.cols);
            printf("%s softmax/%s %s: bad_elements=%d, bad_rows=%d, max_abs_error=%.9g, row_sum_error=%.9g\n",
                   result.passed() ? "PASS" : "FAIL", variants[variant], test.name,
                   result.bad_elements, result.bad_rows, result.max_abs_error, result.max_row_sum_error);
            ++total_runs;
            if (!result.passed()) ++failed_runs;
        }
        check_cuda(cudaFree(x_gpu), "free x");
        check_cuda(cudaFree(y_gpu), "free y");
    }
    printf("%s: %zu cases, %d variant/case runs, %d failed\n",
           failed_runs == 0 ? "PASS" : "FAIL", tests.size(), total_runs, failed_runs);
    return failed_runs == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
