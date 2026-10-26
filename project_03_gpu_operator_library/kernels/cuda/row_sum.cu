#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// X: contiguous row-major FP32 [rows, cols]; Y: separate FP32 [rows].
// Launch contract: exactly 'rows' blocks, exactly 128 threads per block.
// Initial workload: rows=37, cols=130.
template <int N>
__global__ void row_sum(const float* x, float* y, int cols) {
    __shared__ float partials[N];

    const int row = blockIdx.x;
    const int t = threadIdx.x;

    // TODO 1: Accumulate this thread's assigned columns into a private sum.
    float sum = 0.0f;
    for (int i = t; i < cols; i += N) {
        sum += x[row * cols + i];
    }

    // TODO 2: Write that sum into partials[t], then synchronize the block.
    partials[t] = sum;
    __syncthreads();

    // TODO 3: Reduce with offsets 64, 32, ..., 1.
    // Only participating threads add; every thread reaches each barrier.
    for (int d = N / 2; d > 0; d /= 2) {
        if (t < d) {
            partials[t] += partials[t + d];
        }
        __syncthreads();
    }

    // TODO 4: Have thread 0 store the row total in y[row].
    if (t == 0) {
        y[row] = partials[0];
    }
}

// Warp-reduction exercise: same input/output contract as row_sum.
// Launch exactly N threads per block; first variant uses N=128.
// The host harness below validates both baseline and warp versions.
template <int N>
__global__ void row_sum_warp(const float* x, float* y, int cols) {
    static_assert(N >= 32 && N <= 1024 && N % 32 == 0,
                  "Use complete warps and at most 1024 threads per block");
    constexpr int num_warps = N / 32;
    __shared__ float warp_partials[num_warps];

    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;
    if (lane == 0) {
        warp_partials[warp] = 0.0f;
    }

    float sum = 0.0f;
    // TODO 1: Accumulate this thread's assigned columns into sum.
    for (int i = t; i < cols; i += N) {
        sum += x[row * cols + i];
    }
    // TODO 2: Reduce sum within each warp using shuffle-down operations.
    // All 32 lanes participate at offsets 16, 8, 4, 2, 1.
    // Pause here for the first review.
    for (int d = 16; d > 0; d /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, d);
    }

    // TODO 3: Each warp's lane 0 writes its total to warp_partials[warp].
    // Then every thread reaches a block-wide barrier.
    if (lane == 0) {
        warp_partials[warp] = sum;
    }
    __syncthreads();

    // TODO 4: Warp 0 loads the shared partials, padding unused lanes with 0.
    // All 32 lanes of warp 0 perform another shuffle reduction.
    if (warp == 0) {
        if (lane < num_warps) {
            sum = warp_partials[lane];
        }
        else {
            sum = 0.0f;
        }
        for (int d = 16; d > 0; d /= 2) sum += __shfl_down_sync(0xffffffff, sum, d);
        if (lane == 0) y[row] = sum;
    }
}

// Standalone correctness harness; timing/profiling will be added separately.
void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int rows = 37;
    constexpr int threads_per_block = 128;
    // Include the original width 130 plus short, warp/block-boundary and long rows.
    const int widths[] = {1, 17, 31, 32, 33, 127, 128, 130, 257, 1025};
    const char* variants[] = {"baseline", "warp"};
    int failed_runs = 0;
    int total_runs = 0;

    for (const int cols : widths) {
        const size_t input_bytes = rows * cols * sizeof(float);
        const size_t output_bytes = rows * sizeof(float);
        std::vector<float> x(rows * cols);
        std::vector<float> y(rows);
        std::vector<double> expected(rows, 0.0);

        // Bounded integer inputs keep all sums exact in FP32 for these cases.
        // Sequential double reference is independent of either reduction tree.
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                x[row * cols + col] = static_cast<float>(row + (col % 17) - 8);
                expected[row] += static_cast<double>(x[row * cols + col]);
            }
        }

        float* x_gpu = nullptr;
        float* y_gpu = nullptr;
        check_cuda(cudaMalloc(&x_gpu, input_bytes), "allocate x");
        check_cuda(cudaMalloc(&y_gpu, output_bytes), "allocate y");
        check_cuda(cudaMemcpy(x_gpu, x.data(), input_bytes, cudaMemcpyHostToDevice),
                   "upload x");

        for (int variant = 0; variant < 2; ++variant) {
            printf("Running row_sum/%s [%d,%d]\n", variants[variant], rows, cols);
            fflush(stdout);
            // Reset for EACH variant so prior valid output cannot hide missing stores.
            check_cuda(cudaMemset(y_gpu, 0xff, output_bytes), "initialize NaN output sentinel");
            if (variant == 0) {
                row_sum<threads_per_block><<<rows, threads_per_block>>>(x_gpu, y_gpu, cols);
            } else {
                row_sum_warp<threads_per_block><<<rows, threads_per_block>>>(x_gpu, y_gpu, cols);
            }
            check_cuda(cudaGetLastError(), "launch row_sum variant");
            check_cuda(cudaDeviceSynchronize(), "execute row_sum variant");
            check_cuda(cudaMemcpy(y.data(), y_gpu, output_bytes, cudaMemcpyDeviceToHost),
                       "download y");

            int error_count = 0;
            for (int row = 0; row < rows; ++row) {
                // Exact comparison also rejects the NaN sentinel and infinity.
                if (static_cast<double>(y[row]) != expected[row]) {
                    if (error_count < 10) {
                        fprintf(stderr, "row_sum/%s [%d,%d] row %d: got %g, expected %g\n",
                                variants[variant], rows, cols, row, y[row], expected[row]);
                    }
                    ++error_count;
                }
            }
            printf("%s row_sum/%s [%d,%d]: %d mismatched rows\n",
                   error_count == 0 ? "PASS" : "FAIL", variants[variant], rows, cols, error_count);
            ++total_runs;
            if (error_count != 0) ++failed_runs;
        }

        check_cuda(cudaFree(x_gpu), "free x");
        check_cuda(cudaFree(y_gpu), "free y");
    }
    printf("%s: %d variant/case runs, %d failed\n",
           failed_runs == 0 ? "PASS" : "FAIL", total_runs, failed_runs);
    return failed_runs == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
