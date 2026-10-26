#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void bias_relu(
    const float* x,
    const float* bias,
    float* y,
    int rows,
    int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        float value = x[idx] + bias[idx % cols];
        y[idx] = value > 0.0f ? value : 0.0f;
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
    const int cols = 130;
    const int count = rows * cols;
    const size_t matrix_bytes = count * sizeof(float);
    const size_t bias_bytes = cols * sizeof(float);

    float* x = new float[count];
    float* bias = new float[cols];
    float* y = new float[count];

    // Small signed integers are exactly representable in FP32.
    // Different columns have different biases; rows also vary.
    for (int col = 0; col < cols; ++col) {
        bias[col] = static_cast<float>((col % 9) - 4);
    }
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            x[row * cols + col] =
                static_cast<float>(((row * 7 + col * 3) % 21) - 10);
        }
    }

    float* x_gpu = nullptr;
    float* bias_gpu = nullptr;
    float* y_gpu = nullptr;
    check_cuda(cudaMalloc(&x_gpu, matrix_bytes), "allocate x");
    check_cuda(cudaMalloc(&bias_gpu, bias_bytes), "allocate bias");
    check_cuda(cudaMalloc(&y_gpu, matrix_bytes), "allocate y");

    check_cuda(cudaMemcpy(x_gpu, x, matrix_bytes, cudaMemcpyHostToDevice),
               "upload x");
    check_cuda(cudaMemcpy(bias_gpu, bias, bias_bytes, cudaMemcpyHostToDevice),
               "upload bias");
    // Fill output bytes with 0xff (FP32 NaNs): missing writes fail comparison.
    check_cuda(cudaMemset(y_gpu, 0xff, matrix_bytes), "initialize output sentinel");

    const int threads_per_block = 256;
    const int num_blocks = (count + threads_per_block - 1) / threads_per_block;
    bias_relu<<<num_blocks, threads_per_block>>>(x_gpu, bias_gpu, y_gpu, rows, cols);
    check_cuda(cudaGetLastError(), "launch bias_relu");
    check_cuda(cudaDeviceSynchronize(), "execute bias_relu");
    check_cuda(cudaMemcpy(y, y_gpu, matrix_bytes, cudaMemcpyDeviceToHost),
               "download y");

    // CPU reference uses row/column indexing instead of the kernel's modulo.
    int error_count = 0;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const int index = row * cols + col;
            const float sum = x[index] + bias[col];
            const float expected = sum > 0.0f ? sum : 0.0f;
            if (y[index] != expected) {
                if (error_count < 10) {
                    fprintf(stderr, "Mismatch at [%d,%d]: got %g, expected %g\n",
                            row, col, y[index], expected);
                }
                ++error_count;
            }
        }
    }

    check_cuda(cudaFree(x_gpu), "free x");
    check_cuda(cudaFree(bias_gpu), "free bias");
    check_cuda(cudaFree(y_gpu), "free y");
    delete[] x;
    delete[] bias;
    delete[] y;

    if (error_count == 0) {
        printf("PASS: all %d outputs match (%d x %d)\n", count, rows, cols);
    } else {
        fprintf(stderr, "FAIL: %d of %d outputs differ\n", error_count, count);
    }
    return error_count == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
