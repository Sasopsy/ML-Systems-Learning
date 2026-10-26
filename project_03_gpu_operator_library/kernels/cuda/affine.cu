// Elementwise affine transformation: y[i] = a * x[i] + b.
// x and y are separate contiguous FP32 arrays containing N elements.
// First version: one output element per thread, with a 1D grid and blocks.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void affine(
    const float* x,
    float* y,
    float a,
    float b,
    int N
) {
    // TODO: Calculate this thread's element index.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: If the index is valid, calculate and store its output.
    if (i < N) {
        y[i] = a * x[i] + b;
    }
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n",
                operation, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

int main() {
    int N = 1000;
    float* x = new float[N];
    float* y = new float[N];
    float a = 2.0f;
    float b = 1.0f;

    float* x_gpu;
    float* y_gpu;
    check_cuda(cudaMalloc(&x_gpu, N * sizeof(float)), "allocating x_gpu");
    check_cuda(cudaMalloc(&y_gpu, N * sizeof(float)), "allocating y_gpu");

    for (int i = 0; i < N; i++) {
        x[i] = i;
    }

    check_cuda(cudaMemcpy(x_gpu, x, N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");

    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;

    affine<<<num_blocks, threads_per_block>>>(x_gpu, y_gpu, a, b, N);

    check_cuda(cudaGetLastError(), "launch affine");
    check_cuda(cudaDeviceSynchronize(), "execute affine");

    check_cuda(cudaMemcpy(y, y_gpu, N * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
    check_cuda(cudaFree(x_gpu), "cudaFree");
    check_cuda(cudaFree(y_gpu), "cudaFree");

    int error_count = 0;
    for (int i = 0; i < N; i++) {
        if (y[i] != a * x[i] + b) {
            printf("Error at index %d: %f != %f\n", i, y[i], a * x[i] + b);
            error_count++;
        }
    }
    if (error_count > 0) {
        printf("Found %d errors\n", error_count);
    }
    delete[] x;
    delete[] y;

    if (error_count == 0) {
        printf("PASS: all %d outputs match\n", N);
    }
    return error_count == 0 ? 0 : 1;
}