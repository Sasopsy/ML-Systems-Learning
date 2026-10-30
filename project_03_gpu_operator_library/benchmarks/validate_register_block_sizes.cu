// Exercise every supported complete-warp block size against the FP64 reference.
#define main softmax_correctness_main
#include "../kernels/cuda/softmax.cu"
#undef main

template <int N, int dims>
bool check_shape() {
    constexpr int rows = 3;
    std::vector<float> input(rows * dims), output(rows * dims);
    for (int i = 0; i < rows * dims; ++i)
        input[i] = float((i % 41) - 20) * 0.25f;
    const auto expected = softmax_test::reference(input, rows, dims);
    float *x, *y;
    check_cuda(cudaMalloc(&x, input.size() * sizeof(float)), "allocate input");
    check_cuda(cudaMalloc(&y, output.size() * sizeof(float)), "allocate output");
    check_cuda(cudaMemcpy(x, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice), "copy input");
    check_cuda(cudaMemset(y, 0xff, output.size() * sizeof(float)), "initialize output sentinel");
    softmax_online_registers<N, dims><<<rows, N>>>(x, y);
    check_cuda(cudaGetLastError(), "launch configurable retained softmax");
    check_cuda(cudaDeviceSynchronize(), "finish configurable retained softmax");
    check_cuda(cudaMemcpy(output.data(), y, output.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy output");
    const auto result = softmax_test::validate(output, expected, rows, dims);
    printf("%s N=%d dims=%d: bad_elements=%d bad_rows=%d\n",
           result.passed() ? "PASS" : "FAIL", N, dims, result.bad_elements, result.bad_rows);
    check_cuda(cudaFree(x), "free input");
    check_cuda(cudaFree(y), "free output");
    return result.passed();
}

template <int N>
bool check_block_sizes() {
    bool passed = check_shape<N, N>();
    passed = check_shape<N, N * 8>() && passed;
    if constexpr (N < 1024) passed = check_block_sizes<N + 32>() && passed;
    return passed;
}

int main() {
    const bool passed = check_block_sizes<32>();
    printf("%s: 32 block sizes, 64 shape checks (three rows each)\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
