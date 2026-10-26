// Minimal CUDA bridge for same-process comparisons with eager PyTorch.
#define main softmax_correctness_main
#include "../kernels/cuda/softmax.cu"
#undef main

extern "C" int launch_online(const float* x, float* y, int rows, int cols, cudaStream_t stream) {
    softmax_online<128><<<rows,128,0,stream>>>(x,y,cols);
    return static_cast<int>(cudaGetLastError());
}
extern "C" int launch_registers(const float* x, float* y, int rows, int cols, cudaStream_t stream) {
    switch(cols) {
#define CASE(C) case C: softmax_online_registers<128,C><<<rows,128,0,stream>>>(x,y); break;
        CASE(128) CASE(512) CASE(1024) CASE(2048) CASE(4096) CASE(8192)
#undef CASE
        default: return static_cast<int>(cudaErrorInvalidValue);
    }
    return static_cast<int>(cudaGetLastError());
}
