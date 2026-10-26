// Capture one warmed-up launch, preserving the learner's kernel implementations.
#define main softmax_correctness_main
#include "../kernels/cuda/softmax.cu"
#undef main
#include <cuda_profiler_api.h>
#include <string>
int main(int argc, char** argv) {
    if (argc!=2 || (std::string(argv[1])!="baseline" && std::string(argv[1])!="warp" && std::string(argv[1])!="online" && std::string(argv[1])!="online_registers")) {
        fprintf(stderr,"Usage: profile_softmax baseline|warp|online|online_registers\n"); return 2;
    }
    const std::string variant=argv[1];
    constexpr int rows=4096, cols=1024, count=rows*cols;
    std::vector<float> x(count), y(count);
    for(int i=0;i<count;++i) x[i]=float((i%41)-20)*0.25f;
    const auto expected=softmax_test::reference(x,rows,cols);
    auto ck=[](cudaError_t s){check_cuda(s,"profile harness");};
    float *dx,*dy;
    ck(cudaMalloc(&dx,count*sizeof(float))); ck(cudaMalloc(&dy,count*sizeof(float)));
    ck(cudaMemcpy(dx,x.data(),count*sizeof(float),cudaMemcpyHostToDevice));
    auto launch=[&]{
        if(variant=="baseline") softmax<128><<<rows,128>>>(dx,dy,cols);
        else if(variant=="warp") softmax_warp<128><<<rows,128>>>(dx,dy,cols);
        else if(variant=="online") softmax_online<128><<<rows,128>>>(dx,dy,cols);
        else softmax_online_registers<128><<<rows,128>>>(dx,dy);
    };
    for(int i=0;i<50;++i) launch();
    ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
    ck(cudaProfilerStart());
    launch(); ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
    ck(cudaProfilerStop());
    ck(cudaMemcpy(y.data(),dy,count*sizeof(float),cudaMemcpyDeviceToHost));
    const auto result=softmax_test::validate(y,expected,rows,cols);
    printf("%s %s: bad_elements=%d bad_rows=%d\n",result.passed()?"PASS":"FAIL",argv[1],result.bad_elements,result.bad_rows);
    ck(cudaFree(dx)); ck(cudaFree(dy));
    return result.passed()?0:1;
}
