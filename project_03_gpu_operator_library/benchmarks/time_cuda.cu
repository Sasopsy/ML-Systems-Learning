// Reuse learner kernels verbatim; rename standalone harness symbols to avoid collisions.
#define main affine_correctness_main
#define check_cuda affine_check
#include "../kernels/cuda/affine.cu"
#undef main
#undef check_cuda
#define main bias_correctness_main
#define check_cuda bias_check
#include "../kernels/cuda/bias_relu.cu"
#undef main
#undef check_cuda
#define main rows_correctness_main
#define check_cuda rows_check
#include "../kernels/cuda/row_sum.cu"
#undef main
#undef check_cuda
#define main softmax_correctness_main
#define check_cuda softmax_check
#include "../kernels/cuda/softmax.cu"
#undef main
#undef check_cuda
#include <functional>
#include <string>

static void ck(cudaError_t status) { affine_check(status, "benchmark CUDA operation"); }

// Direct launches on stream 0. Event intervals include any GPU idle launch gaps.
static void bench(const char* op, int rows, int cols) {
    const int count = rows * cols;
    const bool reduction = std::string(op) == "row_sum";
    const bool sm = std::string(op) == "softmax";
    const int outputs = reduction ? rows : count;
    std::vector<float> x(count), bias(cols), y(outputs);
    for (int i=0; i<count; ++i) x[i] = float((i % 41)-20)*0.25f;
    for (int i=0; i<cols; ++i) bias[i] = float((i%9)-4);
    float *dx, *dy, *db;
    ck(cudaMalloc(&dx, count*sizeof(float)));
    ck(cudaMalloc(&dy, outputs*sizeof(float)));
    ck(cudaMalloc(&db, cols*sizeof(float)));
    ck(cudaMemcpy(dx,x.data(),count*sizeof(float),cudaMemcpyHostToDevice));
    ck(cudaMemcpy(db,bias.data(),cols*sizeof(float),cudaMemcpyHostToDevice));
    std::vector<std::function<void()>> launches;
    std::vector<std::string> names;
    if (sm) {
        launches = {[=]{softmax<128><<<rows,128>>>(dx,dy,cols);},
                    [=]{softmax_warp<128><<<rows,128>>>(dx,dy,cols);},
                    [=]{softmax_online<128><<<rows,128>>>(dx,dy,cols);}};
        names = {"baseline","warp","online"};
        if (cols == 1024) {
            launches.push_back([=]{softmax_online_registers<128><<<rows,128>>>(dx,dy);});
            names.push_back("online_registers");
        }
    } else if (reduction) {
        launches = {[=]{row_sum<128><<<rows,128>>>(dx,dy,cols);},
                    [=]{row_sum_warp<128><<<rows,128>>>(dx,dy,cols);}};
        names = {"baseline","warp"};
    } else {
        if (std::string(op)=="affine")
            launches = {[=]{affine<<<(count+255)/256,256>>>(dx,dy,2.0f,1.0f,count);}};
        else launches = {[=]{bias_relu<<<(count+255)/256,256>>>(dx,db,dy,rows,cols);}};
        names = {"baseline"};
    }
    std::vector<double> expected(outputs);
    if (sm) expected = softmax_test::reference(x,rows,cols);
    else if (reduction) {
        for(int r=0;r<rows;++r) for(int c=0;c<cols;++c) expected[r]+=double(x[r*cols+c]);
    } else for(int i=0;i<count;++i)
        expected[i] = std::string(op)=="affine" ? 2.0*double(x[i])+1.0 : std::max(0.0,double(x[i])+double(bias[i%cols]));
    auto validate = [&]{
        ck(cudaMemcpy(y.data(),dy,outputs*sizeof(float),cudaMemcpyDeviceToHost));
        bool good = true;
        if(sm) good=softmax_test::validate(y,expected,rows,cols).passed();
        else for(int i=0;i<outputs;++i) if(double(y[i])!=expected[i]) good=false;
        if(!good) { fprintf(stderr,"FAIL correctness %s %dx%d\n",op,rows,cols); exit(1); }
    };
    for (auto& launch: launches) {
        ck(cudaMemset(dy,0xff,outputs*sizeof(float)));
        launch(); ck(cudaGetLastError()); ck(cudaDeviceSynchronize()); validate();
        for(int i=0;i<50;++i) launch();
        ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
    }
    cudaEvent_t start,stop;
    ck(cudaEventCreate(&start)); ck(cudaEventCreate(&stop));
    constexpr int rounds=9, repeats=500;
    for(int round=0; round<rounds; ++round) {
        // Rotate variant order across rounds to reduce systematic ordering bias.
        for(size_t k=0;k<launches.size();++k) {
            const size_t v=(k+round)%launches.size();
            ck(cudaEventRecord(start));
            for(int i=0;i<repeats;++i) launches[v]();
            ck(cudaEventRecord(stop)); ck(cudaEventSynchronize(stop)); ck(cudaGetLastError());
            float ms; ck(cudaEventElapsedTime(&ms,start,stop));
            validate();
            printf("%s,%s,%d,%d,%d,%d,%.6f\n",op,names[v].c_str(),rows,cols,round,repeats,ms*1000.0/repeats);
        }
    }
    ck(cudaEventDestroy(start)); ck(cudaEventDestroy(stop));
    ck(cudaFree(dx)); ck(cudaFree(dy)); ck(cudaFree(db));
}
int main() {
    puts("operation,variant,rows,cols,round,repeats,us_per_call");
    bench("affine",1,1000);
    bench("bias_relu",37,130);
    for(const char* op: {"row_sum","softmax"}) {
        bench(op,37,130);
        bench(op,4096,1024);
    }
    return 0;
}
