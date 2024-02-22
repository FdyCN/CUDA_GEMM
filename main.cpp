//
// Created by FdyCN on 2024/2/22.
//

#include <cuda.h>
#include <cuda_runtime.h>
#include "gemm.hpp"
#include "common.hpp"
#include "test_utils.hpp"

#define CHECK_CUDA(expr) \
  if((expr) != cudaSuccess) \
  { \
    std::cout << "cuda function: [ " << #expr << " ] error!" << std::endl; \
    return -1; \
  }


int get_gpu_properties() {
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        printf("=================GPU [%d]=================\n", i);
        CHECK_CUDA(cudaGetDeviceProperties(&prop, i));
        printf("GPU Name = %s\n", prop.name);
        printf("Compute Capability = %d.%d\n", prop.major, prop.minor);
        printf("GPU SMs = %d\n", prop.multiProcessorCount);
        printf("GPU CUDA cores = %d\n", query_cudacore_per_sm(prop.major, prop.minor) * prop.multiProcessorCount);
        printf("GPU Mem clock rate = %.3f GHz\n", prop.memoryClockRate / 1e6);
        printf("[In Boost]GPU SM clock rate = %.3f GHz\n", prop.clockRate / 1e6);
        printf("[In Boost]FP32 Peak Performance = %.3f GFLOPS\n",
               query_cudacore_per_sm(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2);
        int total_cudacore = query_cudacore_per_sm(prop.major, prop.minor) * prop.multiProcessorCount;
        if (support_fp16(prop.major, prop.minor)) {
            printf("[In Boost]FP16 Peak Performance = %.3f GFLOPS\n",
                   total_cudacore * (prop.clockRate / 1e6) * 2 * 2);
        }
        if (support_int8(prop.major, prop.minor)) {
            printf("[In Boost]INT8 Peak Performance = %.3f GFLOPS\n",
                   total_cudacore * (prop.clockRate / 1e6) * 2 * 4);
        }
        if (support_tensorcore_v1(prop.major, prop.minor)) {
            printf("[In Boost]Tensor Core V1 FP16 Peak Performance = %.3f GFLOPS\n",
                   total_cudacore * (prop.clockRate / 1e6) * 2 * 8);
        }
        if (support_tensorcore_v2(prop.major, prop.minor)) {
            printf("[In Boost]Tensor Core V2 FP16 Peak Performance = %.3f GFLOPS\n",
                   total_cudacore * (prop.clockRate / 1e6) * 2 * 8);
            printf("[In Boost]Tensor Core V2 INT8 Peak Performance = %.3f GFLOPS\n",
                   total_cudacore * (prop.clockRate / 1e6) * 2 * 16);
        }
        if (support_tensorcore_v3(prop.major, prop.minor)) {
            printf("[In Boost]Tensor Core V3 FP16 Peak Performance = %.3f GFLOPS\n",
                   total_cudacore * (prop.clockRate / 1e6) * 2 * 16);
            printf("[In Boost]Tensor Core V3 INT8 Peak Performance = %.3f GFLOPS\n",
                   total_cudacore * (prop.clockRate / 1e6) * 2 * 32);
        }
        printf("\n");
    }
    return 0;
}

int test_gemm_kernels(const int M, const int N, const int K, GEMM_OP op) {
    float *a = (float *) malloc(M * K * sizeof(float));
    generate_random_float(a, M * K);
    float *b = (float *) malloc(N * K * sizeof(float));
    generate_random_float(b, N * K);
    float *c = (float *) malloc(M * N * sizeof(float));
    memset(c, 0, M * N * sizeof(float));
    float *d = (float *) malloc(M * N * sizeof(float));
    memset(d, 0, M * N * sizeof(float));

    float *dev_a;
    cudaMalloc((void **) &dev_a, M * K * sizeof(float));
    cudaMemcpy(dev_a, a, M * K * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
    float *dev_b;
    cudaMalloc((void **) &dev_b, N * K * sizeof(float));
    cudaMemcpy(dev_b, b, N * K * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
    float *dev_c;
    cudaMalloc((void **) &dev_c, M * N * sizeof(float));

    switch (op) {
        case GEMM_OP::FLOAT_NAIVE_GEMM_N_T: {
            CHECK_RETURN(gemm_interface<float>(dev_a, dev_b, dev_c, M, N, K, op), "FLOAT_NAIVE_GEMM_N_T");
            cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            standard_gemm_host<float>(a, b, d, M, N, K, false, true);
            CHECK_TEST(compare_results<float>(d, c, M * N), "FLOAT_NAIVE_GEMM_N_T");
            break;
        }
        default: {
            std::cout << "unsupported test op type!" << std::endl;
            break;
        }
    }
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);
    free(d);
    return 0;
}

int main() {
    int M = 16;
    int N = 16;
    int K = 16;
    get_gpu_properties();
    test_gemm_kernels(M, N, K, GEMM_OP::FLOAT_NAIVE_GEMM_N_T);
    return 0;
}