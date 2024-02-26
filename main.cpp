//
// Created by FdyCN on 2024/2/22.
//

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include "gemm.hpp"
#include "test_utils.hpp"
#include "cublas_v2.h"

#define CHECK_CUBLAS(expr) \
  if((expr) != CUBLAS_STATUS_SUCCESS) \
  { \
    std::cout << "cublas function: [ " << #expr << " ] error!" << std::endl; \
    return -1; \
  }

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

int
test_gemm_kernels(const int M, const int N, const int K, GEMM_OP op, float *perf = nullptr, bool perf_only = false) {
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

    int iter = 10;
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    switch (op) {
        case GEMM_OP::FLOAT_NAIVE_GEMM_N_T: {
            CHECK_RETURN(gemm_float(dev_a, dev_b, dev_c, M, N, K, iter, op, perf), "FLOAT_NAIVE_GEMM_N_T");
            if (!perf_only) {
                cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
                standard_gemm_host<float>(a, b, d, M, N, K, false, true);
                CHECK_TEST(compare_results<float>(d, c, M * N), "FLOAT_NAIVE_GEMM_N_T");
            }
            break;
        }
        case GEMM_OP::FLOAT_SRAM_GEMM_N_N: {
            CHECK_RETURN(gemm_float(dev_a, dev_b, dev_c, M, N, K, iter, op, perf), "FLOAT_SRAM_GEMM_N_N");
            if (!perf_only) {
                cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
                standard_gemm_host<float>(a, b, d, M, N, K, false, false);
                CHECK_TEST(compare_results<float>(d, c, M * N), "FLOAT_SRAM_GEMM_N_N");
            }
            break;
        }
        case GEMM_OP::FLOAT_CUBLAS_GEMM_N_N: {
            CHECK_RETURN(cublas_gemm_float((void *) handle, dev_a, dev_b, dev_c, M, N, K, iter, op, perf),
                         "FLOAT_CUBLAS_GEMM_N_N");
            if (!perf_only) {
                cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
                standard_gemm_host<float>(a, b, d, M, N, K, false, false);
                CHECK_TEST(compare_results<float>(d, c, M * N), "FLOAT_CUBLAS_GEMM_N_N");
            }
            break;
        }
        case GEMM_OP::FLOAT_CUBLAS_GEMM_N_T: {
            CHECK_RETURN(cublas_gemm_float((void *) handle, dev_a, dev_b, dev_c, M, N, K, iter, op, perf),
                         "FLOAT_CUBLAS_GEMM_N_N");
            if (!perf_only) {
                cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
                standard_gemm_host<float>(a, b, d, M, N, K, false, true);
                CHECK_TEST(compare_results<float>(d, c, M * N), "FLOAT_CUBLAS_GEMM_N_N");
            }
            break;
        }
        case GEMM_OP::FLOAT_CUBLAS_GEMM_T_N: {
            CHECK_RETURN(cublas_gemm_float((void *) handle, dev_a, dev_b, dev_c, M, N, K, iter, op, perf),
                         "FLOAT_CUBLAS_GEMM_T_N");
            if (!perf_only) {
                cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
                standard_gemm_host<float>(a, b, d, M, N, K, true, false);
                CHECK_TEST(compare_results<float>(d, c, M * N), "FLOAT_CUBLAS_GEMM_T_N");
            }
            break;
        }
        case GEMM_OP::FLOAT_CUBLAS_GEMM_T_T: {
            CHECK_RETURN(cublas_gemm_float((void *) handle, dev_a, dev_b, dev_c, M, N, K, iter, op, perf),
                         "FLOAT_CUBLAS_GEMM_N_N");
            if (!perf_only) {
                cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
                standard_gemm_host<float>(a, b, d, M, N, K, true, true);
                CHECK_TEST(compare_results<float>(d, c, M * N), "FLOAT_CUBLAS_GEMM_N_N");
            }
            break;
        }
        default: {
            std::cout << "unsupported test op type!" << std::endl;
            break;
        }
    }
    // Destroy the handle
    CHECK_CUBLAS(cublasDestroy(handle));

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);
    free(d);
    return 0;
}

int
test_gemm_kernels_half(const int M, const int N, const int K, GEMM_OP op, float *perf = nullptr,
                       bool perf_only = false) {
    half_float::half *a = (half_float::half *) malloc(M * K * sizeof(half_float::half));
    generate_random_half(a, M * K);
    half_float::half *b = (half_float::half *) malloc(N * K * sizeof(half_float::half));
    generate_random_half(b, N * K);
    half_float::half *c = (half_float::half *) malloc(M * N * sizeof(half_float::half));
    memset(c, 0, M * N * sizeof(half_float::half));
    half_float::half *d = (half_float::half *) malloc(M * N * sizeof(half_float::half));
    memset(d, 0, M * N * sizeof(half_float::half));

    void *dev_a;
    cudaMalloc((void **) &dev_a, M * K * sizeof(half_float::half));
    cudaMemcpy(dev_a, a, M * K * sizeof(half_float::half), cudaMemcpyKind::cudaMemcpyHostToDevice);
    void *dev_b;
    cudaMalloc((void **) &dev_b, N * K * sizeof(half_float::half));
    cudaMemcpy(dev_b, b, N * K * sizeof(half_float::half), cudaMemcpyKind::cudaMemcpyHostToDevice);
    void *dev_c;
    cudaMalloc((void **) &dev_c, M * N * sizeof(half_float::half));

    int iter = 10;
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    switch (op) {
        case GEMM_OP::HALF_NAIVE_TENSORCORE_N_T: {
            CHECK_RETURN(gemm_half((void *) dev_a, (void *) dev_b, (void *) dev_c, M, N, K, iter, op, perf),
                         "HALF_NAIVE_TENSORCORE_N_T");
            if (!perf_only) {
                cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
                standard_gemm_host_half(a, b, d, M, N, K, false, true);
                CHECK_TEST(compare_results_half(d, c, M * N), "HALF_NAIVE_TENSORCORE_N_T");
            }
            break;
        }
        case GEMM_OP::HALF_CUBLAS_GEMM_N_N: {
            CHECK_RETURN(
                    cublas_gemm_half((void *) handle, (void *) dev_a, (void *) dev_b, (void *) dev_c, M, N, K, iter, op,
                                     perf),
                    "HALF_CUBLAS_GEMM_N_N");
            if (!perf_only) {
                cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
                standard_gemm_host_half(a, b, d, M, N, K, false, false);
                CHECK_TEST(compare_results_half(d, c, M * N), "HALF_CUBLAS_GEMM_N_N");
            }
            break;
        }
        case GEMM_OP::HALF_CUBLAS_GEMM_N_T: {
            CHECK_RETURN(
                    cublas_gemm_half((void *) handle, (void *) dev_a, (void *) dev_b, (void *) dev_c, M, N, K, iter, op,
                                     perf),
                    "HALF_CUBLAS_GEMM_N_T");
            if (!perf_only) {
                cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
                standard_gemm_host_half(a, b, d, M, N, K, false, true);
                CHECK_TEST(compare_results_half(d, c, M * N), "HALF_CUBLAS_GEMM_N_T");
            }
            break;
        }
        case GEMM_OP::HALF_CUBLAS_GEMM_T_N: {
            CHECK_RETURN(
                    cublas_gemm_half((void *) handle, (void *) dev_a, (void *) dev_b, (void *) dev_c, M, N, K, iter, op,
                                     perf),
                    "HALF_CUBLAS_GEMM_T_N");
            if (!perf_only) {
                cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
                standard_gemm_host_half(a, b, d, M, N, K, true, false);
                CHECK_TEST(compare_results_half(d, c, M * N), "HALF_CUBLAS_GEMM_T_N");
            }
            break;
        }
        case GEMM_OP::HALF_CUBLAS_GEMM_T_T: {
            CHECK_RETURN(
                    cublas_gemm_half((void *) handle, (void *) dev_a, (void *) dev_b, (void *) dev_c, M, N, K, iter, op,
                                     perf),
                    "HALF_CUBLAS_GEMM_T_T");
            if (!perf_only) {
                cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
                standard_gemm_host_half(a, b, d, M, N, K, true, true);
                CHECK_TEST(compare_results_half(d, c, M * N), "HALF_CUBLAS_GEMM_T_T");
            }
            break;
        }
        default: {
            std::cout << "unsupported test op type!" << std::endl;
            break;
        }
    }
    // Destroy the handle
    CHECK_CUBLAS(cublasDestroy(handle));

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);
    free(d);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 5 && argc != 1) {
        printf("usage: ./main [M] [N] [N] [Performance only] or ./main (means batch tests performace) \n");
        exit(0);
    }
    get_gpu_properties();
    if (argc == 5) {
        size_t M = atoi(argv[1]);
        size_t K = atoi(argv[2]);
        size_t N = atoi(argv[3]);
        bool performance_only = atoi(argv[4]) != 0 ? true : false;
        test_gemm_kernels(M, N, K, GEMM_OP::FLOAT_SRAM_GEMM_N_N, nullptr, performance_only);
        test_gemm_kernels(M, N, K, GEMM_OP::FLOAT_CUBLAS_GEMM_N_N, nullptr, performance_only);
        test_gemm_kernels(M, N, K, GEMM_OP::FLOAT_CUBLAS_GEMM_T_N, nullptr, performance_only);
        test_gemm_kernels(M, N, K, GEMM_OP::FLOAT_CUBLAS_GEMM_N_T, nullptr, performance_only);
        test_gemm_kernels(M, N, K, GEMM_OP::FLOAT_CUBLAS_GEMM_T_T, nullptr, performance_only);
    }

    if (argc == 1) {
        std::vector<int> MNK(16, 256);
        int index = 1;
        for(auto& v: MNK){
            v *= index;
            index++;
        }
        std::string fname = "gemm_batch_test.csv";
        std::ofstream out_file(fname, std::ios::out);
        if (out_file.is_open()) {
            // titles
            out_file << "name" << ','
                     << "M" << ','
                     << "N" << ','
                     << "K" << ','
                     << "Perf(GFLOPS)" << std::endl;
            for (const auto &v: MNK) {
                float perf = 0.0f; // returned in GFLOPS
                test_gemm_kernels(v, v, v, GEMM_OP::FLOAT_SRAM_GEMM_N_N, &perf, true);
                out_file << "FLOAT_SRAM_GEMM_N_N" << ',' << v << ',' << v << ',' << v << ',' << perf << std::endl;
                test_gemm_kernels(v, v, v, GEMM_OP::FLOAT_CUBLAS_GEMM_N_N, &perf, true);
                out_file << "FLOAT_CUBLAS_GEMM_N_N" << ',' << v << ',' << v << ',' << v << ',' << perf << std::endl;
                test_gemm_kernels(v, v, v, GEMM_OP::FLOAT_CUBLAS_GEMM_N_T, &perf, true);
                out_file << "FLOAT_CUBLAS_GEMM_N_T" << ',' << v << ',' << v << ',' << v << ',' << perf << std::endl;
                test_gemm_kernels(v, v, v, GEMM_OP::FLOAT_CUBLAS_GEMM_T_N, &perf, true);
                out_file << "FLOAT_CUBLAS_GEMM_T_N" << ',' << v << ',' << v << ',' << v << ',' << perf << std::endl;
                test_gemm_kernels(v, v, v, GEMM_OP::FLOAT_CUBLAS_GEMM_T_T, &perf, true);
                out_file << "FLOAT_CUBLAS_GEMM_T_T" << ',' << v << ',' << v << ',' << v << ',' << perf << std::endl;
            }
            out_file.close();
        } else {
            std::cout << "Cannot save to file: " << fname << std::endl;
            std::cout << "ONLY print in terminal...." << std::endl;
            for (const auto &v: MNK) {
                float perf = 0.0f; // returned in GFLOPS
                test_gemm_kernels(v, v, v, GEMM_OP::FLOAT_SRAM_GEMM_N_N, &perf, true);
                test_gemm_kernels(v, v, v, GEMM_OP::FLOAT_CUBLAS_GEMM_N_N, &perf, true);
                test_gemm_kernels(v, v, v, GEMM_OP::FLOAT_CUBLAS_GEMM_T_N, &perf, true);
                test_gemm_kernels(v, v, v, GEMM_OP::FLOAT_CUBLAS_GEMM_N_T, &perf, true);
                test_gemm_kernels(v, v, v, GEMM_OP::FLOAT_CUBLAS_GEMM_T_T, &perf, true);
            }
        }

    }

    return 0;
}