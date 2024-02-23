//
// Created by FdyCN on 2024/2/22.
//

#include "stdlib.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "gemm.hpp"
#include "common.hpp"
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

template<typename T>
__global__ void naive_add_kernel(const T *in0, const T *in1, T *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = in0[idx] + in1[idx];
}

template void __global__ naive_add_kernel<float>(const float *, const float *, float *);

// N_T means transpose_A = false, transpose_B = true
template<typename T>
__global__ void naive_gemm_kernel_N_T(const T *a, const T *b, T *out, const int M, const int N, const int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // N-axis
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // M-axis

    if (idx < N && idy < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            float tmp_a = a[idy * K + k];
            float tmp_b = b[idx * K + k];
            sum += tmp_a * tmp_b;
        }
        out[idy * N + idx] = sum;
    }
}

template void __global__
naive_gemm_kernel_N_T<float>(const float *, const float *, float *, const int, const int, const int);

template<typename T>
void naive_add(const T *in0, const T *in1, T *out) {
    auto kernel = &naive_add_kernel<float>;
    kernel<<<1, 1>>>(in0, in1, out);
}

template void naive_add<float>(const float *, const float *, float *);

template<typename T>
int gemm_interface(const T *a, const T *b, T *out, const int M, const int N, const int K, const int iter, GEMM_OP op) {

    // warming up
    switch (op) {
        case GEMM_OP::FLOAT_NAIVE_GEMM_N_T: {
            auto kernel = &naive_gemm_kernel_N_T<float>;
            dim3 block(16, 16);
            dim3 grid(UP_DIV(N, block.x), UP_DIV(M, block.y));
            kernel<<<grid, block>>>(a, b, out, M, N, K);
            break;
        }
        default: {
            std::cout << "Unsupported GEMM type!" << std::endl;
            return -1;
        }
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iter; i++) {
        switch (op) {
            case GEMM_OP::FLOAT_NAIVE_GEMM_N_T: {
                auto kernel = &naive_gemm_kernel_N_T<float>;
                dim3 block(16, 16);
                dim3 grid(UP_DIV(N, block.x), UP_DIV(M, block.y));
                kernel<<<grid, block>>>(a, b, out, M, N, K);
                break;
            }
            default: {
                std::cout << "Unsupported GEMM type!" << std::endl;
                return -1;
            }
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    float msecPerMatrixMul = time / iter;
    double flopsPerMatrixMul = 2.0 * (double) M * (double) N * (double) K;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
            "op #%d: [M, N, K] = [%d, %d, %d] ==> Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
            static_cast<int>(op), M, N, K,
            gigaFlops,
            msecPerMatrixMul,
            flopsPerMatrixMul);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

template int
gemm_interface<float>(const float *, const float *, float *, const int, const int, const int, const int, GEMM_OP);

// modified from: https://github.com/zchee/cuda-sample/blob/master/0_Simple/matrixMulCUBLAS/matrixMulCUBLAS.cpp
template<typename T>
int cublas_gemm_interface(const T *a, const T *b, T *out, const int M, const int N, const int K, const int iter,
                          GEMM_OP op) {
    // CUBLAS version 2.0
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasHandle_t handle;
        cudaEvent_t start, stop;

        CHECK_CUBLAS(cublasCreate(&handle));

        //Perform warmup operation with cublas
        switch (op) {
            case GEMM_OP::FLOAT_CUBLAS_GEMM_N_T: {
                CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, b, K, a, K, &beta, out, N));
                break;
            }
            default: {
                std::cout << "Unsupported GEMM type!" << std::endl;
                return -1;
            }
        }


        // Allocate CUDA events that we'll use for timing
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        // Record the start event
        CHECK_CUDA(cudaEventRecord(start));

        for (int j = 0; j < iter; j++) {
            // note cublas is column primary!
            // need to transpose the order
            // so C^T = B^T @ A^T
            switch (op) {
                case GEMM_OP::FLOAT_CUBLAS_GEMM_N_T: {
                    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, b, K, a, K, &beta, out, N));
                    break;
                }
                default: {
                    std::cout << "Unsupported GEMM type!" << std::endl;
                    return -1;
                }
            }
        }

        // Record the stop event
        CHECK_CUDA(cudaEventRecord(stop));

        // Wait for the stop event to complete
        CHECK_CUDA(cudaEventSynchronize(stop));

        float msecTotal = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&msecTotal, start, stop));

        // Compute and print the performance
        float msecPerMatrixMul = msecTotal / iter;
        double flopsPerMatrixMul = 2.0 * (double) M * (double) N * (double) K;
        double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
        printf(
                "[cublasSgemm]: [M, N, K] = [%d, %d, %d] ==> Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
                M, N, K,
                gigaFlops,
                msecPerMatrixMul,
                flopsPerMatrixMul);

        // Destroy the handle
        CHECK_CUBLAS(cublasDestroy(handle));
    }
    return 0;
}

template int
cublas_gemm_interface<float>(const float *, const float *, float *, const int, const int, const int, const int,
                             GEMM_OP);