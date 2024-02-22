//
// Created by FdyCN on 2024/2/22.
//

#include "stdlib.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "gemm.hpp"
#include "common.hpp"

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
int gemm_interface(const T *a, const T *b, T *out, const int M, const int N, const int K, GEMM_OP op) {


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
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
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "op #" << static_cast<int>(op) << ": [M, N, K] = [" << M << ", " << N << ", " << K << "] "<<" kernel cost is " << time << " ms" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

template int gemm_interface<float>(const float *, const float *, float *, const int, const int, const int, GEMM_OP);