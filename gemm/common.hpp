//
// Created by FdyCN on 2024/2/22.
//

#ifndef CUDA_GEMM_COMMON_HPP
#define CUDA_GEMM_COMMON_HPP

#include <random>
#include <iostream>

#define UP_DIV(x, y) (((x) + ((y) - 1)) / (y))
#define UP_ROUND(x, y) ((((x) + ((y) - 1)) / (y)) * (y))

template<typename T>
void standard_gemm_host(const T *a, const T *b, T *out, const int M, const int N, const int K, bool transA = false,
                        bool transB = true) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            auto sum = 0;
            for (int k = 0; k < K; k++) {
                int a_index = transA ? k * M + m : m * K + k;
                int b_index = transB ? n * K + k : k * N + n;
                sum += a[a_index] * b[b_index];
            }
            out[m * N + n] = sum;
        }
    }
}
#endif // CUDA_GEMM_COMMON_HPP
