//
// Created by FdyCN on 2024/2/22.
//


#ifndef CUDA_GEMM_GEMM_HPP
#define CUDA_GEMM_GEMM_HPP

#include "stdlib.h"

enum class GEMM_OP {
    FLOAT_NAIVE_GEMM_N_T = 0, // naive float GEMM, trans_a = false, trans_b = true
    FLOAT_CUBLAS_GEMM_N_T = 1, // cublas float GEMM, trans_a = false, trans_b = true
};

template<typename T>
void naive_add(const T* in0, const T* in1, T* out);

template<typename T>
int gemm_interface(const T* a, const T* b, T* out, const int M, const int N, const int K, const int iter, GEMM_OP op);

template<typename T>
int cublas_gemm_interface(const T* a, const T* b, T* out, const int M, const int N, const int K, const int iter, GEMM_OP op);
#endif //CUDA_GEMM_GEMM_HPP
