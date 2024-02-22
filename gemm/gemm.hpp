//
// Created by FdyCN on 2024/2/22.
//


#ifndef CUDA_GEMM_GEMM_HPP
#define CUDA_GEMM_GEMM_HPP

#include "stdlib.h"

enum class GEMM_OP {
    FLOAT_NAIVE_GEMM_N_T = 0,
};

template<typename T>
void naive_add(const T* in0, const T* in1, T* out);

template<typename T>
int gemm_interface(const T* a, const T* b, T* out, const int M, const int N, const int K, GEMM_OP op);
#endif //CUDA_GEMM_GEMM_HPP
