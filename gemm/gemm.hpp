//
// Created by FdyCN on 2024/2/22.
//


#ifndef CUDA_GEMM_GEMM_HPP
#define CUDA_GEMM_GEMM_HPP

#include "stdlib.h"

template<typename T>
void naive_add(const T* in0, const T* in1, T* out);

template<typename T>
void naive_gemm(const T* a, const T* b, T* out, const int M, const int N, const int K, bool tran_a = false, bool trans_b = true);
#endif //CUDA_GEMM_GEMM_HPP
