//
// Created by FdyCN on 2024/2/22.
//


#ifndef CUDA_GEMM_GEMM_HPP
#define CUDA_GEMM_GEMM_HPP

#include "stdlib.h"

enum class GEMM_OP {
    FLOAT_NAIVE_GEMM_N_T = 0, // naive float GEMM, trans_a = false, trans_b = true
    FLOAT_SRAM_GEMM_N_N = 1,  // shared memory float GEMM, trans_a = false, trans_b = false

    HALF_NAIVE_TENSORCORE_N_T = 10, // naive gemm with tensor core for fp16
    HALF_SRAM_TENSORCORE_N_T = 11,

    // cublas related
    FLOAT_CUBLAS_GEMM_N_N = 20, // cublas float GEMM, trans_a = false, trans_b = false
    FLOAT_CUBLAS_GEMM_T_N = 21, // cublas float GEMM, trans_a = true, trans_b = false
    FLOAT_CUBLAS_GEMM_N_T = 22, // cublas float GEMM, trans_a = false, trans_b = true
    FLOAT_CUBLAS_GEMM_T_T = 23, // cublas float GEMM, trans_a = true, trans_b = true

    HALF_CUBLAS_GEMM_N_N = 30, // cublas half GEMM, trans_a = false, trans_b = false
    HALF_CUBLAS_GEMM_T_N = 31, // cublas half GEMM, trans_a = true, trans_b = false
    HALF_CUBLAS_GEMM_N_T = 32, // cublas half GEMM, trans_a = false, trans_b = true
    HALF_CUBLAS_GEMM_T_T = 33, // cublas half GEMM, trans_a = true, trans_b = true
};

int gemm_float(float* a, float* b, float* out, const int M, const int N, const int K, const int iter, GEMM_OP op, float* perf = nullptr);

int gemm_half(void* a, void* b, void* out, const int M, const int N, const int K, const int iter, GEMM_OP op, float* perf = nullptr);

int cublas_gemm_float(void** v_handle, float* a, float* b, float* out, const int M, const int N, const int K, const int iter, GEMM_OP op, float* perf = nullptr);

int cublas_gemm_half(void** v_handle, void* a,  void* b,  void* out, const int M, const int N, const int K, const int iter, GEMM_OP op, float* perf = nullptr);
#endif //CUDA_GEMM_GEMM_HPP
