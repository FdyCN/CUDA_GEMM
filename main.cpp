//#include <cuda.h>
#include <cuda_runtime.h>
#include "gemm.hpp"
#include "math.h"
#include "stdio.h"
#include "common.hpp"
#include <random>
#include <iostream>

void generate_random_float(float *input, const int length) {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(1.0, 2.0);
    for (int i = 0; i < length; i++) {
//        input[i] = dis(gen);
        input[i] = static_cast<float>(rand() % 2 + 1);
    }
}

template<typename T>
int compare_results(const T *gt, const T *compute, const int length, float err_thresh = 1e-5) {
    if (!gt || !compute) {
        std::cout << "gt or compute ptr is null" << std::endl;
        return -1;
    }
    T max_val = std::numeric_limits<T>::min();
    int err_num = 0;
    for (int i = 0; i < length; i++) {
        if (std::is_same<T, float>::value) {
            if (isnan(static_cast<float>(compute[i]))) {
                std::cout << "compute[" << i << "] is Nan! " << std::endl;
                return -1;
            }
            auto err = fabs(compute[i] - gt[i]);
            if (err > err_thresh) {
                err_num++;
                max_val = err > max_val ? err : max_val;
            }
        } else {
            std::cout << "unsupported data type" << std::endl;
            return -1;
        }
    }

    if (err_num > 0) {
        std::cout << "data compare failed! max err: " << max_val << ", err_num: " << err_num << std::endl;
        return -1;
    }
    return 0;
}

enum class TestOP {
    FLOAT_NAIVE_GEMM_N_T = 0,
};

int test_gemm_kernels(const int M, const int N, const int K, TestOP op) {
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
        case TestOP::FLOAT_NAIVE_GEMM_N_T: {
            naive_gemm<float>(dev_a, dev_b, dev_c, M, N, K, false, true);
            cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            standard_gemm_host<float>(a, b, d, M, N, K, false, true);
            CHECK_TEST(compare_results<float>(d, c, M * N));
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
    test_gemm_kernels(M, N, K, TestOP::FLOAT_NAIVE_GEMM_N_T);
    return 0;
}