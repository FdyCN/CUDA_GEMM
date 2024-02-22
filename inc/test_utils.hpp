//
// Created by FdyCN on 2024/2/22.
//

#ifndef CUDA_GEMM_TEST_UTILS_HPP
#define CUDA_GEMM_TEST_UTILS_HPP

#include <random>
#include <iostream>
#include <unordered_map>

#define CHECK_TEST(expr, op) \
{                        \
do{                      \
    auto res = (expr);   \
    if((res)){             \
        std::cout << "func:  [" << op << "]   compare FAILED! " << std::endl; \
        return -1;       \
    }                    \
    else{                \
        std::cout << "func:  [" << op << "]   compare PASSED! " << std::endl; \
        return 0;\
    }\
    }while(0);                  \
}

#define CHECK_RETURN(x, log) \
{                        \
do{                      \
    if((x)){             \
        std::cout << "func:  [" << log << "]  error! " << std::endl; \
        return -1;       \
    }                    \
    }while(0);                  \
}

void generate_random_float(float *input, const int length) {
    for (int i = 0; i < length; i++) {
        input[i] = static_cast<float>(rand() % 5 + 1);
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

int query_cudacore_per_sm(int major, int minor) {
    static std::unordered_map<int, int> query_map = {
            {60,  64},
            {61, 128},
            {62, 128},
            {70, 64},
            {72, 64},
            {75, 64},
            {80, 64},
            {86, 128},
    };

    int sm = major * 10 + minor;
    auto res= query_map.find(sm);
    if(res == query_map.end()){
        std::cout << "Unsupported sm arch in the query map!" << std::endl;
        return 0;
    }
    return query_map[sm];
}

// calculate capacity reference: https:
// 1. docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix
// 2. https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#gpu-arch
bool support_fp16(int major, int minor) {
    int sm = major * 10 + minor;
    return sm >= 60;
}

bool support_int8(int major, int minor) {
    int sm = major * 10 + minor;
    return sm >= 61;
}

bool support_tensorcore_v1(int major, int minor) {
    int sm = major * 10 + minor;
    return sm >= 70 && sm < 75;
}

bool support_tensorcore_v2(int major, int minor) {
    int sm = major * 10 + minor;
    return sm >= 75 && sm < 80;
}

bool support_tensorcore_v3(int major, int minor) {
    int sm = major * 10 + minor;
    return sm >= 80 && sm < 89;
}

// H100 is sm_90 but I don not have permission to access related docs....
bool support_tensorcore_v4(int major, int minor) {
    int sm = major * 10 + minor;
    return sm >= 89 && sm < 90;
}

#endif //CUDA_GEMM_TEST_UTILS_HPP
