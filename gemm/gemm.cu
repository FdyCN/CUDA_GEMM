#include "stdlib.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "gemm.hpp"

template<typename T>
__global__ void naive_add_kernel(const T* in0, const T* in1, T* out){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = in0[idx] + in1[idx];
}

template void __global__ naive_add_kernel<float>(const float*,const float*,float*);

template<typename T>
void naive_add(const T* in0, const T* in1, T* out){
    auto kernel = &naive_add_kernel<float>;
    kernel<<<1,1>>>(in0, in1, out);
}
template void naive_add<float>(const float*,const float*,float*);