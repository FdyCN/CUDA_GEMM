//#include <cuda.h>
#include <cuda_runtime.h>
#include "gemm.hpp"
#include "math.h"
#include "stdio.h"

int main(){
    float* a = (float*)malloc(sizeof(float));
    a[0] = 1.0f;
    float* b = (float*)malloc(sizeof(float));
    b[0] = 2.0f;
    float* c = (float*)malloc(sizeof(float));
    float* d = (float*)malloc(sizeof(float));
    d[0] = a[0] + b[0];

    float* dev_a;
    cudaMalloc((void**)&dev_a, sizeof(float));
    cudaMemcpy(dev_a, a, sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
    float* dev_b;
    cudaMalloc((void**)&dev_b, sizeof(float));
    cudaMemcpy(dev_b, b, sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
    float* dev_c;
    cudaMalloc((void**)&dev_c, sizeof(float));


    naive_add<float>((const float*)dev_a,(const float*)dev_b,(float*)dev_c);

    cudaMemcpy(c,dev_c,sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    if(fabs(c[0] - d[0]) < 1e-6){
      printf("naive_add passed!\n");
    }
    else{
      printf("naive_add failed!\n");
    }


    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}