# find CUDAToolkit is supported when cmake >= 3.17
cmake_minimum_required(VERSION 3.17)

project(CUDA_GEMM LANGUAGES CXX CUDA)

find_package(CUDAToolkit)

if(CUDAToolkit_FOUND)
    message ( STATUS  "CUDAToolkit_FOUND=${CUDAToolkit_FOUND}" )
    message ( STATUS  "CUDAToolkit_VERSION=${CUDAToolkit_VERSION}" )
else ()
    message("CUDA NOT Found....")
endif ()



