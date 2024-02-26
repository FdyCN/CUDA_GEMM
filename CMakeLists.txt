# find CUDAToolkit is supported when cmake >= 3.17
cmake_minimum_required(VERSION 3.17)

project(CUDA_GEMM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    message (STATUS  "CUDAToolkit_FOUND=${CUDAToolkit_FOUND}" )
    message (STATUS  "CUDAToolkit_VERSION=${CUDAToolkit_VERSION}" )

    set(CUDA_SRC "" inc/test_utils.hpp)
    set(CUDA_INC "")
    enable_language(CUDA)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_FORCE_INLINES -Wno-deprecated-gpu-targets")
    if(CMAK_BUILD_TYPE MATCHES Debug)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O0")
    else()
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
    endif ()

    # build cuda AOT kernel into a library
    # reference: https://developer.nvidia.com/blog/building-cuda-applications-cmake/
    add_library(cuda_aot_lib STATIC "gemm/gemm.cu")
    set_property(TARGET cuda_aot_lib
            PROPERTY CUDA_ARCHITECTURES "86;")
    target_include_directories(cuda_aot_lib PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_include_directories(cuda_aot_lib PUBLIC "gemm/")


    # build executable which linked cuda AOT lib
    list(APPEND CUDA_SRC "main.cpp")
    add_executable(cuda_gemm_test ${CUDA_SRC})

    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_include_directories(cuda_gemm_test PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_include_directories(cuda_gemm_test PUBLIC "gemm/")
    target_include_directories(cuda_gemm_test PUBLIC "inc/")
    target_include_directories(cuda_gemm_test PUBLIC "half/half/")
    target_link_libraries(cuda_gemm_test PUBLIC cuda_aot_lib)
    target_link_libraries(cuda_gemm_test PUBLIC CUDA::cudart)
    target_link_libraries(cuda_gemm_test PUBLIC CUDA::nvrtc)
    target_link_libraries(cuda_gemm_test PUBLIC CUDA::cublas)
else ()
    message("CUDA NOT Found....")
endif ()





