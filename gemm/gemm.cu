//
// Created by FdyCN on 2024/2/22.
//

#include "mma.h"
#include "gemm.hpp"
#include "cublas_v2.h"
#include "../half/half/half.hpp"

#define UP_DIV(x, y) (((x) + ((y) - 1)) / (y))
#define UP_ROUND(x, y) ((((x) + ((y) - 1)) / (y)) * (y))

#define CHECK_CUBLAS(expr) \
{ auto res = (expr);                         \
  if((res) != CUBLAS_STATUS_SUCCESS) \
  { \
    std::cout << "cublas function: [ " << #expr << " ] error! return: " << static_cast<int>(res) << std::endl; \
    return -1; \
  } \
}

#define CHECK_CUDA(expr) \
{ auto res = (expr);                  \
  if((res) != cudaSuccess) \
  { \
    std::cout << "cuda function: [ " << #expr << " ] error! return: " << static_cast<int>(res) << std::endl; \
    return -1; \
  } \
}

#define ENUM_CHIP_TYPE_CASE(x)   case x: return(#x);

static std::string get_test_name(GEMM_OP op) {
    switch (op) {
        ENUM_CHIP_TYPE_CASE(GEMM_OP::FLOAT_NAIVE_GEMM_N_T);
        ENUM_CHIP_TYPE_CASE(GEMM_OP::FLOAT_SRAM_GEMM_N_N);
        ENUM_CHIP_TYPE_CASE(GEMM_OP::FLOAT_CUBLAS_GEMM_N_N);
        ENUM_CHIP_TYPE_CASE(GEMM_OP::FLOAT_CUBLAS_GEMM_N_T);
        ENUM_CHIP_TYPE_CASE(GEMM_OP::FLOAT_CUBLAS_GEMM_T_N);
        ENUM_CHIP_TYPE_CASE(GEMM_OP::FLOAT_CUBLAS_GEMM_T_T);
        ENUM_CHIP_TYPE_CASE(GEMM_OP::HALF_NAIVE_TENSORCORE_N_T);
        ENUM_CHIP_TYPE_CASE(GEMM_OP::HALF_CUBLAS_GEMM_N_N);
        ENUM_CHIP_TYPE_CASE(GEMM_OP::HALF_CUBLAS_GEMM_N_T);
        ENUM_CHIP_TYPE_CASE(GEMM_OP::HALF_CUBLAS_GEMM_T_N);
        ENUM_CHIP_TYPE_CASE(GEMM_OP::HALF_CUBLAS_GEMM_T_T);
    }
    return "Unknown test name!";
}

#pragma mark float_kernel
// N_T means transpose_A = false, transpose_B = true
__global__ void
naive_sgemm_kernel_N_T(float *__restrict__ a, float *__restrict__ b, float *__restrict__ out, const int M,
                       const int N,
                       const int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // N-axis
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // M-axis

    if (idx < N && idy < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            float tmp_a = a[idy * K + k];
            float tmp_b = b[idx * K + k];
            sum += tmp_a * tmp_b;
        }
        out[idy * N + idx] = sum;
    }
}

// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// reference: https://github.com/Liu-xiandong/How_to_optimize_in_GPU/blob/master/sgemm/sgemm_v1.cu
template<
        const int BLOCK_SIZE_M,
        const int BLOCK_SIZE_N,
        const int BLOCK_SIZE_K,
        const int THREAD_SIZE_Y,  // height of a Tile that each thread calculate
        const int THREAD_SIZE_X   // width of a Tile that each thread calculate
>
__global__ void
blocked_sram_sgemm_kernel_N_N(float *a, float *b, float *out, const int M,
                              const int N,
                              const int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // the threads number in Block of X,Y
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    const int tid = ty * THREAD_X_PER_BLOCK + tx;

    // shared memory, use double buffer for global-->sram async copy.
    __shared__ float sram_a[2][BLOCK_SIZE_K][BLOCK_SIZE_M]; // store matrix A, transposed for outer-product
    __shared__ float sram_b[2][BLOCK_SIZE_K][BLOCK_SIZE_N]; // store matrix B, no need to transpose

    // registry for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    // async copy of sram_a->frag_a for A in outer-product, so need double buffer
    float frag_a[2][THREAD_SIZE_Y] = {0};
    // async copy of sram_b->frag_b for B in outer-product, so need double buffer
    float frag_b[2][THREAD_SIZE_X] = {0};

    // load from global to register by float4, so THREAD_NUM_PER_BLOCK * 4
    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4);
    const int ldg_num_b = BLOCK_SIZE_N * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4);

    // used for temp store when global-->sram, async copy of global-->sram.
    float ldg_a_reg[4 * ldg_num_a];
    float ldg_b_reg[4 * ldg_num_b];

    // threads number needed in load one row of A\B
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    // row index that will NOT be changed in one thread THREAD_NUM_PER_BLOCK < BLOCK_K or BLOCK_N
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    // col index that will be NOT changed in one thread if THREAD_NUM_PER_BLOCK > BLOCK_K or BLOCK_N
    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    // blocked index.
    float *a_start = &a[(BLOCK_SIZE_M * by) * K];
    float *b_start = &b[BLOCK_SIZE_N * bx];

    // transfer first tile from global mem to shared mem
    // load A from global memory to shared memory, use ldg_a_reg for transpose A from [a_tile_row,a_tile_col] to [a_tile_col,a_tile_row]
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(a_start[OFFSET(
                A_TILE_ROW_START + i, // row
                A_TILE_COL, // col
                K)]);
        sram_a[0][A_TILE_COL][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index];
        sram_a[0][A_TILE_COL + 1][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 1];
        sram_a[0][A_TILE_COL + 2][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 2];
        sram_a[0][A_TILE_COL + 3][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 3];
    }

    // load B from global memory to shared memory directly.
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        FETCH_FLOAT4(sram_b[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(b_start[OFFSET(
                B_TILE_ROW_START + i, // row
                B_TILE_COL, // col
                N)]);
    }
    __syncthreads();

    // load first in col of A from sram to register
    for (int idy = 0; idy < THREAD_SIZE_Y; idy += 4) {
        FETCH_FLOAT4(frag_a[0][idy]) = FETCH_FLOAT4(sram_a[0][0][THREAD_SIZE_Y * ty + idy]);
    }
    // load first in row of B from sram to register
    for (int idx = 0; idx < THREAD_SIZE_X; idx += 4) {
        FETCH_FLOAT4(frag_b[0][idx]) = FETCH_FLOAT4(sram_b[0][0][THREAD_SIZE_X * tx + idx]);
    }

    // state machine flag, stands for the 0\1 index in double buffer, switching for write and read.
    int write_stage_idx = 1;
    int tile_index = 0;

    do {
        // load the NEXT tile, first tile has already loaded before, .
        tile_index += BLOCK_SIZE_K;

        // if there is rest tile, load from global to register,
        if (tile_index < K) {
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(a_start[OFFSET(
                        A_TILE_ROW_START + i, // row
                        A_TILE_COL + tile_index, // col
                        K)]);
            }
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(b_start[OFFSET(
                        tile_index + B_TILE_ROW_START + i, // row
                        B_TILE_COL, // col
                        N)]);
            }
        }

        // read and write use mutually exclusive buffer
        int read_stage_idx = write_stage_idx ^ 1;
        int j = 0;
#pragma unroll
        for (; j < BLOCK_SIZE_K - 1; j++) {
            // load NEXT frag of A from sram to register, the frag is load before
#pragma unroll
            for (int idy = 0; idy < THREAD_SIZE_Y; idy += 4) {
                // bit shift instead (j + 1) % 2
                FETCH_FLOAT4(frag_a[(j + 1) % 2][idy]) = FETCH_FLOAT4(
                        sram_a[read_stage_idx][j + 1][THREAD_SIZE_Y * ty + idy]);
            }
            // load the NEXT frag of B from sram to register, the frag is load before
#pragma unroll
            for (int idx = 0; idx < THREAD_SIZE_X; idx += 4) {
                // bit shift instead (j + 1) % 2
                FETCH_FLOAT4(frag_b[(j + 1) % 2][idx]) = FETCH_FLOAT4(
                        sram_b[read_stage_idx][j + 1][THREAD_SIZE_X * tx + idx]);
            }

            // calc the CURRENT using outer-product, intermediate result stored in accum
#pragma unroll
            for (int idy = 0; idy < THREAD_SIZE_Y; idy++) {
#pragma unroll
                for (int idx = 0; idx < THREAD_SIZE_X; idx++) {
                    accum[idy][idx] += frag_a[j % 2][idy] * frag_b[j % 2][idx];
                }
            }
        }

        // load the NEXT tile from register ldg_a\b_reg into sram_a\b
        if (tile_index < K) {
#pragma unroll
            // transpose A
            for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                sram_a[write_stage_idx][A_TILE_COL][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index];
                sram_a[write_stage_idx][A_TILE_COL + 1][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 1];
                sram_a[write_stage_idx][A_TILE_COL + 2][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 2];
                sram_a[write_stage_idx][A_TILE_COL + 3][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 3];
            }
            for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(sram_b[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(
                        ldg_b_reg[ldg_index]);
            }

            // use double buffer, only need one sync
            __syncthreads();
            // switch stage
            write_stage_idx ^= 1;
        }

        // now, let's deal with the last tile.
#pragma unroll
        for (int idy = 0; idy < THREAD_SIZE_Y; idy += 4) {
            int buffer_id = ((j + 1) << 31) >> 31;
            FETCH_FLOAT4(frag_a[0][idy]) = FETCH_FLOAT4(
                    sram_a[read_stage_idx ^ 1][0][THREAD_SIZE_Y * ty + idy]);
        }
#pragma unroll
        for (int idx = 0; idx < THREAD_SIZE_Y; idx += 4) {
            int buffer_id = ((j + 1) << 31) >> 31;
            FETCH_FLOAT4(frag_b[0][idx]) = FETCH_FLOAT4(
                    sram_b[read_stage_idx ^ 1][0][THREAD_SIZE_X * tx + idx]);
        }
#pragma unroll
        for (int idy = 0; idy < THREAD_SIZE_Y; idy++) {
#pragma unroll
            for (int idx = 0; idx < THREAD_SIZE_X; idx++) {
                int buffer_index = (j << 31) >> 31;
                accum[idy][idx] += frag_a[1][idy] * frag_b[1][idx];
            }
        }
    } while (tile_index < K);


    // store back to C
#pragma unroll
    for (int idy = 0; idy < THREAD_SIZE_Y; ++idy) {
#pragma unroll
        for (int idx = 0; idx < THREAD_SIZE_X; idx += 4) {
            FETCH_FLOAT4(out[OFFSET(
                    BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + idy,
                    BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + idx,
                    N)]) = FETCH_FLOAT4(accum[idy][idx]);
        }
    }
}

#pragma mark half_kernel
#define WARP_SIZE 32
using namespace nvcuda;

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

__global__ void
naive_tensorcore_kernel_N_T(half *__restrict__ a, half *__restrict__ b, half *__restrict__ out, const int M,
                            const int N, const int K) {
    int warp_id_n = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warp_id_m = blockIdx.y;

    wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, half> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    int m_start = warp_id_m * TILE_M;
    int n_start = warp_id_n * TILE_N;

    for (int k = 0; k < K; k += TILE_K) {
        if (m_start < M && n_start < N) {
            wmma::load_matrix_sync(a_frag, a + m_start * K + k, K);
            wmma::load_matrix_sync(b_frag, b + n_start * K + k, K);

            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // store out
    if (m_start < M && n_start < N) {
        wmma::store_matrix_sync(out + m_start * N + n_start, acc_frag, N, wmma::mem_row_major);
    }
}

int gemm_float(float *a, float *b, float *out, const int M, const int N, const int K, const int iter, GEMM_OP op,
               float *perf) {

    // warming up
    switch (op) {
        case GEMM_OP::FLOAT_NAIVE_GEMM_N_T: {
            dim3 block(16, 16);
            dim3 grid(UP_DIV(N, block.x), UP_DIV(M, block.y));
            naive_sgemm_kernel_N_T<<<grid, block>>>(a, b, out, M, N, K);
            break;
        }
        case GEMM_OP::FLOAT_SRAM_GEMM_N_N: {
            const int BLOCK_SIZE_M = 128;
            const int BLOCK_SIZE_K = 8;
            const int BLOCK_SIZE_N = 128;
            const int THREAD_SIZE_X = 8;
            const int THREAD_SIZE_Y = 8;
            dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
            dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
            blocked_sram_sgemm_kernel_N_N<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_SIZE_Y, THREAD_SIZE_X>
            <<< dimGrid, dimBlock >>>(a, b, out, M, N, K);
            break;
        }
        default: {
            std::cout << "Unsupported GEMM type!" << std::endl;
            return -1;
        }
    }
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iter; i++) {
        switch (op) {
            case GEMM_OP::FLOAT_NAIVE_GEMM_N_T: {
                dim3 block(16, 16);
                dim3 grid(UP_DIV(N, block.x), UP_DIV(M, block.y));
                naive_sgemm_kernel_N_T<<<grid, block>>>(a, b, out, M, N, K);
                break;
            }
            case GEMM_OP::FLOAT_SRAM_GEMM_N_N: {
                const int BLOCK_SIZE_M = 128;
                const int BLOCK_SIZE_K = 8;
                const int BLOCK_SIZE_N = 128;
                const int THREAD_SIZE_X = 8;
                const int THREAD_SIZE_Y = 8;
                dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
                dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
                blocked_sram_sgemm_kernel_N_N<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_SIZE_Y, THREAD_SIZE_X>
                <<< dimGrid, dimBlock >>>(a, b, out, M, N, K);
                break;
            }
            default: {
                std::cout << "Unsupported GEMM type!" << std::endl;
                return -1;
            }
        }
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float time;
    CHECK_CUDA(cudaEventElapsedTime(&time, start, stop));
    float msecPerMatrixMul = time / iter;
    double flopsPerMatrixMul = 2.0 * (double) M * (double) N * (double) K;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
            "[%s]: [M, N, K] = [%d, %d, %d] ==> Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
            get_test_name(op).c_str(), M, N, K,
            gigaFlops,
            msecPerMatrixMul,
            flopsPerMatrixMul);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    if (perf) { *perf = static_cast<float>(gigaFlops); }
    return 0;
}

int gemm_half(void *a, void *b, void *out, const int M, const int N, const int K, const int iter, GEMM_OP op,
              float *perf) {
    // warming up
    switch (op) {
        case GEMM_OP::HALF_NAIVE_TENSORCORE_N_T: {
            dim3 block(64, 4);
            dim3 grid(UP_DIV(N, TILE_N * 2), UP_DIV(M, TILE_M * block.y));
            naive_tensorcore_kernel_N_T<<<grid, block>>>((half *) a, (half *) b, (half *) out, M, N, K);
            break;
        }
        default: {
            std::cout << "Unsupported GEMM type!" << std::endl;
            return -1;
        }
    }
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iter; i++) {
        switch (op) {
            case GEMM_OP::HALF_NAIVE_TENSORCORE_N_T: {
                dim3 block(64, 4);
                dim3 grid(UP_DIV(N, TILE_N * 2), UP_DIV(M, TILE_M * block.y));
                naive_tensorcore_kernel_N_T<<<grid, block>>>((half *) a, (half *) b, (half *) out, M, N, K);
                break;
            }
            default: {
                std::cout << "Unsupported GEMM type!" << std::endl;
                return -1;
            }
        }
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float time;
    CHECK_CUDA(cudaEventElapsedTime(&time, start, stop));
    float msecPerMatrixMul = time / iter;
    double flopsPerMatrixMul = 2.0 * (double) M * (double) N * (double) K;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
            "[%s]: [M, N, K] = [%d, %d, %d] ==> Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
            get_test_name(op).c_str(), M, N, K,
            gigaFlops,
            msecPerMatrixMul,
            flopsPerMatrixMul);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    if (perf) { *perf = static_cast<float>(gigaFlops); }
    return 0;
}

// modified from: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/4_CUDA_Libraries/simpleCUBLAS/simpleCUBLAS.cpp
int cublas_gemm_float(void **v_handle, float *a, float *b, float *out, const int M, const int N, const int K,
                      const int iter,
                      GEMM_OP op, float *perf) {
    // CUBLAS version 2.0
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t *handle = (cublasHandle_t *) v_handle;
    cudaEvent_t start, stop;
    //Perform warmup operation with cublas
    switch (op) {
        case GEMM_OP::FLOAT_CUBLAS_GEMM_N_N: {
            CHECK_CUBLAS(
                    cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b, N, a, K, &beta, out, N));
            break;
        }
        case GEMM_OP::FLOAT_CUBLAS_GEMM_N_T: {
            CHECK_CUBLAS(
                    cublasSgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, b, K, a, K, &beta, out, N));
            break;
        }
        case GEMM_OP::FLOAT_CUBLAS_GEMM_T_N: {
            CHECK_CUBLAS(
                    cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, &alpha, b, N, a, M, &beta, out, N));
            break;
        }
        case GEMM_OP::FLOAT_CUBLAS_GEMM_T_T: {
            CHECK_CUBLAS(
                    cublasSgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_T, N, M, K, &alpha, b, K, a, M, &beta, out, N));
            break;
        }
        default: {
            std::cout << "Unsupported GEMM type!" << std::endl;
            return -1;
        }
    }
    // Allocate CUDA events that we'll use for timing
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record the start event
    CHECK_CUDA(cudaEventRecord(start));

    for (int j = 0; j < iter; j++) {
        // note cublas is column primary!
        // need to transpose the order
        // so C^T = B^T @ A^T
        switch (op) {
            case GEMM_OP::FLOAT_CUBLAS_GEMM_N_N: {
                CHECK_CUBLAS(
                        cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b, N, a, K, &beta, out, N));
                break;
            }
            case GEMM_OP::FLOAT_CUBLAS_GEMM_N_T: {
                CHECK_CUBLAS(
                        cublasSgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, b, K, a, K, &beta, out, N));
                break;
            }
            case GEMM_OP::FLOAT_CUBLAS_GEMM_T_N: {
                CHECK_CUBLAS(
                        cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, &alpha, b, N, a, M, &beta, out, N));
                break;
            }
            case GEMM_OP::FLOAT_CUBLAS_GEMM_T_T: {
                CHECK_CUBLAS(
                        cublasSgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_T, N, M, K, &alpha, b, K, a, M, &beta, out, N));
                break;
            }
            default: {
                std::cout << "Unsupported GEMM type!" << std::endl;
                return -1;
            }
        }
    }

    // Record the stop event
    CHECK_CUDA(cudaEventRecord(stop));

    // Wait for the stop event to complete
    CHECK_CUDA(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / iter;
    double flopsPerMatrixMul = 2.0 * (double) M * (double) N * (double) K;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
            "[%s]: [M, N, K] = [%d, %d, %d] ==> Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
            get_test_name(op).c_str(), M, N, K,
            gigaFlops,
            msecPerMatrixMul,
            flopsPerMatrixMul);

    if (perf) { *perf = static_cast<float>(gigaFlops); }

    return 0;
}

int cublas_gemm_half(void **v_handle, void *a, void *b, void *out, const int M, const int N, const int K,
                     const int iter,
                     GEMM_OP op, float *perf) {
    // CUBLAS version 2.0
    const __half alpha = 1.0f;
    const __half beta = 0.0f;
    cublasHandle_t *handle = (cublasHandle_t *) v_handle;
    cudaEvent_t start, stop;

    //Perform warmup operation with cublas
    switch (op) {
        case GEMM_OP::HALF_CUBLAS_GEMM_N_N: {
            CHECK_CUBLAS(cublasHgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, (__half *) &alpha, (__half *) b, N,
                                     (__half *) a, K, (__half *) &beta, (__half *) out, N));
            break;
        }
        case GEMM_OP::HALF_CUBLAS_GEMM_N_T: {
            CHECK_CUBLAS(cublasHgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, (__half *) &alpha, (__half *) b, K,
                                     (__half *) a, K, (__half *) &beta, (__half *) out, N));
            break;
        }
        case GEMM_OP::HALF_CUBLAS_GEMM_T_N: {
            CHECK_CUBLAS(cublasHgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, (__half *) &alpha, (__half *) b, N,
                                     (__half *) a, M, (__half *) &beta, (__half *) out, N));
            break;
        }
        case GEMM_OP::HALF_CUBLAS_GEMM_T_T: {
            CHECK_CUBLAS(cublasHgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_T, N, M, K, (__half *) &alpha, (__half *) b, K,
                                     (__half *) a, M, (__half *) &beta, (__half *) out, N));
            break;
        }
        default: {
            std::cout << "Unsupported GEMM type!" << std::endl;
            return -1;
        }
    }

    // Allocate CUDA events that we'll use for timing
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record the start event
    CHECK_CUDA(cudaEventRecord(start));

    for (int j = 0; j < iter; j++) {
        // note cublas is column primary!
        // need to transpose the order
        // so C^T = B^T @ A^T
        switch (op) {
            case GEMM_OP::HALF_CUBLAS_GEMM_N_N: {
                CHECK_CUBLAS(cublasHgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, (__half *) &alpha, (__half *) b, N,
                                         (__half *) a, K, (__half *) &beta, (__half *) out, N));
                break;
            }
            case GEMM_OP::HALF_CUBLAS_GEMM_N_T: {
                CHECK_CUBLAS(cublasHgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, (__half *) &alpha, (__half *) b, K,
                                         (__half *) a, K, (__half *) &beta, (__half *) out, N));
                break;
            }
            case GEMM_OP::HALF_CUBLAS_GEMM_T_N: {
                CHECK_CUBLAS(cublasHgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, (__half *) &alpha, (__half *) b, N,
                                         (__half *) a, M, (__half *) &beta, (__half *) out, N));
                break;
            }
            case GEMM_OP::HALF_CUBLAS_GEMM_T_T: {
                CHECK_CUBLAS(cublasHgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_T, N, M, K, (__half *) &alpha, (__half *) b, K,
                                         (__half *) a, M, (__half *) &beta, (__half *) out, N));
                break;
            }
            default: {
                std::cout << "Unsupported GEMM type!" << std::endl;
                return -1;
            }
        }
    }

    // Record the stop event
    CHECK_CUDA(cudaEventRecord(stop));

    // Wait for the stop event to complete
    CHECK_CUDA(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / iter;
    double flopsPerMatrixMul = 2.0 * (double) M * (double) N * (double) K;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
            "[%s]: [M, N, K] = [%d, %d, %d] ==> Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
            get_test_name(op).c_str(), M, N, K,
            gigaFlops,
            msecPerMatrixMul,
            flopsPerMatrixMul);

    if (perf) { *perf = static_cast<float>(gigaFlops); }

    return 0;
}
