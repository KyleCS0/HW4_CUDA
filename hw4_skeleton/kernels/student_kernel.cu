#include "../math_utils.h"
#include <stdio.h>

// =============================================================================
// HW4: CUDA Matmul Optimization
//
// Implement your optimized single-precision GEMM here:
//     C = alpha * (A @ B) + beta * C      (row-major, M = N = K per test size)
//
// Hard rules (violation => 0 points for the performance / rank components):
//   1. DO NOT call cuBLAS, cuDNN, CUTLASS, or any vendor GEMM library.
//   2. DO NOT modify any file outside kernels/student_kernel.cu.
//   3. The signature of runStudent() MUST remain unchanged — TA's grading
//      harness calls it directly.
//
// Your kernel is verified against the cuBLAS reference (tolerance 1e-2) for
// all sizes in {128, 256, 512, 1024, 2048, 4096}. Failing any size zeroes out
// the performance and rank components of the grade.
// =============================================================================

__global__ void StudentKernel(int M, int N, int K, float alpha,
                              float *A, float *B, float beta, float *C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }

    int idx = row * N + col;
    C[idx] = alpha * sum + beta * C[idx];
}

void runStudent(int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    StudentKernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
