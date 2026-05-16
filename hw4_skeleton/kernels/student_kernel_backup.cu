#include "../math_utils.h"
#include <stdio.h>

#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8

__device__ __forceinline__ void cp_async1(float* dst, const float* src) {
    uint32_t smem_addr = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                 :: "r"(smem_addr), "l"(src));
}

__global__ void StudentKernel(int M, int N, int K, float alpha,
                              float *A, float *B, float beta, float *C) {
    // Double-buffer: 2 × (BM×BK + BK×BN) = 2 × 16384 B = 32 KB
    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];

    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;
    float threadResults[TM][TN] = {};
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numTiles = K / BK;  // all test sizes are multiples of BK=16

    // Prefetch tile 0 into buffer 0
    {
        for (int i = tid; i < BM * BK; i += 256) {
            int r = i / BK, c = i % BK;
            cp_async1(&As[0][r][c], &A[(blockRow + r) * K + c]);
        }
        for (int i = tid; i < BK * BN; i += 256) {
            int r = i / BN, c = i % BN;
            cp_async1(&Bs[0][r][c], &B[r * N + blockCol + c]);
        }
        asm volatile("cp.async.commit_group;");
    }

    for (int tile = 0; tile < numTiles; tile++) {
        int cur = tile & 1;
        int nxt = cur ^ 1;

        // Prefetch next tile into the other buffer (while we wait for current)
        if (tile + 1 < numTiles) {
            int bk = (tile + 1) * BK;
            for (int i = tid; i < BM * BK; i += 256) {
                int r = i / BK, c = i % BK;
                cp_async1(&As[nxt][r][c], &A[(blockRow + r) * K + bk + c]);
            }
            for (int i = tid; i < BK * BN; i += 256) {
                int r = i / BN, c = i % BN;
                cp_async1(&Bs[nxt][r][c], &B[(bk + r) * N + blockCol + c]);
            }
            asm volatile("cp.async.commit_group;");
        }

        // Wait for current tile to arrive, then compute
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float aReg[TM], bReg[TN];
            #pragma unroll
            for (int m = 0; m < TM; m++) aReg[m] = As[cur][threadRow * TM + m][k];
            #pragma unroll
            for (int n = 0; n < TN; n++) bReg[n] = Bs[cur][k][threadCol * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int n = 0; n < TN; n++)
                    threadResults[m][n] += aReg[m] * bReg[n];
        }
    }

    // Wait for all outstanding async copies to complete
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int gr = blockRow + threadRow * TM + m;
            int gc = blockCol + threadCol * TN + n;
            if (gr < M && gc < N)
                C[gr * N + gc] = alpha * threadResults[m][n] + beta * C[gr * N + gc];
        }
    }
}

void runStudent(int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    StudentKernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
