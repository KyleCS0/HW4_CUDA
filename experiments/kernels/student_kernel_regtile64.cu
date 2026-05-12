#include "../math_utils.h"
#include <stdio.h>

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
#define BLOCK_X 16
#define BLOCK_Y 16

__global__ void StudentKernel(int M, int N, int K, float alpha,
                              float *A, float *B, float beta, float *C) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * BLOCK_X + tx;

    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;
    int rowBase = blockRow + ty * TM;
    int colBase = blockCol + tx * TN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++) {
#pragma unroll
        for (int j = 0; j < TN; j++) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (int load = 0; load < 4; load++) {
            int aIndex = tid * 4 + load;
            int aRow = aIndex / BK;
            int aCol = aIndex - aRow * BK;
            int globalRow = blockRow + aRow;
            int globalCol = k0 + aCol;

            As[aRow][aCol] =
                (globalRow < M && globalCol < K)
                    ? A[globalRow * K + globalCol]
                    : 0.0f;

            int bIndex = tid * 4 + load;
            int bRow = bIndex / BN;
            int bCol = bIndex - bRow * BN;
            globalRow = k0 + bRow;
            globalCol = blockCol + bCol;

            Bs[bRow][bCol] =
                (globalRow < K && globalCol < N)
                    ? B[globalRow * N + globalCol]
                    : 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; k++) {
            float aFrag[TM];
            float bFrag[TN];

#pragma unroll
            for (int i = 0; i < TM; i++) {
                aFrag[i] = As[ty * TM + i][k];
            }

#pragma unroll
            for (int j = 0; j < TN; j++) {
                bFrag[j] = Bs[k][tx * TN + j];
            }

#pragma unroll
            for (int i = 0; i < TM; i++) {
#pragma unroll
                for (int j = 0; j < TN; j++) {
                    acc[i][j] += aFrag[i] * bFrag[j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; i++) {
        int row = rowBase + i;
        if (row < M) {
#pragma unroll
            for (int j = 0; j < TN; j++) {
                int col = colBase + j;
                if (col < N) {
                    int idx = row * N + col;
                    C[idx] = alpha * acc[i][j] + beta * C[idx];
                }
            }
        }
    }
}

void runStudent(int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((N + BN - 1) / BN,
              (M + BM - 1) / BM);

    StudentKernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
