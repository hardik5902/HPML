#include "matmultKernel.h"

#define SUBTILE (FOOTPRINT_SIZE / BLOCK_SIZE)  // = 2

__device__ float GetElement(const Matrix m, int row, int col) {
    return m.elements[row * m.stride + col];
}
__device__ void SetElement(Matrix m, int row, int col, float value) {
    m.elements[row * m.stride + col] = value;
}
__device__ Matrix GetSubMatrix(Matrix m, int row, int col, int size) {
    Matrix sub;
    sub.width = size;
    sub.height = size;
    sub.stride = m.stride;
    sub.elements = &m.elements[m.stride * size * row + size * col];
    return sub;
}

// matmultKernel01: register-blocked tiled matmul (Part A, Q4).
//
// Thread block: 16x16 = 256 threads; block tile of C: 32x32 (FOOTPRINT_SIZE).
// Each thread computes a 2x2 subtile at rows {ty, ty+16}, cols {tx, tx+16}.
// Using a stride of BLOCK_SIZE (not adjacent columns) keeps every global load,
// shared-memory access and global store coalesced / conflict-free: consecutive
// threadIdx.x always touch consecutive columns.
// Each thread loads 4 elements of As and 4 of Bs per tile (coalesced), then
// the inner k-loop (fully unrolled) does 4 FMAs per 4 shared loads, and each
// value loaded from global memory is reused 32 times (vs 16 in kernel00).
__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol, FOOTPRINT_SIZE);

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    float c00 = 0.f, c01 = 0.f, c10 = 0.f, c11 = 0.f;

    __shared__ float As[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
    __shared__ float Bs[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

    int numTiles = A.width / FOOTPRINT_SIZE;
    for (int m = 0; m < numTiles; ++m) {
        Matrix Asub = GetSubMatrix(A, blockRow, m, FOOTPRINT_SIZE);
        Matrix Bsub = GetSubMatrix(B, m, blockCol, FOOTPRINT_SIZE);

        #pragma unroll
        for (int i = 0; i < SUBTILE; ++i)
            #pragma unroll
            for (int j = 0; j < SUBTILE; ++j) {
                As[ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] =
                    GetElement(Asub, ty + i * BLOCK_SIZE, tx + j * BLOCK_SIZE);
                Bs[ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] =
                    GetElement(Bsub, ty + i * BLOCK_SIZE, tx + j * BLOCK_SIZE);
            }
        __syncthreads();

        #pragma unroll
        for (int e = 0; e < FOOTPRINT_SIZE; ++e) {
            float a0 = As[ty][e];
            float a1 = As[ty + BLOCK_SIZE][e];
            float b0 = Bs[e][tx];
            float b1 = Bs[e][tx + BLOCK_SIZE];
            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }
        __syncthreads();
    }

    SetElement(Csub, ty,              tx,              c00);
    SetElement(Csub, ty,              tx + BLOCK_SIZE, c01);
    SetElement(Csub, ty + BLOCK_SIZE, tx,              c10);
    SetElement(Csub, ty + BLOCK_SIZE, tx + BLOCK_SIZE, c11);
}
