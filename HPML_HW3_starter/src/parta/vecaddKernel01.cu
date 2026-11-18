#include "vecaddKernel.h"

// vecaddKernel01: COALESCED vector add (Part A, Q2).
//
// Grid-stride-style mapping: at loop iteration j, thread tid touches
// element j*totalThreads + tid. Consecutive threads in a warp therefore touch
// consecutive floats (one 128-byte transaction per warp per array), while each
// thread still handles exactly valuesPerThread elements, so the same N elements
// are covered as in vecaddKernel00.
__global__ void AddVectors(const float *A, const float *B, float *C, int valuesPerThread)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;
    for (int j = 0; j < valuesPerThread; ++j) {
        int idx = j * totalThreads + tid;
        C[idx] = A[idx] + B[idx];
    }
}
