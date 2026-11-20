#include "common.h"

// Part C, C1: naive scaled dot-product attention (implemented).
//
// Compute, for Q, K, V of shape (N, d):
//   1. scores:  S[i,j] = (1/sqrt(d)) * sum_t Q[i,t] * K[j,t]      (N x N)
//   2. softmax: row-wise over S, SAFE (subtract the row max before exp)
//   3. output:  O[i,t] = sum_j P[i,j] * V[j,t]                    (N x d)
//
// This is the "naive" baseline: materialize the full N x N score matrix S
// in global memory and read it back. That HBM traffic is exactly what the
// tiled C2 kernel removes. All three implementations must produce the same
// attention output; the grader checks each one element-wise against an
// independent reference (the printed checksum is only for your report).
//
// Implementation (three kernels, launched in order in main()):
//   scores_kernel       2-D grid of 16x16 blocks, one thread per S[i,j];
//                       j is on threadIdx.x so the writes to S are coalesced.
//   softmax_rows_kernel one 256-thread block per row: shared-memory tree
//                       reduction for the row max, exp(S - max) with a second
//                       reduction for the row sum, then divide -- in place.
//   output_kernel       one thread per O[i,t] = sum_j P[i,j] * V[j,t]; t is on
//                       threadIdx.x so the reads of V and writes of O coalesce.

// Fill S (N x N, row-major) with the scaled scores: S[i,j] must end up equal
// to scale * dot(Q[i,:], K[j,:]), where Q and K are (N, d) row-major. You
// choose the thread/block decomposition.
__global__ void scores_kernel(int N, int d, double scale,
                               const double *Q, const double *K, double *S)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // key index (fast -> coalesced S writes)
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // query index
    if (i >= N || j >= N) return;
    double acc = 0.0;
    for (int t = 0; t < d; ++t) acc += Q[(size_t)i * d + t] * K[(size_t)j * d + t];
    S[(size_t)i * N + j] = scale * acc;
}

// Replace S in place with its numerically safe row-wise softmax: subtract
// each row's max before exponentiating, then divide by the row sum. You
// choose how to parallelize the per-row reduction.
__global__ void softmax_rows_kernel(int N, double *S)
{
    // One block per row; block-wide max then sum reductions in shared memory.
    __shared__ double red[256];
    double *row = S + (size_t)blockIdx.x * N;
    int tid = threadIdx.x;

    double mx = -INFINITY;
    for (int j = tid; j < N; j += blockDim.x) mx = fmax(mx, row[j]);
    red[tid] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmax(red[tid], red[tid + s]);
        __syncthreads();
    }
    mx = red[0];
    __syncthreads();

    double sum = 0.0;
    for (int j = tid; j < N; j += blockDim.x) {
        double e = exp(row[j] - mx);
        row[j] = e;
        sum += e;
    }
    red[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    sum = red[0];
    for (int j = tid; j < N; j += blockDim.x) row[j] /= sum;
}

// Fill O (N x d, row-major) with O[i,t] = sum_j P[i,j] * V[j,t], where P is
// the softmax weights (N x N, row-major) and V is (N, d) row-major. You
// choose the thread/block decomposition.
__global__ void output_kernel(int N, int d, const double *P, const double *V, double *O)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;  // output column
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // output row
    if (i >= N || t >= d) return;
    double acc = 0.0;
    for (int j = 0; j < N; ++j) acc += P[(size_t)i * N + j] * V[(size_t)j * d + t];
    O[(size_t)i * d + t] = acc;
}

int main(int argc, char **argv)
{
    AttnDims a = parseDims(argc, argv);
    double scale = 1.0 / sqrt((double)a.d);

    double *hQ = makeQ(a), *hK = makeK(a), *hV = makeV(a);
    size_t qkvBytes = (size_t)a.N * a.d * sizeof(double);
    size_t sBytes = (size_t)a.N * a.N * sizeof(double);

    double *dQ, *dK, *dV, *dS, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, qkvBytes));
    CUDA_CHECK(cudaMalloc(&dK, qkvBytes));
    CUDA_CHECK(cudaMalloc(&dV, qkvBytes));
    CUDA_CHECK(cudaMalloc(&dS, sBytes));
    CUDA_CHECK(cudaMalloc(&dO, qkvBytes));
    CUDA_CHECK(cudaMemcpy(dQ, hQ, qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK, qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV, qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dO, 0, qkvBytes));  // zero O before the kernels write it

    cudaEvent_t startE, stopE;
    cudaEventCreate(&startE);
    cudaEventCreate(&stopE);
    cudaEventRecord(startE);
    // Launch configuration: scores on a ceil(N/16) x ceil(N/16) grid of 16x16
    // blocks, softmax with one 256-thread block per row, output on a
    // ceil(d/16) x ceil(N/16) grid of 16x16 blocks. Timed by the given events.
    dim3 blk(16, 16);
    scores_kernel<<<dim3((a.N + 15) / 16, (a.N + 15) / 16), blk>>>(a.N, a.d, scale, dQ, dK, dS);
    softmax_rows_kernel<<<a.N, 256>>>(a.N, dS);
    output_kernel<<<dim3((a.d + 15) / 16, (a.N + 15) / 16), blk>>>(a.N, a.d, dS, dV, dO);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stopE);
    cudaEventSynchronize(stopE);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, startE, stopE);

    double *hO = (double *)malloc(qkvBytes);
    CUDA_CHECK(cudaMemcpy(hO, dO, qkvBytes, cudaMemcpyDeviceToHost));
    double cs = checksum(hO, (size_t)a.N * a.d);

    // Required program output: "checksum,time_ms" with 3 decimals (spec: -3 pts if wrong format).
    printf("%.3f,%.3f\n", cs, (double)ms);
    if ((size_t)a.N * a.d <= 4096) dumpOutput(hO, (size_t)a.N * a.d);
    // Machine-parseable line for the autograder / grade_local.py.
    printf("RESULT,program=c1_naive,N=%d,d=%d,time_ms=%.3f,checksum=%.6f\n",
           a.N, a.d, (double)ms, cs);

    free(hQ); free(hK); free(hV); free(hO);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dS); cudaFree(dO);
    return 0;
}
