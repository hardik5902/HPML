#include "common.h"
#include <cublas_v2.h>

// Part C, C3: the same attention built from cuBLAS GEMMs (implemented).
//
// The two matmuls in attention (Q*K^T and P*V) are exactly what a BLAS
// library is tuned for. Call cublasDgemm for both and keep only the softmax
// as a hand-written kernel (BLAS has no softmax). Must produce the same
// checksum as c1_naive / c2_tiled.
//
// Q, K, V, S, O are all row-major. cuBLAS is column-major, so you must work
// out the op flags, operand order, and leading dimensions yourself so that
// each GEMM produces the row-major result the next step expects; your report
// explains that mapping.
//
// Implementation: S = scale*Q*K^T and O = P*V are each one cublasDgemm call;
// the safe row softmax between them is the same kernel as C1. A row-major
// r x c buffer is read by cuBLAS as its column-major transpose (c x r, ld c),
// so no explicit transposes are needed (see the comments at each call).

#define CUBLAS_CHECK(x) do { \
    cublasStatus_t _s = (x); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)_s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

// Safe row softmax of S (N x N, row-major) in place -- same kernel as C1:
// one 256-thread block per row, shared-memory reductions for max and sum.
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

int main(int argc, char **argv)
{
    AttnDims a = parseDims(argc, argv);
    int N = a.N, d = a.d;
    double scale = 1.0 / sqrt((double)d);

    double *hQ = makeQ(a), *hK = makeK(a), *hV = makeV(a);
    size_t qkvBytes = (size_t)N * d * sizeof(double);
    size_t sBytes = (size_t)N * N * sizeof(double);

    double *dQ, *dK, *dV, *dS, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, qkvBytes));
    CUDA_CHECK(cudaMalloc(&dK, qkvBytes));
    CUDA_CHECK(cudaMalloc(&dV, qkvBytes));
    CUDA_CHECK(cudaMalloc(&dS, sBytes));
    CUDA_CHECK(cudaMalloc(&dO, qkvBytes));
    CUDA_CHECK(cudaMemset(dS, 0, sBytes));
    CUDA_CHECK(cudaMemset(dO, 0, qkvBytes));
    CUDA_CHECK(cudaMemcpy(dQ, hQ, qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK, qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV, qkvBytes, cudaMemcpyHostToDevice));

    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));

    // alpha/beta scalars for the two GEMMs (the QK^T GEMM uses scale):
    const double alpha_s = scale, beta0 = 0.0, alpha1 = 1.0;
    (void)alpha_s; (void)beta0; (void)alpha1;

    // Warm-up (outside the timed region): one full untimed pass so cuBLAS pays
    // its one-time initialization before timing. It is a complete matching
    // pair -- S = scale*Q*K^T, softmax, then O = P*V reading that S -- so every
    // QK^T GEMM is followed by the PV GEMM that consumes it. S and O are fully
    // overwritten by the timed run (beta = 0).
    CUBLAS_CHECK(cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, N, d,
                             &alpha_s, dK, d, dQ, d, &beta0, dS, N));
    softmax_rows_kernel<<<N, 256>>>(N, dS);
    CUBLAS_CHECK(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, d, N, N,
                             &alpha1, dV, d, dS, N, &beta0, dO, d));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t startE, stopE;
    cudaEventCreate(&startE);
    cudaEventCreate(&stopE);
    cudaEventRecord(startE);

    // Timed block: 1. S = scale*Q*K^T (cublasDgemm), 2. row softmax of S
    // (<<<N, 256>>>), 3. O = P*V (cublasDgemm). Results are row-major.
    // Row-major X (r x c, ld=c) is the column-major matrix X^T (c x r, ld=c).
    // GEMM 1 (column-major view): S^T(NxN) = Kc^T * Qc, Kc=K^T (d x N, ld d).
    CUBLAS_CHECK(cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, N, d,
                             &alpha_s, dK, d, dQ, d, &beta0, dS, N));
    softmax_rows_kernel<<<N, 256>>>(N, dS);
    // GEMM 2: O^T(d x N) = V^T(d x N, ld d) * P^T(N x N, ld N).
    CUBLAS_CHECK(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, d, N, N,
                             &alpha1, dV, d, dS, N, &beta0, dO, d));

    cudaEventRecord(stopE);
    cudaEventSynchronize(stopE);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, startE, stopE);

    double *hO = (double *)malloc(qkvBytes);
    CUDA_CHECK(cudaMemcpy(hO, dO, qkvBytes, cudaMemcpyDeviceToHost));
    double cs = checksum(hO, (size_t)N * d);

    printf("%.3f,%.3f\n", cs, (double)ms);
    if ((size_t)N * d <= 4096) dumpOutput(hO, (size_t)N * d);
    printf("RESULT,program=c3_cublas,N=%d,d=%d,time_ms=%.3f,checksum=%.6f\n",
           N, d, (double)ms, cs);

    cublasDestroy(h);
    free(hQ); free(hK); free(hV); free(hO);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dS); cudaFree(dO);
    return 0;
}
