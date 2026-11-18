#include "common.h"

// TODO (Part C, C2 -- 25 pts): tiled, FlashAttention-style attention.
//
// The naive C1 kernel writes the whole N x N score matrix to global memory
// and reads it back. This kernel must NOT materialize it. Compute the same
// output as C1 by processing the keys and values in tiles and carrying the
// softmax statistics across tiles, so the full row of scores is never stored
// at once. You decide what to stage in shared memory. Derive the update that
// lets you fold each new tile of scores into a running result without having
// first seen the whole row; the lecture's FlashAttention material is the
// reference for the idea, but the recurrence is yours to work out.
//
// Choose the block-to-row mapping and the tile width yourself, and place a
// barrier wherever one thread's shared-memory writes must be visible to the
// others before they read. The output must match C1 element by element.
//
// You own the whole implementation: the block-to-row mapping, the tile
// width, the shared-memory layout, and the launch configuration in main().
// The stub is empty and the driver zeros the output, so the program compiles
// and runs but will not match the reference until you implement the kernel
// and its launch.
#define BC 16        // keys per tile
#define TPB 128      // threads per block
#define MAXACC 8     // supports d <= TPB * MAXACC = 1024

// One block per query row i. Shared memory (dynamic): q[d], Ks[BC][d+1],
// Vs[BC][d+1], s[BC]. Running state per row: max m, denominator l, and the
// output accumulator acc[d] (distributed over threads in registers).
__global__ void attn_tiled(int N, int d, double scale,
                            const double *Q, const double *K, const double *V, double *O)
{
    extern __shared__ double sm[];
    int ld = d + 1;                 // padding avoids bank conflicts
    double *q  = sm;
    double *Ks = q + d;
    double *Vs = Ks + BC * ld;
    double *s  = Vs + BC * ld;

    int tid = threadIdx.x;
    int i = blockIdx.x;

    for (int t = tid; t < d; t += TPB) q[t] = Q[(size_t)i * d + t];

    double acc[MAXACC];
    #pragma unroll
    for (int r = 0; r < MAXACC; ++r) acc[r] = 0.0;
    double m = -INFINITY, l = 0.0;

    for (int j0 = 0; j0 < N; j0 += BC) {
        int bc = min(BC, N - j0);
        // Cooperative, coalesced load of the K and V tiles.
        for (int idx = tid; idx < bc * d; idx += TPB) {
            int jj = idx / d, t = idx % d;
            Ks[jj * ld + t] = K[(size_t)(j0 + jj) * d + t];
            Vs[jj * ld + t] = V[(size_t)(j0 + jj) * d + t];
        }
        __syncthreads();                       // tiles + q visible

        if (tid < bc) {
            double dot = 0.0;
            for (int t = 0; t < d; ++t) dot += q[t] * Ks[tid * ld + t];
            s[tid] = scale * dot;
        }
        __syncthreads();                       // scores visible

        // Every thread redundantly folds the tile into (m, l); identical results.
        double tmax = -INFINITY;
        for (int jj = 0; jj < bc; ++jj) tmax = fmax(tmax, s[jj]);
        double mnew = fmax(m, tmax);
        double alpha = exp(m - mnew);          // exp(-inf)=0 on first tile
        double psum = 0.0;
        double p[BC];
        #pragma unroll
        for (int jj = 0; jj < BC; ++jj) {
            p[jj] = (jj < bc) ? exp(s[jj] - mnew) : 0.0;
            psum += p[jj];
        }
        l = l * alpha + psum;
        m = mnew;

        // Rescale the old accumulator by alpha, then add this tile's P*V.
        #pragma unroll
        for (int r = 0; r < MAXACC; ++r) {
            int t = tid + r * TPB;
            if (t < d) {
                double a = acc[r] * alpha;
                for (int jj = 0; jj < bc; ++jj) a += p[jj] * Vs[jj * ld + t];
                acc[r] = a;
            }
        }
        __syncthreads();                       // done reading tiles/s before overwrite
    }

    #pragma unroll
    for (int r = 0; r < MAXACC; ++r) {
        int t = tid + r * TPB;
        if (t < d) O[(size_t)i * d + t] = acc[r] / l;
    }
}

int main(int argc, char **argv)
{
    AttnDims a = parseDims(argc, argv);
    double scale = 1.0 / sqrt((double)a.d);

    double *hQ = makeQ(a), *hK = makeK(a), *hV = makeV(a);
    size_t qkvBytes = (size_t)a.N * a.d * sizeof(double);

    double *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, qkvBytes));
    CUDA_CHECK(cudaMalloc(&dK, qkvBytes));
    CUDA_CHECK(cudaMalloc(&dV, qkvBytes));
    CUDA_CHECK(cudaMalloc(&dO, qkvBytes));
    CUDA_CHECK(cudaMemcpy(dQ, hQ, qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK, qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV, qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dO, 0, qkvBytes));  // defined output until you launch your kernel

    cudaEvent_t startE, stopE;
    cudaEventCreate(&startE);
    cudaEventCreate(&stopE);
    cudaEventRecord(startE);
    // TODO (C2): choose the grid, block, and dynamic shared-memory size for
    // your design and launch attn_tiled here. The timing scaffold around
    // this launch is provided; do not change it.
    size_t shmem = (size_t)(a.d + 2 * BC * (a.d + 1) + BC) * sizeof(double);
    if (shmem > 48 * 1024)
        CUDA_CHECK(cudaFuncSetAttribute(attn_tiled, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem));
    attn_tiled<<<a.N, TPB, shmem>>>(a.N, a.d, scale, dQ, dK, dV, dO);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stopE);
    cudaEventSynchronize(stopE);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, startE, stopE);

    double *hO = (double *)malloc(qkvBytes);
    CUDA_CHECK(cudaMemcpy(hO, dO, qkvBytes, cudaMemcpyDeviceToHost));
    double cs = checksum(hO, (size_t)a.N * a.d);

    printf("%.3f,%.3f\n", cs, (double)ms);
    if ((size_t)a.N * a.d <= 4096) dumpOutput(hO, (size_t)a.N * a.d);
    printf("RESULT,program=c2_tiled,N=%d,d=%d,time_ms=%.3f,checksum=%.6f\n",
           a.N, a.d, (double)ms, cs);

    free(hQ); free(hK); free(hV); free(hO);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return 0;
}
