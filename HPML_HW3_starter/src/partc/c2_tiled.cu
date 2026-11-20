#include "common.h"

// Part C, C2: tiled, FlashAttention-style attention (implemented).
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
// Implementation: one block (TPB threads) per query row; K and V are
// streamed in tiles of BC keys through shared memory, and the online-softmax
// state (running max m, running denominator l, rescaled accumulator acc[d])
// is carried across tiles, so the N x N scores never reach global memory.
// When a tile raises the max from m to m', the old l and acc are multiplied
// by alpha = exp(m - m') before the new tile's terms are added.
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
    CUDA_CHECK(cudaMemset(dO, 0, qkvBytes));  // zero O before the kernel writes it

    cudaEvent_t startE, stopE;
    cudaEventCreate(&startE);
    cudaEventCreate(&stopE);
    cudaEventRecord(startE);
    // Launch: N blocks (one per query row) of TPB threads, with dynamic shared
    // memory for q[d], Ks[BC][d+1], Vs[BC][d+1] and s[BC] (padded rows avoid
    // bank conflicts). Opt in to >48 KB shared memory only if d needs it.
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
