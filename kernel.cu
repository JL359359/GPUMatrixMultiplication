// Matrix multiplication benchmark: CPU vs naive CUDA vs shared-memory tiled CUDA
// Build in Visual Studio (CUDA Runtime project), Release | x64.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE 32

#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            printf("CUDA error: %s (%s:%d)\n", cudaGetErrorString(err),      \
                   __FILE__, __LINE__);                                      \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// ---------------- CPU baseline ----------------
void matmulCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N * N; i++) C[i] = 0.0f;
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            float a = A[i * N + k];
            for (int j = 0; j < N; j++)
                C[i * N + j] += a * B[k * N + j];
        }
}

// ---------------- Naive GPU kernel ----------------
__global__ void matmulNaive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// ---------------- Tiled (shared memory) GPU kernel ----------------
__global__ void matmulTiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    int numTiles = (N + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = sum;
}

typedef void (*KernelFn)(const float*, const float*, float*, int);

// Returns average kernel time in ms using CUDA events
float timeGPU(KernelFn kernel, const float* dA, const float* dB, float* dC,
    int N, int reps = 10) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    kernel << <grid, block >> > (dA, dB, dC, N);   // warm-up
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    for (int r = 0; r < reps; r++)
        kernel << <grid, block >> > (dA, dB, dC, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return ms / reps;
}

float maxAbsDiff(const float* x, const float* y, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) m = fmaxf(m, fabsf(x[i] - y[i]));
    return m;
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (compute capability %d.%d)\n\n", prop.name, prop.major, prop.minor);

    int sizes[] = { 256, 512, 1024, 2048 };
    printf("%6s | %10s | %10s | %10s | %9s | %9s | %9s | %9s\n",
        "N", "CPU(ms)", "Naive(ms)", "Tiled(ms)", "Naive x", "Tiled x",
        "Tiled GF", "MaxDiff");
    printf("-------+------------+------------+------------+-----------+-----------+-----------+----------\n");

    for (int s = 0; s < 4; s++) {
        int N = sizes[s];
        size_t count = (size_t)N * N;
        size_t bytes = count * sizeof(float);

        float* hA = (float*)malloc(bytes);
        float* hB = (float*)malloc(bytes);
        float* hC_cpu = (float*)malloc(bytes);
        float* hC_gpu = (float*)malloc(bytes);
        for (size_t i = 0; i < count; i++) {
            hA[i] = (float)rand() / RAND_MAX;
            hB[i] = (float)rand() / RAND_MAX;
        }

        // CPU timing (single run)
        auto t0 = std::chrono::high_resolution_clock::now();
        matmulCPU(hA, hB, hC_cpu, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        float cpuMs = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // GPU buffers: copy once
        float* dA, * dB, * dC;
        CHECK(cudaMalloc(&dA, bytes));
        CHECK(cudaMalloc(&dB, bytes));
        CHECK(cudaMalloc(&dC, bytes));
        CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

        float naiveMs = timeGPU(matmulNaive, dA, dB, dC, N);
        float tiledMs = timeGPU(matmulTiled, dA, dB, dC, N);

        // Verify the tiled result against the CPU result
        CHECK(cudaMemcpy(hC_gpu, dC, bytes, cudaMemcpyDeviceToHost));
        float diff = maxAbsDiff(hC_cpu, hC_gpu, (int)count);

        double gflops = (2.0 * N * N * N) / (tiledMs * 1e6);

        printf("%6d | %10.2f | %10.3f | %10.3f | %8.1fx | %8.1fx | %9.1f | %9.2e\n",
            N, cpuMs, naiveMs, tiledMs, cpuMs / naiveMs, cpuMs / tiledMs,
            gflops, diff);

        CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
        free(hA); free(hB); free(hC_cpu); free(hC_gpu);
    }

    printf("\nDone.\n");
    return 0;
}