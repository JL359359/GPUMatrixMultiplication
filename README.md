# GPUMatrixMultiplication
CUDA C++ matrix multiplication: CPU vs naive vs shared-memory tiled kernels, benchmarked on an RTX 4050 (218x over CPU, 749 GFLOPS at N=2048)

Three implementations of dense float matrix multiplication (C = A × B), benchmarked against each other:

1. **CPU baseline:** single-threaded C++ (cache-friendly i-k-j loop order)
2. **Naive CUDA kernel:** one thread per output element, all reads from global memory
3. **Tiled CUDA kernel:** shared-memory tiling to reuse data within each thread block

Built to learn GPU programming fundamentals: the thread/block/grid model, memory hierarchy, warps, and CUDA-event benchmarking.

## Results

**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (compute capability 8.9), Windows, Visual Studio, CUDA Toolkit, Release x64.

### Full results (TILE = 32)

| N | CPU (ms) | Naive (ms) | Tiled (ms) | Tiled vs CPU | Tiled GFLOPS | Max diff |
|---|---------:|-----------:|-----------:|-------------:|-------------:|---------:|
| 256 | 9.50 | 0.090 | 0.071 | 134.6x | 475.6 | 2.29e-05 |
| 512 | 56.54 | 0.549 | 0.438 | 129.0x | 612.5 | 4.58e-05 |
| 1024 | 493.28 | 4.303 | 3.345 | 147.5x | 642.0 | 9.16e-05 |
| 2048 | 4990.91 | 33.212 | 22.943 | 217.5x | 748.8 | 1.83e-04 |

### Effect of tile size (N = 2048)

| TILE | Threads/block | Naive (ms) | Tiled (ms) | Tiled vs naive | Tiled vs CPU | Tiled GFLOPS |
|-----:|--------------:|-----------:|-----------:|---------------:|-------------:|-------------:|
| 8 | 64 | 42.031 | 34.177 | 1.23x | 141.4x | 502.7 |
| 16 | 256 | 34.129 | 24.490 | 1.39x | 207.5x | 701.5 |
| 32 | 1024 | 33.212 | 22.943 | 1.45x | 217.5x | 748.8 |

Correctness: every configuration matched the CPU output with a max absolute difference of 1.83e-4 or less at N = 2048.

## Analysis

- **Larger tiles helped, with diminishing returns.** Going from TILE 8 to 16 cut the tiled kernel's time by about 28%, while 16 to 32 cut it by only about 6%.
- **Why tiling helps:** each value loaded into shared memory is reused TILE times, so global-memory loads drop to roughly 1/TILE of the naive kernel's.
- **Why the measured gain is modest (1.2x to 1.45x):** the theoretical reduction in global loads is much larger, but the GPU's L1/L2 caches already absorb many of the naive kernel's repeated reads. This is my interpretation and is not profiled.
- **The naive kernel also improved with larger block sizes** (42.0 ms to 33.2 ms), likely from better warp utilization and memory coalescing.
- **Still far from peak.** A vendor-tuned library such as cuBLAS would likely be much faster. Next steps would be register blocking and vectorized loads, and comparing against `cublasSgemm`.

## Methodology

- GPU times are averaged over 10 runs after one warm-up, measured with `cudaEvent_t`.
- Host-to-device copies are done once and excluded from kernel timings.
- CPU time is a single run per size, and it varied by about 5% between runs (4831 to 5083 ms at N = 2048), so speedup-vs-CPU differences across tile sizes are partly CPU noise.
- GFLOPS = 2N³ / time.
- Inputs are random floats in [0, 1]. Results are verified against the CPU output.

## Build and run

**Visual Studio (Windows):**
1. Install the NVIDIA CUDA Toolkit (with Visual Studio integration) and an NVIDIA driver.
2. Create a new **CUDA Runtime** project and replace `kernel.cu` with this repo's `kernel.cu`.
3. Select **Release** and **x64**, then press Ctrl+F5.

**Command line:**
```
nvcc -O3 -arch=sm_89 kernel.cu -o matmul
./matmul
```
Change `sm_89` to match your GPU's compute capability. To change the tile size, edit `#define TILE` at the top of `kernel.cu` (16 or 32 recommended; 32 is the maximum since 32 × 32 = 1024 threads per block).

## Files

- `kernel.cu`: CPU baseline, naive kernel, tiled kernel, timing, and verification
- `README.md`: this file

## What I learned

Thread, block, and grid indexing; global versus shared memory; `__syncthreads()` and tile halos; warps and coalesced access; asynchronous kernel launches and event-based timing; and how to report benchmark results honestly, including noise and limits.
