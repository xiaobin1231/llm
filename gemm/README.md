# GEMM — General Matrix Multiplication

> From triple loops to Tensor Cores — a step-by-step exploration of high-performance matrix multiplication on GPU.

## Implementations

| Impl | File | Status | Description |
|------|------|--------|-------------|
| Naive CPU | `naive_cpu.cc` | Done | Triple-nested loop, single-threaded fp32 |
| Naive CUDA | `naive_cuda.cu` | Done | Tiled shared-memory GEMM, fp32 |
| PTX MMA | `mma.cu` | Done | Hand-written `mma.sync` via inline PTX, fp16 accumulation → fp32 output |
| WMMA | `wmma.cu` | WIP | CUDA WMMA API |
| WGMMA | `wgmma.cu` | WIP | SM90+ asynchronous warp-group MMA |
| CuTe | `cute.cu` | WIP | CUTLASS CuTe abstraction |
| Multi-stage CuTe | `multi_stage_cute.cu` | WIP | Multi-stage software pipeline with CuTe |

## Optimization Techniques

These are general GPU GEMM optimization techniques applicable across implementations — from the basic shared-memory tiled version to WMMA, WGMMA, and CuTe. The `mma.cu` implementation serves as a concrete example exercising several of them:

- **Vectorized global → shared loads** — `float4` (16 bytes) per thread per load, reducing the number of memory transactions. Applicable to any kernel that moves data through shared memory.
- **Swizzle remapping** — XOR-based address permutation `col ^ ((row & 3) * 8)` to mitigate shared memory bank conflicts. Relevant whenever shared memory access patterns are regular.
- **`cp.async`** — asynchronous copy from global to shared memory, decoupling data movement from computation. Planned for integration across multiple implementations.
- **Tensor Core acceleration** — leveraging dedicated hardware units (`mma.sync`, WMMA, WGMMA) for matrix multiply-accumulate at higher throughput than CUDA cores.
- **Multi-stage pipelining** — overlapping global memory loads of the next tile with Tensor Core computation of the current tile to hide memory latency. Planned for the CuTe-based implementations.

## Implementation

- [MMA Detail](../docs/mma_zh.md)

## Performance Analysis

Coming soon.
