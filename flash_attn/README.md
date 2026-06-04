# FlashAttention

> From naive implementation to performance optimization — understanding attention kernels from the ground up.

FlashAttention is not a single optimization technique. It's a rare example of algorithm, system, and hardware co-design. This directory contains implementations that progressively introduce the key ideas behind modern attention kernels.

## Implementations

| File | Description | Key Ideas |
|------|-------------|-----------|
| `flash_attention_cpu_naive_impl.cc` | CPU reference implementation | Standard scaled dot-product attention, causal mask |
| `flash_attention_v1_naive_impl.cu` | CUDA naive V1 | Tiling over Q, recomputing softmax per tile |
| `flash_attention_v2_naive_impl.cu` | CUDA naive V2 | Online softmax, reduced SRAM footprint |
| `flash_attention_v2_performance_impl.cu` | CUDA optimized V2 | Warp-level parallelism, `cp.async`, vectorized loads |

## Progression

### 1. CPU Reference

A straightforward implementation of scaled dot-product attention — compute Q·K^T, apply causal mask, softmax, then multiply by V. Serves as the correctness baseline.

### 2. V1 Naive (CUDA)

The first GPU version. Introduces tiling: Q is partitioned into tiles of size `Br × d_k`, and each thread block computes one Q tile against all K/V tiles. Softmax is recomputed per tile, which is correct but redundant.

### 3. V2 Naive (CUDA)

Replaces per-tile softmax recomputation with **online softmax** — maintaining running `m` (max) and `l` (sum) statistics across KV tiles, updating the partial output in-place. This is the core algorithmic insight from the FlashAttention paper.

### 4. V2 Performance (CUDA)

Takes the V2 algorithm and applies systems-level optimizations:

- **Warp-level reduction** — each warp handles one Q row independently, reducing inter-thread synchronization
- **`cp.async`** — asynchronous global-to-shared memory copies, hiding memory latency
- **Vectorized loads** — `float4` loads for higher memory bandwidth utilization

## Key Concepts

- **Tiling over Q** — Q is divided into tiles to fit in shared memory; K/V tiles stream through
- **Online softmax** — incremental softmax that avoids materializing the full attention matrix
- **Causal masking** — upper-triangular mask applied during Q·K^T computation
- **GPU memory hierarchy** — global → shared memory data movement is critical to performance

## References

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
