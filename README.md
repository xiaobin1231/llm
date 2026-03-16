# Background
As for me, writing implementations of FlashAttention from scratch is a highly commendable and challenging engineering exercise!

# Introduction
## [self_attention_cpu_naive_impl](https://github.com/xiaobin1231/llm/blob/main/transformer/self_attention_cpu_naive_impl.cc)

## [flash_attention_v1_naive_impl.cu](https://github.com/xiaobin1231/llm/blob/main/transformer/flash_attention_v1_naive_impl.cu)
This implementation has significant bottlenecks regarding hardware utilization and flexibility. Here is a detailed breakdown of why this code struggles with performance and generality.
- Uncoalesced Global Memory Access: Threads (tx) iterate over matrix rows while the loop variable (d) iterates over columns. This strided memory access pattern prevents the warp from loading contiguous chunks of memory simultaneously, severely bottlenecking global memory bandwidth.

- Extremely Low Block Occupancy: Kernel configuration uses exactly 32 threads per block (dim3 block(Br) where Br = 32), which equals a single warp. GPU performance relies heavily on latency hiding—switching to other active warps when one is waiting for memory. With only one warp per block, the Streaming Multiprocessor (SM) sits completely idle during memory stalls.

- Absence of Tensor Core Utilization: The matrix multiplications (Q @ K.T and S @ V) are implemented using standard scalar FMA (Fused Multiply-Add) operations in nested for loops. Modern GPUs achieve peak deep learning throughput via Tensor Cores, which require specialized warp-level matrix multiply-accumulate (wmma) instructions or NVIDIA's CuTe library.

- Shared Memory Bank Conflicts: Accessing shared memory with a stride that is a multiple of 32 (like your d_k = 64) causes multiple threads in a warp to hit the same memory bank simultaneously. This forces the hardware to serialize the memory requests, drastically reducing shared memory throughput.

- Rigid Thread-to-Dimension Coupling: Tying the block size strictly to the tile size (Br == blockDim.x) removes the flexibility to tune thread counts independently for maximum occupancy. Furthermore, the code assumes seq_len is perfectly divisible by Br and Bc, which will cause out-of-bounds memory access (and likely a core dump) for arbitrary or unpadded sequence lengths.

- Strict FP32 Precision