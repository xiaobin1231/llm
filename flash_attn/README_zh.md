# FlashAttention

> 从朴素实现到性能优化 —— 自底向上理解 Attention Kernel 的设计思想。

FlashAttention 不是某一层面的孤立优化，而是一次算法、系统、硬件协同设计的范例。本目录下的实现循序渐进地引入了现代 Attention Kernel 背后的核心技术。

## 实现列表

| 文件 | 说明 | 核心思路 |
|------|------|----------|
| `flash_attention_cpu_naive_impl.cc` | CPU 参考实现 | 标准 scaled dot-product attention，causal mask |
| `flash_attention_v1_naive_impl.cu` | CUDA 朴素 V1 | Q 分块，逐 tile 重算 softmax |
| `flash_attention_v2_naive_impl.cu` | CUDA 朴素 V2 | Online softmax，减少 SRAM 占用 |
| `flash_attention_v2_performance_impl.cu` | CUDA 优化 V2 | Warp 级并行、`cp.async`、向量化加载 |

## 演进路线

### 1. CPU 参考实现

标准 scaled dot-product attention 的直接翻译：计算 Q·K^T → causal mask → softmax → 乘以 V。作为后续所有 GPU 实现的正确性基准。

### 2. V1 朴素实现（CUDA）

第一个 GPU 版本。引入 **tiling** 策略：将 Q 按 `Br × d_k` 的 tile 划分，每个 thread block 负责一个 Q tile，遍历所有 K/V tile 完成计算。softmax 在每个 tile 内独立重算，结果正确但存在冗余计算。

### 3. V2 朴素实现（CUDA）

用 **online softmax** 替代逐 tile 的独立 softmax：维护全局的 `m`（最大值）和 `l`（和）统计量，在遍历 K/V tile 时增量更新部分输出。这是 FlashAttention 论文最核心的算法洞察 —— 不需要完整物化注意力矩阵即可得到精确结果。

### 4. V2 性能优化（CUDA）

在 V2 算法基础上引入系统层优化：

- **Warp 级并行** — 每个 warp 独立处理一行 Q，减少线程间同步开销
- **`cp.async`** — 异步全局内存到共享内存的数据搬运，隐藏访存延迟
- **向量化加载** — 使用 `float4` 提高内存带宽利用率

## 关键概念

- **Q 分块（Tiling）** — Q 划块放入共享内存，K/V tile 流式遍历
- **Online Softmax** — 增量式计算 softmax，避免物化完整注意力矩阵
- **Causal Mask** — 在 Q·K^T 阶段直接应用上三角 mask
- **GPU 内存层级** — Global → Shared Memory 的数据搬运是性能优化的重点

## 参考文献

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
