# GEMM — 矩阵乘法实现与优化

> 从三重循环到 Tensor Core —— 一步步探索 GPU 上高性能矩阵乘法的实现方法。

## 实现列表

| 实现 | 文件 | 状态 | 说明 |
|------|------|------|------|
| 朴素 CPU | `naive_cpu.cc` | 已完成 | 三重循环，单线程 fp32 |
| 朴素 CUDA | `naive_cuda.cu` | 已完成 | 共享内存分块 GEMM，fp32 |
| PTX MMA | `mma.cu` | 已完成 | 手写 `mma.sync` 内联 PTX 汇编，fp16 累加 → fp32 输出 |
| WMMA | `wmma.cu` | 开发中 | CUDA WMMA API |
| WGMMA | `wgmma.cu` | 开发中 | SM90+ 异步 warp-group MMA |
| CuTe | `cute.cu` | 开发中 | CUTLASS CuTe 抽象 |
| Multi-stage CuTe | `multi_stage_cute.cu` | 开发中 | 基于 CuTe 的多阶段流水线 |

## 优化技术

以下是 GPU GEMM 实现中通用的优化手段，从基本的共享内存分块版本到 WMMA、WGMMA、CuTe 均可应用。`mma.cu` 作为一个具体示例，实践了其中的多项技术：

- **向量化全局内存加载** — 单线程单次读写 `float4`（16 Bytes），减少访存事务数。任何通过共享内存搬运数据的 kernel 都可以采用。
- **Swizzle 重映射** — 通过 `col ^ ((row & 3) * 8)` 的 XOR 地址变换缓解共享内存 bank 冲突。只要共享内存访问模式规整，该技巧就适用。
- **`cp.async` 异步拷贝** — 将全局内存到共享内存的数据搬运与计算解耦。计划在多个实现中集成。
- **Tensor Core 加速** — 利用专用硬件单元（`mma.sync`、WMMA、WGMMA）进行矩阵乘加，吞吐量远超 CUDA Core。
- **多阶段流水线** — 将下一块数据的全局内存加载与当前块的 Tensor Core 计算 overlap，隐藏访存延迟。计划在 CuTe 相关实现中落地。

## 实现
- [MMA 详解](../docs/mma_zh.md)

## 性能分析

待补充。
