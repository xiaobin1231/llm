# LLM Kernel Lab

> FlashAttention · GEMM · CUDA · Tensor Core · CUTLASS · CuTe

A personal learning and experimentation repository focused on high-performance CUDA kernels, including FlashAttention, GEMM, Tensor Core programming, CUTLASS, CuTe, and modern GPU optimization techniques.

🇨🇳 **中文版本：** [README_zh.md](./README_zh.md)

---

# The Road to Mastering Performance Is Muddy, but My Passion Remains

## Introduction

Performance optimization is a vast topic.

Making a program work is relatively easy. Making it run efficiently and fully utilize modern hardware is a completely different challenge.

In my view, performance optimization can be understood through three layers.

### Hardware

Understanding hardware is the foundation of optimization.

General-purpose operators can achieve decent performance, but they rarely outperform implementations specifically designed for a target architecture. As algorithms mature and ecosystems stabilize, specialized accelerators and ASICs inevitably become the next step forward.

### Systems

If hardware provides computational power, system architecture determines how that power is utilized.

Modern AI systems are inherently heterogeneous. CPUs, GPUs, NPUs, and memory subsystems must work together efficiently. The challenge is not merely executing workloads, but orchestrating resources to maximize throughput while minimizing latency.

The value of system design is not simply making models run, but making them serve users efficiently.

### Algorithms and Models

Algorithms ultimately define the upper bound.

Model architecture, parameter count, quantization precision, pruning strategies, and distillation techniques all involve trade-offs.

Achieving high accuracy with low-precision computation remains one of the most challenging goals in modern AI systems.

---

## Why FlashAttention

The first time I read the FlashAttention paper, what impressed me most was not the speedup itself, but the depth of engineering behind it.

Online Softmax is an algorithmic innovation.

Multi-stage pipelining is a systems optimization.

Tensor Core and TMA utilization require deep understanding of GPU hardware.

FlashAttention is not a single optimization technique. It is a rare example of algorithm, system, and hardware co-design.

Reading papers and studying source code can teach concepts, but true understanding comes from implementation.

> What I hear, I may forget.
> What I see, I may remember.
> What I do, I understand.

This repository was created to walk that path myself.

Here I document my experiments, implementations, and thoughts on CUDA, GEMM, FlashAttention, and modern GPU optimization techniques.

---

## Project Structure

<div align="center">
  <img src="./images/repo_roadmap.svg" alt="Learning Roadmap" width="100%"/>
</div>

### 🐭 FlashAttention

Implementations and experiments exploring modern attention kernels, from basic versions to highly optimized designs.

👉 [Enter FlashAttention](./flash_attn/README.md)

### 🐭 GEMM

Matrix multiplication kernels, Tensor Core programming, CUDA optimization techniques, CUTLASS, and CuTe.

👉 [Enter GEMM](./gemm/README.md)

---

## Goals

This repository is not intended to become a production-ready framework.

Instead, it serves as a place to understand and reproduce the ideas behind:

* FlashAttention
* GEMM optimization
* Tensor Core programming
* CUDA kernel optimization
* CUTLASS / CuTe
* Modern GPU architectures

The goal is not only to know *what* these techniques are, but also to understand *why* they work.

---

If these experiments help someone learn GPU optimization a little faster, then this repository has already achieved more than I expected.

And hopefully, I never lose the passion that brought me here in the first place.
