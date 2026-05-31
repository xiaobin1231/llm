# GEMM实现与优化方法总结

## 基本范式
- 朴素cpu
- 朴素cuda
- ptx mma
- wmma
- wgmma
- cute

## 优化技巧
- Global Mem -> Shared Mem向量化加载
- Swizzle重映射缓解bank冲突
- 异步数据搬运
- Tensor Core加速
- 多阶段流水线加速，数据搬运与矩阵计算并行

## 性能分析
待补充。。。