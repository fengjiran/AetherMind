# CPU SiluMul

`SiluMul(gate, up)` 先计算 SiLU(gate)，再乘 up。Llama Dense 构图直接产生该算子，通用图也可由 SiLU 和逐元素 Mul 融合。输入的语义次序由 schema 固定。

CPU-05 提供 FP32 和 BF16 的 AVX2/FMA prepared kernel。连续同形状且元素数至少为 32 时可进入 SIMD。Scalar reference 支持 rank-0、broadcast、合法 stride、零元素和精确原地输出。FP16 保留 reference。

FP32 SIMD 在有界输入上使用分段稳定公式和向量 exp 近似。BF16 先拓宽至 FP32，乘积完成后执行一次 RNE 窄化。两条路径都保留特例和非默认舍入的 scalar 回退。见 [优化与数值合同](silu-mul-optimization.md) 和 [机器证据](benchmarks/)。
