# FP16 / FP32 批量转换

这是 dtypes/backend 的数值 primitive，不新增 GraphOp、Cast pass 或 registry。遵循 [开发与优化工作流](../../guides/operator-development-workflow.md) 的合同、reference、真实调用路径和证据要求。

- O0/O1：checked byte-batch API、scalar reference、固定 RNE、gradual underflow、signed zero、quiet NaN 与浮点环境不变；无重叠、任意对齐、严格长度和安全尾部。
- O2：同时覆盖 batch API、真实 `ConvertRawModelWeightsToFp32` 及 `ExecutableModel::Load`；不以 microbenchmark 推导 token 吞吐。
- O3：[实现方案](fp16-conversion-optimization.md)。
- O4：[DESKTOP-54H5MMI 证据](benchmarks/desktop-54h5mmi-fp16-conversion.md)。

当前实现：FP16→FP32 为 F16C 8 路（提前整数 quieting），FP32→FP16 为 AVX2 8 路整数转换，不要求 FMA；每个方向都有 scalar fallback。AVX-512/SAE、ARM SIMD 和 native 16-bit inference 不在本次实现中。

公共接口见 [API](../../api/public-api.md)，模型入口见 [FP32 compatibility](../../guides/fp32-compatible-inference.md)。
