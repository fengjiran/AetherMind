# CPU AddRmsNorm

CPU-05 首段为 FP32 AddRmsNorm 接入 plain 和 identity-packed AVX2 kernel。O2/packed 可在 AVX2/FMA 可用时解析 SIMD descriptor。短 hidden、非单位列 stride 或 ISA 不可用时使用 scalar reference。实现和验收见 [优化方案](add-rmsnorm-optimization.md) 与 [机器证据](benchmarks/)。

输入依次为 `input`、`residual`、`weight`，两个输出依次为 `output` 和 `new_residual`。CPU kernel 的两个 activation、两个输出和 Norm 权重均为 FP32。先将逐元素 Add 舍入并存到 `new_residual`，再用该值计算归一化结果。平方和按 hidden 维升序用 FP64 累加，缩放和乘权重也使用 FP64，最终写回 FP32。

行视图、输出无重叠和 exact packing recipe 由现有 params builder 验证。两个只读 activation 可共享存储。两个输出各自与所有输入和对方输出互斥。支持合法的行 padding、列 stride 和零行。FP16/BF16 A16 的 graph fusion 准入仍由 [精度专题](../../improvement-plan/11-fp16-bf16-execution-support.md) 管理。
