# AddRmsNorm FP32 SIMD

O2 的 AddRmsNorm fusion 保留 FP32 residual 给后续 consumer，同时输出归一化结果。AVX2 entry 复用 plain/identity 的 prepared params 和校验。兼容形状的 hidden 至少为 8，所有列 stride 为 1。行 stride 可以不同。相同的 identity recipe 供 scalar 和 AVX2 消费。

第一遍每组 8 个元素向量执行 FP32 Add 并写 `new_residual`，再转成 FP64 计算平方，按原列顺序累加。第二遍从 `new_residual` 读取，每组 4 个元素用 FP64 乘逆 RMS 和权重，再转换回 FP32。尾部使用标量算术。ISA 翻译单元沿用 AVX2/FMA 构建隔离，关闭 fast-math 和 FMA contraction。调用使用固定大小的局部变量和已绑定输出，无额外 workspace。

保留按列顺序的 FP64 归约有利于直接与 reference 对拍，也限制了 SIMD 对累加吞吐的提升。独立 plain RMSNorm AVX2 使用 FP32 归约，和融合路径的数值合同不同。比较时分别记录未融合、融合 scalar 和融合 SIMD。O2 对照仅通过 `PassContext::enable_fused_add_rms_norm` 禁用对应 pass，以隔离 fusion 影响。

后续小算子按模型 profiling 决定。Prepared 与模型级比较，以及当前机器上的热点分布见 [机器证据](benchmarks/)。
