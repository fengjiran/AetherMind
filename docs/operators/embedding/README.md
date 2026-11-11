# CPU Embedding

当前实现包含 F32 reference，以及 F16/BF16 weight 到 F32 output 的 plain / identity-packed kernel。W16 的 portable reference 和 feature-gated F16C/AVX2 行级 widening 均经 registered/prepared 与模型执行路径验证。

token ids 接受 contiguous int32、uint32、int64，rank 至少为 1。weight 为正尺寸 contiguous `[vocab, hidden]`，output 为 contiguous `[token_axes..., hidden]`。零 token 为空操作，非空调用要求有效指针。output 与 tokens/weight 不得重叠，布局与字节范围在 binding 时校验。

`EmbeddingParams::output_dtype` undefined 时 semantic output 跟随 weight，显式 F32 启用 W16→F32 语义。新字段的 serde 保留旧裸 `Embedding` 格式。执行逐 token 检查 vocabulary bounds，越界返回 OutOfRange，先前完成的 output rows 保留。

W16 读取采用 byte-safe batch conversion，finite/subnormal/signed zero 位结果精确，NaN 保留 sign/payload 并 quiet。reference 不要求 CPU feature。F16C/AVX2 kernel 依 backend 的 effective features resolve，在 binding 时固定行级函数，执行无 CPU 检测或堆分配。

实现见 [entry](../../../src/backend/cpu/kernels/embedding/embedding_w16_entry.cpp)，测试见 [W16 kernel regression](../../../tests/unit/backend/cpu/kernels/test_cpu_embedding_w16_kernel.cpp)，性能证据见 [本机记录](benchmarks/desktop-54h5mmi-embedding.md)。遵循 [算子工作流](../../guides/operator-development-workflow.md)。
