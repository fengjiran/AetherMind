# CPU KVCacheUpdate 与 P3 KV16 合同

当前 CPU production kernel 含 F32（向 F32 cache 写 F32 activation）与 KV16（F32 activation 经固定 RNE 窄化写入 F16/BF16 cache）reference；operator inference 已接受四路同型与 A32→KV16 mixed 签名，`state_dtype` 参与 selector 身份。**P3 mixed-KV reference 已实现，并在 reduced fixture 上通过四组合模型验收与 plan 事务回归（F16/BF16）；§2.8 校准数值 fixture 的正式生产验收已通过，SIMD/优化路径仍待推进。** 总体支持矩阵、配置和验收预算见 [11 号提案 §2.7–2.8](../../improvement-plan/11-fp16-bf16-execution-support.md)，执行遵循 [算子工作流](../../guides/operator-development-workflow.md)。

## 端口、输出与转换位置（P3 O0）

| 端口 | dtype | 逻辑 shape / 语义 |
|---|---|---|
| `k` / `v` | scalar F32 | `[seq_len, num_kv_heads * head_dim]` |
| `k_cache_in` / `v_cache_in` | 同一 scalar F16 或 BF16 | `[num_kv_heads, cache_len, head_dim]` |
| `k_cache_out` / `v_cache_out` | 跟随各自 cache input | 原 state spec 与原物理 plane 的 alias |

保留旧同型语义；新增 mixed tuple 只允许 A32→F16/BF16 state。转换由该端口 tuple 的语义定义，精度来自真实 TensorSpec，不在 OpParams 再复制 kv_dtype 字段，现有参数 serde 格式保持。不新增通用 Cast，不支持 K/V 异型、不同层 state 异型或 A16→另一种 KV16 的隐式转换。语义输出是 state alias，kernel 仍通过 invocation `kv_append` 写入 Manager-owned storage，不分配 TensorView output。

K 在既有 F32 RoPE 完成后窄化，V 从 F32 projection 输出窄化。每个新 cache 元素只在此边界舍入一次。Prefill 和 Decode 使用同一规则；后续 Attention 必须使用已窄化的 cache，包括当前 token，不得使用平行 F32 数据源。

## 转换数值合同

- F32→F16/BF16 使用固定 round-to-nearest-even、gradual underflow；保留 signed zero，overflow 到同号 Inf。
- NaN 保留 sign 与可容纳 payload，强制 quiet bit，不因低 payload 全被截断而变成 Inf。
- 转换只处理原始位模式，不依赖/修改调用方 rounding mode、FTZ/DAZ 或已有 exception flags。转换允许非有限值，不增加 payload 有限性预扫描。
- raw load/store 使用 byte-safe 方式，不要求 cache bytes 存在 Half/BFloat16 数组对象。scalar 和后续 SIMD 必须按独立 oracle 验证位结果。

既有数值权威为 [FP16 batch 合同](../fp16_conversion/README.md) 和 [BF16 batch 合同](../bf16_conversion/README.md)。此保证只属于转换边界，不扩展到 RoPE、Attention 或完整模型的浮点环境。

## Shape、stride、alias 与失败

activation 为 rank 2，两个 shape 相等；seq_len、head 数、head_dim 为正，hidden 等于 checked 的 `num_kv_heads * head_dim`。activation 使用既有合法正元素 stride，实际读地址满足 float 自然对齐。首批零序列仍拒绝，不用批量 converter 的空范围合同放宽有状态算子语义。

cache 的 token/head stride 按字节计，必须是实际两字节元素的整数倍，覆盖 logical head_dim、capacity 与 append window；offset/footprint/address 加法均 checked。生产基址遵守 layout alignment，不附加 float 对齐要求。只写 `[begin,end)` 的 logical elements，不触及 padding、已提交历史或窗口外数据。

两个只读 activation 可以共享 bytes；每个 activation 的读取范围必须与两个 cache 写入窗口都不重叠。K/V destinations 互不重叠；state in/out 仅允许对应 K→K、V→V 的原地 alias。范围证明沿用现有 strided footprint 的保守拒绝规则，不承诺所有稀疏交错布局均可接受。

准备阶段保存稳定 activation 信息及预期 state 精度；KV 指针、begin/end 与实际 layer binding 属单次调用，不能捕获到 prepared 参数。每次写入前校验 state specs、selector、KV view/layout、binding dtype 和 geometry 一致。

参数/本次 binding 校验失败发生在本次更新写入前。稍后 layer 或 Attention 失败时，未提交窗口可以已被改写；execution 不发布 watermark，不能读取该窗口作为跨 plan 历史，重试必须重新写完整范围。KVUpdate 不自行 commit。完整 plan、每层更新成功后才由 execution 调用 CommitAppend；generation、单 reservation 和外部串行化约束保持。

## 验收与现有证据

- 独立 oracle 验证转换位结果，包括 midpoint、邻点、subnormal、signed zero、NaN/Inf 和 overflow。
- 覆盖正 activation stride、tail、padding、写入窗口、合法共享只读输入、非法 source/destination overlap、错误 dtype 与 byte overflow。
- 两种 KV16 都覆盖 Prefill/Decode、多层成功、后续失败不发布、重试、capacity、generation/stale view 及 prepared 执行零分配。
- 真实资源按 KVCacheLayout 与 allocator 请求记账：两字节 logical elements 不等于任何 alignment 下 physical bytes 恰好减半。

当前源码见 [entry](../../../src/backend/cpu/kernels/kvcache_update/kvcache_update_entry.cpp)、[F32 reference](../../../src/backend/cpu/kernels/kvcache_update/kvcache_update_f32_reference.cpp)。已有事务回归见 [test_kvcache_update_kernel.cpp](../../../tests/unit/execution/test_kvcache_update_kernel.cpp)。[QHIHOGQ 机器记录](benchmarks/qhihogq-kv-cache-update.md) 是历史 F32 执行片段证据，不能解释为 P3 KV16 性能或数值验证。
