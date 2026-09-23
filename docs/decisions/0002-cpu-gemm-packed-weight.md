# ADR-0002: CPU GEMM 的 descriptor-owned packed-weight recipe

- **状态**: Accepted
- **日期**: 2026-09-23
- **作者**: AetherMind contributors
- **关联代码**: `KernelDescriptor::packing_recipe`、`Backend::GetPackingRecipe`、`PrepareExecutableModel`、`CpuWeightPrepacker`、CPU GEMM packed-B drivers

## 背景

CPU packed weight 需要让 kernel selection、模型准备、packing、artifact store 和 prepared execution 对物理字节解释完全一致。按 selector 在 prepacker 内自行推导 recipe 无法区分同一 selector 下的不同 kernel layout；在 request 生成时查询 backend 又违反 compiler 对 backend 的依赖边界。

GEMM B-panel layout 另外需要固定 `NR/KC` 和 padded byte size。`KC` 会改变 physical format 与放大率，未经同机 Decode/Prefill、hot/streaming 对照不能作为生产性能决策。

## 决策

1. `KernelDescriptor` 拥有 packed descriptor 的精确 `PackingRecipe`；plain descriptor 不得声明 recipe。recipe 继续只含 `{layout, alignment}`，shape 和尺寸从 artifact logical metadata 计算。
2. backend 的 recipe query 与 kernel prepare 共用 descriptor eligibility、CPU feature 过滤和 priority resolver。Inference 在获得 backend 后把 recipe 注入每个 packing request；共享权重只有在 consumer op、binding、selector 和 recipe 一致时才 coalesce，否则准备期明确失败。
3. packing service 接收显式 recipe，artifact、`WeightArtifactKey` 与 `ResolvedKernel.expected_packing_recipe` 必须相等。`CpuBackend::PackWeights(..., recipe)` 还会校验 recipe 等于同一 backend/feature policy 当前解析出的 descriptor recipe；旧三参数 overload 也按当前 descriptor 选择后委托。隔离 candidate 测试/benchmark 直接调用 prepacker primitive，不伪装成 global backend 选择结果。CPU identity recipe 和已有 plain 路径保留。
4. 实现 `cpu_bpanel_f32_v1_avx2_kc512_candidate` 作为可独立验证的物理布局候选：`NR=16`、`KC=512`、64-byte 对齐，存储顺序为 `[K-panel][N-block][KC][NR]`，K/N 尾部补零，artifact 字节数严格为 `ceil(N/NR) * ceil(K/KC) * KC * NR * sizeof(float)`。不对齐的 QKV/GateUp component slice 使用 packed scan/reference driver。
5. 当前 identity descriptor 与 bpanel descriptor 保持相同 priority，identity 先注册，因此常规全局 dispatch 继续使用 identity。只有完成目标机器的噪声 floor、KC 候选比较、prepared Decode/Prefill 和 break-even 门禁后，才讨论提升 bpanel priority。benchmark 的短 setup break-even 数值仅用于诊断。

## 权衡

| 备选方案 | 结论 | 原因 |
|---|---|---|
| 在 compiler 或 ModelLoader 内查询 kernel/recipe | 拒绝 | 会让上层编译/加载流程依赖 backend，破坏模块边界 |
| `CpuWeightPrepacker::RecipeFor(selector)` 作为多 layout 权威 | 拒绝 | descriptor selection 与 packing policy 会漂移，无法安全表示多个物理布局 |
| 在 request 生成时按首个 consumer 去重 | 拒绝 | 可能丢弃共享权重的 recipe/op 冲突，造成 plan 错误或错误字节解释 |
| descriptor-owned recipe + inference 注入 + exact-key store | 采用 | selection 与 artifact identity 共用单一合同，冲突能在准备期报告 |
| 立即让 KC512 bpanel 替代 identity | 暂缓 | 当前 WSL2 小样本存在较大进程方差，且没有 KC256 对照；性能门禁未完成 |

## 后果

- 全局 graph-wide packed lowering 已有 Linear、RmsNorm、Embedding、QkvLinear、GateUpLinear 和 AddRmsNorm 的 identity 消费者，可准备 tiny Llama packed artifact。
- Linear/QkvLinear/GateUpLinear 的 AVX2 bpanel candidate 已实现，并可通过隔离 registry 测试；默认 production recipe 仍为 identity。
- packed-B correctness、padding、tail、alias、zero-size 和 fused component slice 有 focused tests。机器级 performance status 仍为 **Needs More Data**；本机诊断与 raw JSON 记在 `docs/operators/gemm/benchmarks/54h5mmi-gemm.md` 和 gitignored benchmark-results 目录。
- `KC=512` 只标识候选 layout，不代表已完成调优或性能接受。

## 关联

- [CPU GEMM 优化方案](../operators/gemm/cpu-gemm-optimization.md)
- [CPU GEMM Packed Weight 提案](../operators/gemm/cpu-gemm-packed-weight.md)
- [算子开发与优化工作流](../guides/operator-development-workflow.md)
