# KV Cache 与 CPU Attention 能力演进方案

- **状态**: Draft
- **版本**: 1.0
- **日期**: 2026-09-24
- **产品边界**: [AetherMind 当前产品 PRD](../products/aethermind_prd.md)；CPU-only、单请求、同步 Generate、静态 KV
- **架构边界**: [AGENTS.md](../../AGENTS.md) §2.1；runtime 拥有 KV 物理资源，execution 建立窄绑定，backend 执行 kernel
- **相关历史草案**: [05 号 KVCache Manager 演进方案](05-kv-cache-manager-evolution.md)（已被本提案取代；其接口草图不得视为当前实现）
- **算子工作流**: [算子开发与优化工作流](../guides/operator-development-workflow.md)

## 1. 结论与范围

当前应把三个不同能力分开决策：

| 能力 | 当前裁决 | 实施前提 |
|---|---|---|
| KV Cache ownership、epoch 与资源合同 | **P0 correctness**；先修现有静态实现 | 不改变单请求、连续预分配产品边界 |
| CPU Attention 的 I/O 与计算优化 | **P1 候选**；先建真实负载基线 | 完整 O0–O4 证据，保持 FP32 reference 独立 |
| Paged KV / PagedAttention | **长期条件项**；当前不实施 | 产品进入多请求或反复前缀复用，并测得容量/吞吐收益可覆盖 indirection 成本 |

这里的 PagedAttention 指非连续 KV block 的寻址与其配套 kernel；FlashAttention 指 exact Attention 的 I/O-aware 分块计算。二者分别改变**物理 KV 管理**和**Attention 算法/数据移动**，可以组合，但不互为前置条件。当前产品不需要引入第二套 Graph IR、在 operators 中表达 page/block id，或以 GPU FlashAttention 的结果推断 CPU 收益。

本提案记录已核对事实、可复现缺陷、准入门槛和实施顺序；**不表示代码已修复或性能收益已测得**。

## 2. 现状分析与证据边界

### 2.1 已验证的当前实现

| 事实 | 代码/测试依据 | 边界 |
|---|---|---|
| `Runtime` 持有单个 `KVCacheManager`；Manager 使用 K/V 两块连续 `Buffer` 和单个 `SessionKVSlot` | [KV manager](../../include/aethermind/runtime/kv_cache_manager.h)、[实现](../../src/runtime/kv_cache_manager.cpp) | 同一时刻只允许一个 reservation；没有 block table、prefix cache 或多请求调度 |
| `KVCacheView` 借用 Manager 的 layout/storage/slot，`valid()` 检查 slot generation | [KV view](../../src/runtime/kv_cache_view.cpp) | view 可复制；generation/owner 目前不能充分证明属于当前 reservation |
| `LayerRunner` 完整 plan 成功后提交 KV watermark；`KVCacheUpdate`/`Attention` 已有窄绑定与真实 tiny Llama 执行证据 | [LayerRunner](../../src/execution/layer_runner.cpp)、[M4/M5 测试](../../tests/unit/inference/test_direct_prefill_decode.cpp) | 这是正确性 baseline；不等于 KV 生命周期全部边界已经验证 |
| CPU KVCacheUpdate 与 Attention descriptor 只接受 FP32 KV storage | [KVCacheUpdate entry](../../src/backend/cpu/kernels/kvcache_update/kvcache_update_entry.cpp)、[Attention entry](../../src/backend/cpu/kernels/attention/attention_entry.cpp) | PRD 的 FP16/BF16 KV 图示尚无 production kernel 证据 |
| FP32 Attention reference 对每个 query/head 扫描 K 两遍，第二遍读 V；不物化完整 score 矩阵 | [reference 实现](../../src/backend/cpu/kernels/attention/attention_f32_reference.cpp) | CPU blocked/online 方案的收益仍需 benchmark；不可直接套用 GPU 结论 |

`tests/unit/execution/test_kv_cache_manager.cpp` 覆盖基本 reserve/reset/release 与越界；本轮未发现 re-init 后旧 view 或 cross-manager Reset/Release 的回归用例。本文的缺陷判断以下述代码路径及一次临时最小程序为依据，**尚未形成仓库内自动回归测试**。

### 2.2 P0：re-init 后旧 view 可复活

[`KVCacheManager::Init`](../../src/runtime/kv_cache_manager.cpp) 将 `slot_` 赋为默认值，generation 重新从 0 开始；[`ReserveForSession`](../../src/runtime/kv_cache_manager.cpp) 随后将其增至 1。旧 view 保存的仍是同一个 Manager 成员 `slot_` 地址和第一次 reservation 的 generation 1，因此 `Init → 再次 Reserve` 后，旧 view 可通过 `valid()` 并指向**新的** KV storage。`Init` 的公开合同允许重新初始化，故不能把调用方不这样做当作修复。

2026-09-24 在 `41ce89fa` + 未提交 M5 工作树构建的 `libAetherMind.so` 上运行仓库外临时复现程序，结果为 `old_view_revalidated=1 fresh_view_valid=1`。该结果证明当前构建中的行为；复现程序位于临时目录，后续实施必须将场景纳入正式回归测试。

### 2.3 P0：跨 Manager view 可操作错误 slot

[`ResetSession` 与 `ReleaseSession`](../../src/runtime/kv_cache_manager.cpp) 仅检查传入 `view.valid()`，未验证 view 的 owner/slot 是 `this`。以 Manager A 的有效 view 调用 Manager B 的 `ReleaseSession`，B 会清空自己的 slot 并使 A 的这个 view 失效；A 的 slot 仍处于 in-use。临时程序实测 `cross_release_ok=1 a_view_valid=0 b_view_valid=0`，这不是允许的 Session 释放语义。

### 2.4 发布权与资源合同

[`KVCacheView::CommitUntil`](../../src/runtime/kv_cache_view.cpp) 是公开成员，只验证范围/阶段，不验证对应 layer 的 KV 已写完。production `LayerRunner` 已按完整 plan 成功后提交，但低层调用方仍可提前发布未写入的 KV；应在保持必要测试能力的同时收窄提交权威。当前 Manager 还在 `Init` 时直接分配完整物理容量并仅报告 `total_bytes()`；未与模型权重、activation、workspace 汇成可在分配前检查的资源预算。PRD 的 FP16/BF16 KV 表述与当前 FP32 kernel 能力应按产品要求重新裁决，不能把可配置 dtype 当作可执行支持。

## 3. 目标职责与不变量

```text
InferenceSession     拥有一次请求的 reservation 生命周期与失败清理
        ↓
Runtime/KV manager   验证 owner、slot、epoch；规划/持有物理 KV 与预算
        ↓
Execution            将当前 append/read range 转成窄 KV binding；完整 plan 成功后发布
        ↓
CPU backend          只消费已解析的 geometry、layout 与数据指针
```

必须满足：

1. 一个 view 只能由创建它的 Manager 操作；`Reset/Release` 还须确认它对应当前 slot 与 epoch。
2. `Init`、release、下一次 reserve 或 Manager 资源替换后，旧 view 永远不再有效；epoch 不可回退或无声回绕。
3. 只有完整计划证明所有必要 layer 写入成功，才能提高公开可读 watermark；失败后的未提交 range 不能被后续读取。
4. 模型 context limit、物理 KV capacity、requested reservation 和实际 committed tokens 分开记录，所有算术检查溢出。
5. 优化 descriptor 的 layout/dtype 合同必须与绑定一致；reference kernel 保持 correctness oracle。
6. 当前 Decode 稳态无堆分配；任何新资源计划或布局策略不得把分配/映射查找移入热路径。

## 4. 方案与备选

### 4.1 当前静态 KV：最小正确性修复

推荐保留 contiguous pool 和现有 `KVCacheView` 调用边界，先补**Manager owner/slot 验证**与**跨 re-init 单调 epoch**。实现可比较稳定 slot identity，或引入受控 owner token；选型时必须覆盖 Manager 移动/销毁合同及 epoch 溢出。`Init` 继续满足现有“重新初始化释放旧 reservation”的公开语义，但必须先使全部旧 view 永久失效；若选择改变该语义为拒绝 active re-init，应作为独立 API 决策并更新调用方。

`CommitUntil` 的收窄可采用只向 execution transaction 暴露的提交入口，或由 Manager 验证一次 append 的完整性；在变更前先清点手工 plan/测试调用者，不让测试便利接口成为 production 发布后门。

### 4.2 CPU Attention：按负载选择算法

当前 reference 已避免完整 score 矩阵，故原始 FlashAttention 的主要“消除 score 矩阵写回”收益不能原样归因到本项目。[FlashAttention 论文](https://arxiv.org/abs/2205.14135)讨论的是 GPU HBM/SRAM I/O-aware exact attention；CPU 应以 cache、带宽、SIMD 和线程拓扑为实际约束。

- **Prefill（query 长度 > 1）**：候选是 Q/K/V 分块、跨 query 复用 KV、online softmax 与受控工作区；比较当前两遍扫描 reference 的内存流量和真实 `Executor` 路径。
- **Decode（query 长度 1）**：候选是单遍 online softmax、KV 顺序访问/布局、GQA 下的 K/V 复用与 SIMD；没有多 query tile 可复用，不应默认采用 Prefill driver。
- 两类方案保持 causal、GQA、数值稳定性、tail/stride、alias 和 FP32 reference 对拍；选择阈值必须由目标硬件的 production-path 证据决定。

非平凡 kernel 方案在进入实现前，按 [算子开发与优化工作流](../guides/operator-development-workflow.md) 写入 `docs/operators/attention/` 的专项提案及机器级 benchmark 记录。本文件只冻结能力边界、候选机制和准入条件，不预设 tile/block 大小或加速倍数。

### 4.3 Paged KV：条件触发的并列 storage policy

PagedAttention 的目标是让 Attention 消费非连续物理 KV blocks；它与 block pool、logical-to-physical table、refcount/COW、prefix identity、eviction 和请求调度共同构成多请求资源方案。[PagedAttention 原论文](https://doi.org/10.1145/3600006.3613165)、[vLLM prefix cache 设计](https://docs.vllm.ai/en/latest/design/prefix_caching/)。仅替换现有 `Offset()` 或增加一个 kernel 名称不能获得这些能力。

保持当前 contiguous baseline。只有产品明确引入多 active requests、跨请求 prefix reuse 或实测静态预留浪费不可接受时，才独立立项 Paged KV；立项材料必须含代表性 prompt/输出分布、并发度、峰值 KV bytes、复用率、碎片/空置比例，以及 block indirection 相对 contiguous baseline 的成本。调度、block ownership 和 eviction 属 runtime/serving 资源层；Graph IR 和 OperatorSchema 不含 page/block id。进入实现后按 concrete layout 选择独立 paged Attention binding/kernel，不能强迫 contiguous reference 热路径支付 page translation 成本。

## 5. 实施步骤与退出证据

| 工作包 | 最小改动范围 | 必须通过的证据 | 状态 |
|---|---|---|---|
| **KV owner/epoch correctness** | `KVCacheManager`/`KVCacheView` 的 owner、slot、epoch 校验；re-init 与回绕策略 | cross-manager Reset/Release 拒绝；re-init/re-reserve 后旧 view 永不复活；ASAN/TSAN 目标测试；既有 M4/M5 Generate 测试继续通过 | 待实施，P0 |
| **KV publish/resource contract** | 收窄 commit 权威；汇总模型、activation、workspace、KV 预算；明确 KV dtype 可执行集合 | plan 中途失败不推进 watermark；提前 commit 拒绝；预算与实际分配字节一致；超限可恢复失败；真实模型配置下的容量/精度证据 | 待设计，P0/P1 |
| **CPU Attention baseline 与候选** | 增加按 Prefill/Decode、context depth、GQA、KV dtype/layout 分组的 benchmark；候选在独立 optimized descriptor | O0/O1 数值、O2 registered/prepared 基线、O4 同机交错测量；production `ExecutionStep` 与端到端结果；误差、bytes、latency 同报 | 待测，P1 |
| **Paged KV 准入审查** | 需求、预算和多请求 workload 研究；获批后另设 block manager 与 paged kernel 专题 | 证明 contiguous 静态策略的真实资源损失，并同时测得 paged 方案收益足以覆盖额外 metadata/寻址成本 | 未触发，长期 |

每项应记录“已验证事实 / 机制推断 / 待测结果”。`verify_docs.py` 只证明文档结构与链接，不能替代 runtime 回归、数值或性能证据。

## 6. 风险与依赖

| 风险 | 应对 |
|---|---|
| 仅修 generation、不校验 owner | 跨 Manager 操作仍可破坏两个 slot；两类回归必须同时过门 |
| epoch 计数回绕或 Manager move 后 identity 不稳定 | 冷路径显式拒绝回绕，定义 move/lifetime 合同并覆盖测试 |
| 将 FP16/BF16 可配置误报为 kernel 支持 | 以真实 `KVCacheUpdate → Attention → Generate` 数值链和内存证据决定支持状态 |
| online softmax/分块改变舍入顺序 | reference 保持独立，用绝对/相对误差与长上下文极值测试验收 |
| 用 Attention microbenchmark 宣称产品速度提升 | 必须同时测 production step 与端到端 Prefill/Decode，记录 Release flags、机器和重复运行 |
| 过早引入 Paged KV | 先验证产品范围与负载，不让 scheduler/block table 泄漏到当前语义 IR |

## 7. 关联资料与变更记录

- [当前产品 PRD](../products/aethermind_prd.md)：当前产品范围与 KV 内存目标。
- [M4/M5 执行闭环](01-inference-session-generate-readiness.md)：真实 CPU FP32 Prefill→Decode 与同步 Generate 的已实现证据。
- [05 号历史草案](05-kv-cache-manager-evolution.md)：存档的 lease/transaction/paged 接口草图；其中 2026-09-16 的“当前状态”已过期。
- [FlashAttention 原论文](https://arxiv.org/abs/2205.14135)、[PagedAttention 原论文](https://doi.org/10.1145/3600006.3613165)：机制来源，不能直接证明 AetherMind 的 CPU 收益。

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-09-24 | 1.0 | 以当前代码和临时复现证据重新划分 KV correctness、CPU Attention 优化与 Paged KV 长期准入；取代过期的 05 号当前状态叙述，未修改代码 |
