# KV Cache 与 CPU Attention 能力演进方案

- **状态**: Draft
- **版本**: 1.1
- **日期**: 2026-10-11
- **产品边界**: [AetherMind 当前产品 PRD](../products/aethermind_prd.md)；CPU-only、单请求、同步 Generate、静态 KV
- **架构边界**: [AGENTS.md](../../AGENTS.md) §2.1；`kv_cache` 拥有 KV 物理状态域，Runtime 按值聚合 Manager，execution 建立窄绑定，backend 执行 kernel
- **相关历史草案**: [05 号 KVCache Manager 演进方案](05-kv-cache-manager-evolution.md)（已被本提案取代；其接口草图不得视为当前实现）
- **算子工作流**: [算子开发与优化工作流](../guides/operator-development-workflow.md)

## 1. 结论与范围

当前应把三个不同能力分开决策：

| 能力 | 当前裁决 | 实施前提 |
|---|---|---|
| KV Cache ownership、epoch 与资源合同 | owner/epoch 与 commit 发布权已实现并有本轮 Release 聚焦回归；资源预算与 dtype 执行合同仍为 **P0 correctness** 工作 | 不改变单请求、连续预分配产品边界 |
| CPU Attention 的 I/O 与计算优化 | **P1 候选**；先建真实负载基线 | 完整 O0–O4 证据，保持 FP32 reference 独立 |
| Paged KV / PagedAttention | **长期条件项**；当前不实施 | 产品进入多请求或反复前缀复用，并测得容量/吞吐收益可覆盖 indirection 成本 |

这里的 PagedAttention 指非连续 KV block 的寻址与其配套 kernel；FlashAttention 指 exact Attention 的 I/O-aware 分块计算。二者分别改变**物理 KV 管理**和**Attention 算法/数据移动**，可以组合，但不互为前置条件。当前产品不需要引入第二套 Graph IR、在 operators 中表达 page/block id，或以 GPU FlashAttention 的结果推断 CPU 收益。

本提案记录已核对事实、历史缺陷、准入门槛和实施顺序。2026-10-11 的 owner/epoch 修复及本轮 commit 发布权收窄均已实现；资源预算与 dtype 执行合同、CPU Attention 优化仍未完成，**没有据此宣称性能收益**。

## 2. 现状分析与证据边界

### 2.1 已验证的当前实现

| 事实 | 代码/测试依据 | 边界 |
|---|---|---|
| `Runtime` 按值持有单个 `KVCacheManager`；Manager 属于 `kv_cache` 源模块，使用 K/V 两块连续 `Buffer` 和单个 `KVCacheSessionState`，Buffer 由 `Init` 收到的 `Allocator&` 分配 | [KV manager](../../include/aethermind/kv_cache/kv_cache_manager.h)、[实现](../../src/kv_cache/kv_cache_manager.cpp)、[分配入口](../../include/aethermind/memory/allocator.h) | 同一时刻只允许一个 reservation；Manager 只借用该 allocator 引用、不保存它，plane 存储与其余受管存储共用同一计账入口，没有 block table、prefix cache 或多请求调度 |
| `KVCacheSessionView` 借用 Manager 的 layout/storage/session state，`valid()` 检查 generation；Manager 在 Reset/Release 校验三个借用指针的身份 | [KV view](../../src/kv_cache/kv_cache_session_view.cpp)、[KV manager](../../src/kv_cache/kv_cache_manager.cpp) | view 可复制；Manager 必须长于所有 view、保持地址稳定，访问由调用方串行化 |
| `LayerRunner` 完整 plan 成功后提交 KV watermark；`KVCacheUpdate`/`Attention` 已有窄绑定与真实 tiny Llama 执行证据 | [LayerRunner](../../src/execution/layer_runner.cpp)、[M4/M5 测试](../../tests/unit/inference/test_direct_prefill_decode.cpp) | 这是正确性 baseline；不等于 KV 生命周期全部边界已经验证 |
| CPU KVCacheUpdate 与 Attention descriptor 只接受 FP32 KV storage | [KVCacheUpdate entry](../../src/backend/cpu/kernels/kvcache_update/kvcache_update_entry.cpp)、[Attention entry](../../src/backend/cpu/kernels/attention/attention_entry.cpp) | PRD 的 FP16/BF16 KV 图示尚无 production kernel 证据 |
| FP32 Attention reference 对每个 query/head 扫描 K 两遍，第二遍读 V；不物化完整 score 矩阵 | [reference 实现](../../src/backend/cpu/kernels/attention/attention_f32_reference.cpp) | CPU blocked/online 方案的收益仍需 benchmark；不可直接套用 GPU 结论 |

[tests/unit/kv_cache/test_kv_cache_manager.cpp](../../tests/unit/kv_cache/test_kv_cache_manager.cpp) 覆盖基本 reserve/reset/release、越界、cross-manager Reset/Release 拒绝、成功或失败 re-init 后旧 view 永不复活，以及未借用 view 时的 Manager 移动。下述旧行为保留为缺陷背景，当前实现与验证边界分别说明。

### 2.2 P0：re-init 后旧 view 可复活

旧实现中，[`KVCacheManager::Init`](../../src/kv_cache/kv_cache_manager.cpp) 将 `slot_` 赋为默认值，generation 重新从 0 开始；[`ReserveForSession`](../../src/kv_cache/kv_cache_manager.cpp) 随后将其增至 1。旧 view 保存的同一个 slot 地址与 generation 1 因此重新匹配，使其可访问新 KV storage。

2026-09-24 的临时复现结果为 `old_view_revalidated=1 fresh_view_valid=1`。2026-10-11 的修复在 Init 清空 reservation 时保留 generation，成功预约前检查上限并递增；Release 在上限仍完成清理，后续预约返回 `kOverflow`。Init 继续释放原 storage 与 active reservation，包括初始化失败的情形。自动回归覆盖 active view 副本、已释放副本，以及失败 re-init 后再初始化/预约；generation 耗尽分支仅作静态审查，未引入私有状态测试 hook。

### 2.3 P0：跨 Manager view 可操作错误 slot

旧实现的 [`ResetSession` 与 `ReleaseSession`](../../src/kv_cache/kv_cache_manager.cpp) 仅检查 `view.valid()`；A 的 view 可被 B 接受并错误修改 B 的 slot。历史复现为 `cross_release_ok=1 a_view_valid=0 b_view_valid=0`。2026-10-11 的修复先验证 view 的 layout/storage/slot 指针均属于当前 Manager，再验证 reservation 是否仍有效；foreign 或 stale view 被拒绝，两个 Manager 的 reservation、watermark 与 view 均保持原状态。Manager 禁止复制，移动仅允许在没有任何 borrowed view（包括 stale 副本）时进行；它不承诺 Manager 销毁或非法移动后访问 view 的安全性。

### 2.4 发布权与资源合同

旧的公开 `KVCacheSessionView::CommitUntil` 允许低层调用方绕过完整 layer 写入证明而提前发布 watermark。当前实现已移除该公开入口，提交收窄为 source-private `detail::KVCacheCommitAccess::CommitAppend`：仅 `LayerRunner` 在全部 layer 依序完成同一 append range 且整个 plan 成功后调用；该入口校验 epoch、append begin、range、首次完整 Prefill 与单 token Decode，失败不修改状态。本轮 Release 聚焦回归已覆盖该路径，证据见 §5.2。当前 Manager 的资源预算聚合与 KV dtype 可执行集合仍未闭环；不得把本次发布权收窄误报为这些问题已修复。

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
2. `Init`、release、下一次 reserve 或 Manager 资源替换后，旧 view 永远不再有效；epoch 不可回退或无声回绕。`ResetSession` 保留 reservation 与 epoch，因此既有 view 副本仍有效，并回到完整 Prefill 前的状态。
3. 只有完整计划证明所有必要 layer 写入成功，才能提高公开可读 watermark；失败后的未提交 range 不能被后续读取。
4. 模型 context limit、物理 KV capacity、requested reservation 和实际 committed tokens 分开记录，所有算术检查溢出。
5. 优化 descriptor 的 layout/dtype 合同必须与绑定一致；reference kernel 保持 correctness oracle。
6. 当前 Decode 稳态无堆分配；任何新资源计划或布局策略不得把分配/映射查找移入热路径。

## 4. 方案与备选

### 4.1 当前静态 KV：最小正确性修复

已落地方案保留 contiguous pool 和现有 `KVCacheSessionView` 边界，使用借用指针身份校验与跨 re-init 保留的 generation，没有新增 heap owner token 或热路径间接寻址。`Init` 保持“重新初始化释放旧 reservation”的公开语义。公开 view 不再有提交 API；source-private commit access 不安装、不提供公开 test API，测试 fixture 通过同一内部 access 路径构造事务。Manager/view 的 lifetime、地址稳定和外部串行化属于调用方合同；不扩展为并发资源仲裁。

当前同步合同要求同一 Manager 的 `Init`、`ReserveForSession`、`ResetSession` 与 `ReleaseSession` 不得插入一次 `Execute`，包括 kernel 重入；transaction 不得逃离执行栈。因此 Reset 不需要 revision/token 来使 view 失效。若未来需要重入或并发，必须另行设计同步与 transaction identity，不能把现有 generation 误称为覆盖所有 Reset ABA。

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
| **KV owner/epoch correctness** | `KVCacheManager`/`KVCacheSessionView` 的 owner、slot、epoch 校验；re-init 与回绕策略 | cross-manager Reset/Release 拒绝；re-init/re-reserve 后旧 view 永不复活；ASAN/TSAN 目标测试；既有 M4/M5 Generate 测试继续通过 | 实现与正式回归已加入；验证结果见下方记录 |
| **KV publish/resource contract** | commit 权威已收窄为 LayerRunner 的 source-private transaction access；汇总模型、activation、workspace、KV 预算及明确 KV dtype 可执行集合仍待完成 | Release 聚焦回归覆盖提交拒绝、失败不发布、诊断 OOM 与零分配；预算与实际分配字节一致、超限可恢复失败、真实模型配置下的容量/精度证据仍待设计 | 发布权已验证；资源/dtype 仍待设计 |
| **CPU Attention baseline 与候选** | 增加按 Prefill/Decode、context depth、GQA、KV dtype/layout 分组的 benchmark；候选在独立 optimized descriptor | O0/O1 数值、O2 registered/prepared 基线、O4 同机交错测量；production `ExecutionStep` 与端到端结果；误差、bytes、latency 同报 | 待测，P1 |
| **Paged KV 准入审查** | 需求、预算和多请求 workload 研究；获批后另设 block manager 与 paged kernel 专题 | 证明 contiguous 静态策略的真实资源损失，并同时测得 paged 方案收益足以覆盖额外 metadata/寻址成本 | 未触发，长期 |

每项应记录“已验证事实 / 机制推断 / 待测结果”。`verify_docs.py` 只证明文档结构与链接，不能替代 runtime 回归、数值或性能证据。

### 5.1 2026-10-11 owner/epoch 修复验证

- 当前工作树的 Debug 与 Release `aethermind_unit_tests` 均构建通过；聚焦 `CPUAllocator`、`Buffer`、`MemoryHandle`、KV、tiny Llama O1/plain 与 O2/packed、HF load/prepare、CPU weight packing，共 111/111 测试分别通过。包含新增 owner/re-init/诊断 OOM 回归、Generate OOM 清理与重试，以及 Decode 零分配检查。
- 焦点 sanitizer executable 直接编译受影响的 KV、CPU allocator、Session 源码及相关测试，链接 Debug 核心库；并非全库 sanitizer 验收。ASAN/UBSAN 的 32/32 测试通过，KV TSAN 的 19/19 测试通过；TSAN 使用 non-PIE executable 和 `setarch x86_64 -R` 避免本机随机地址映射启动失败。
- sanitizer 验证禁用 glibc malloc interposer，避免绕过 sanitizer allocator；故障注入用例在正常 Debug/Release target 验证。LSAN 受到当前执行环境 ptrace 限制，不能完成扫描；ASAN/UBSAN 运行使用 `detect_leaks=0`，不据此宣称泄漏验收通过。
- generation 耗尽策略已静态审查；未用私有状态 hook 构造回绕。Manager 必须长于全部 borrowed views、持有 view 时不可移动，所有请求仍由调用方串行化；上述验证不构成并发支持承诺。

### 5.2 本轮发布权收窄：Release 聚焦验证

本轮将公开 `KVCacheSessionView::CommitUntil` 替换为 execution-only 的 source-private commit access，并将 session lifecycle 收敛为 `Idle`、`AwaitingPrefill`、`PrefillCommitted`。Release build 已通过；聚焦 filter 的 128/128 测试通过，覆盖 `KVLayoutContract`、KV manager/session view、commit transaction、invalid reservations、`ExecutionContext`、KV update kernel、Runtime、workspace 与 tiny Llama O1/O2 路径；其中包含 4 项 diagnostic OOM 回归和 Decode 稳态零分配证据。

该结果支持提交入口的受控调用、首次完整 Prefill、单 token Decode、失败不发布和现有 epoch/lifetime 合同。source-private access 是 supported API 的权限收窄，不是不可伪造的安全边界；它也不覆盖不遵守当前同步合同的 Reset/重入。generation 的 `uint64_t` 饱和分支已静态核对，尚未通过黑盒状态构造覆盖。资源预算与 dtype 执行合同不在本轮范围内。

本轮还以 `cmake --build build-tsan --target AetherMind -j8` 成功构建 TSAN core，并将聚焦 non-PIE executable 链接到该 instrumented core。为避免与 sanitizer allocator 冲突，测试临时禁用了 `malloc_interposer.cpp` 的 glibc hooks；首次普通启动遇到 TSAN 的 `unexpected memory mapping`（66），按既有非 PIE 运行方法用 `setarch x86_64 -R` 重试，在 `TSAN_OPTIONS=halt_on_error=1` 下通过。filter 为 `KVLayoutContract.*:KVCacheManager.*:KVCacheSessionView.*:KVCacheCommitTest.*:InvalidReservations/KVCacheInvalidStateTest.*:KVCacheUpdateKernel.*`，结果为 64 ran、56 passed、8 skipped，且无 TSAN race 报告；8 个跳过项为依赖已禁用 hook 的故障注入用例，包含 4 个新的 commit OOM 用例，它们已由上述 Release 128/128 覆盖。临时 compile command 和 stub 仅作为本轮证据，不是仓库长期 artifact。该 TSAN 结果不构成并发支持验收。
- 本机以当前编译选项测得 `sizeof(KVCacheSessionState) == 40`，此前为 48 字节；这仅是对象布局变化，不构成推理性能收益证据。按值拥有该状态的 C++ 对象布局会受影响，调用方需重新编译。

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
| 2026-10-11 | 1.1 | 修复 owner/epoch 实现，加入正式回归并明确 Manager 的 move/lifetime/串行访问合同；publish/resource 与 Attention 优化仍保持待办 |
| 2026-09-24 | 1.0 | 以当前代码和临时复现证据重新划分 KV correctness、CPU Attention 优化与 Paged KV 长期准入；取代过期的 05 号当前状态叙述，未修改代码 |
