# AetherMind 系统能力演进路线图

- **状态**: Draft
- **版本**: 1.1
- **日期**: 2026-09-16
- **产品边界**: [AetherMind 当前产品 PRD](../products/aethermind_prd.md)
- **架构基线**: [架构总览](../designs/architecture/architecture_overview.md)
- **工程质量基线**: [工程质量体系建设方案](02-engineering-quality-system.md)
- **定位**: 全仓库 capability gap 总表与实施排序；专题细节由 01–05 号提案承载

## 1. 结论

AetherMind 当前不需要再次进行顶层架构重写。`model → graph/operators → compiler → execution → runtime/backend` 的责任方向已经基本正确，真正阻碍当前产品交付的是若干跨模块 capability chain 尚未闭环。

推荐按以下顺序演进：

1. **先修 correctness contract 漂移**：模型 activation、ElementwiseMul broadcast、KV commit/lifetime；
2. **闭环可执行 FP32 baseline**：ExecutableModel preparation、KV state binding、KVCacheUpdate、Attention、Prefill→Decode、同步 Generate；
3. **完善 physical planning**：shape/layout-aware kernel preparation、activation liveness、workspace reuse、exact packed recipe；
4. **建设 INT8/INT4 weight-only 生产链**：格式解析、量化语义、packing、kernel、精度和端到端验证；
5. **再做系统性能演进**：GEMM/Attention、线程池与 topology、NUMA、AArch64；
6. **最后冻结 public API/ABI 与发布门禁**：C++ Session、C ABI、错误模型、版本与兼容测试。

本路线图将模块分为三类：

- **交付阻塞模块**：不演进就无法完成当前产品；
- **性能/资源提升模块**：正确性 baseline 后推进；
- **当前无需架构演进模块**：保持现有边界，只做问题驱动的维护。

## 2. 盘点方法与事实边界

本路线图基于 2026-09-17 当前源码、测试、PRD 和现有 01–05 号提案。状态含义：

| 状态 | 含义 |
|---|---|
| 已闭环 | production/source path 已存在，且有与该能力相称的验证 |
| 部分闭环 | 核心对象或局部测试存在，但缺少完整数据流、数值或生命周期证据 |
| 未闭环 | PRD/架构目标存在，但生产入口或关键实现缺失 |
| 长期方向 | 当前产品不承诺，不进入近期实现范围 |

以下情况不能被当作 capability 已闭环：

- operator inference 测试通过，但没有 backend kernel；
- kernel 单测通过，但没有进入 `ExecutionPlan → PreparedExecutionBindings → Executor`；
- graph/lowering 成功，但 production plan 无法 resolve 全部 kernel；
- 基础设施测试通过，但没有 tiny Llama Prefill→Decode 数值验证；
- microbenchmark 有提升，但没有生产 shape、实际 build flags 和 end-to-end 证据。

## 3. 当前模块能力总表

| 模块 | 当前状态 | 主要缺口 | 优先级 | 对应提案/章节 |
|---|---|---|---:|---|
| public C++ / C API | 未闭环 | `Session::Generate`、`am_session_*`、ABI/version/error ownership | P0 | [01](01-inference-session-generate-readiness.md)、§15 |
| model loader / HF adapter | 部分闭环 | 支持范围真值、activation 漂移、量化格式、模型 fingerprint | P0/P1 | §5 |
| operators | 部分闭环 | semantic/capability matrix 漂移（KV/Attention 契约已闭环） | P0 | §6 |
| graph | 基础已闭环 | pass outcome 验证、contract drift；无需 target-aware 重写 | P1 | §7 |
| compiler/lowering | 基础已闭环 | 1:1 step 限制；仅在真实 1→N 需求出现后引入 ImplementationPlan | P1/P2 | §8 |
| execution planning | 部分闭环 | ExecutableModel preparation、shape-aware prepare、activation liveness | P0/P1 | [01](01-inference-session-generate-readiness.md)、§9 |
| runtime | 部分闭环 | KV transaction、resource budget、线程/topology、metrics | P0/P1 | [05](05-kv-cache-manager-evolution.md)、§10 |
| backend dispatch | 基础已闭环 | prepare request 缺 concrete shape/layout；packing service 边界 | P1 | §11 |
| CPU kernels | 部分闭环 | reference 主链已齐备（6/6）；量化和优化覆盖不足 | P0/P1 | [01](01-inference-session-generate-readiness.md)、[04](04-cpu-gemm-optimization.md)、§12 |
| memory/allocator | 基础已闭环 | activation/workspace/KV provider 统一、budget/NUMA policy | P1 | §13 |
| shape inference | 基础已闭环 | specialization 诊断与 runtime constraint 证据；不需通用动态 shape engine | P1 | §14 |
| base/dtypes/container | 当前足够 | 问题驱动维护；不作为推理主链重构目标 | P2 | §14 |
| tests/benchmark/quality | 部分闭环 | vertical slice、allocation/determinism、真实模型基线 | P0/P1 | [02](02-engineering-quality-system.md)、§16 |
| documentation | 治理中 | 存量漂移、PRD/实现矛盾、状态/命名迁移 | P1 | [03](03-documentation-stabilization.md)、§17 |

## 4. 跨模块目标数据流

当前缺少的不是另一层 semantic IR，而是从 compiler artifact 到可重复执行 Session 的 production preparation：

```text
HF directory
  -> ModelLoader
  -> LoadedModel
  -> ModelCompiler
  -> LoweredModelArtifact
  -> PrepareExecutableModel(Runtime, artifact, options)
       -> resolve kernels with concrete-enough capability request
       -> build graph-driven packing requests
       -> materialize exact packed artifacts
       -> build Prefill/Decode ExecutionPlan(s)
       -> plan activation/workspace/KV memory
  -> ExecutableModel
  -> CreateSession
       -> KVCacheLease + workspace/activation specialization
       -> Prefill
       -> Decode loop
  -> Token IDs
```

### 4.1 `ExecutableModel` 的推荐所有权

`ExecutableModel` 应作为 model/compiler artifact 与 Session 之间的 production-ready 边界，拥有或保持：

- `LoweredModelArtifact` 或保证其 raw weight backing 生命周期；
- graph-driven `PackedWeightStore`；
- 一个共享 `kBoth` plan，或经证据证明需要的 Prefill/Decode plans；
- plan 对应的 memory requirements；
- model identity/fingerprint 与 KV spec；
- 自动 external-weight binding metadata。

它不应拥有：

- 单 Session KV position；
- request-specific token buffer；
- mutable sampling state；
- wide `Runtime` 指针下传到 kernel；
- request scheduler 或 batch queue。

是否拆分 Prefill/Decode plan 由以下证据决定：kernel、workspace、topology、state/resource、shape specialization 或 layout/packing 确实不同。仅因为控制流名称不同，不足以复制两套 plan。

## 5. Model / HF Adapter 演进

### 5.1 当前已闭环

- HF directory、config、Safetensors index/file 读取；
- weight-name validation 与 logical weight resolution；
- `LoadedModel` 只拥有 config + resolved raw weights/backing；
- `ModelGraphBuilder` 是 HF → semantic graph 唯一转换权威；
- RoPE HF 配置已规范化到 typed `RoPEAlgorithmParams`；
- ModelLoader 不构图、不 resolve kernel、不 prepack。

### 5.2 P0：统一“接受的模型”和“实际语义”

当前 `HfModelValidator` 接受 `hidden_act = silu/gelu/relu`，而 `ModelGraphBuilder::BuildLlamaDense` 的 MLP 固定生成 `SiluMul`。这会把接受的 GELU/ReLU 配置静默编译成错误语义。

推荐裁决：

- 当前产品只承诺 SwiGLU 时，validator 只接受与 Llama SwiGLU 对应的 `silu`；
- 如果未来支持 GELU/ReLU，先增加对应 typed operator/graph path/kernel，再扩大 validator；
- 每个 accepted config variant 必须有 `config → semantic graph → execution` vertical test；
- parser 能读出字段不等于产品支持该字段。

关联代码：

- [`HfModelValidator`](../../src/model/formats/hf/hf_model_validator.cpp)
- [`ModelGraphBuilder`](../../src/model/model_graph_builder.cpp)
- [`HfModelConfig`](../../include/aethermind/model/formats/hf/hf_model_config.h)

### 5.3 P0/P1：INT8/INT4 weight-only 模型链

PRD 承诺 INT8 per-channel 和 INT4 group-wise weight-only，但默认 validation 仍拒绝 quantized tensors，当前 CPU registry 也没有对应 production kernel。量化不能只通过在 `KernelSelector` 增加 dtype 完成，必须闭环：

```text
HF quantization config/tensor schema
  -> validated QuantizationSpec
  -> graph/lowered value metadata
  -> exact packing request
  -> scale/zero-point ownership
  -> descriptor-specific packed artifact
  -> INT8/INT4 kernel
  -> accuracy + memory + throughput evidence
```

必须先冻结：

- 支持的外部格式和版本；
- weight dtype、scale dtype、group axis/group size；
- symmetric/asymmetric 与 zero-point policy；
- fused QKV/Gate-Up 不同分量的 scale 排布；
- tied embedding/lm_head 量化一致性；
- logical quantization identity 如何进入 `WeightArtifactKey/PackingRecipe`；
- reference dequantize path与 optimized integer path 的误差门禁。

量化细节在真正实施前应从本路线图拆出独立提案；在合同未冻结前，[CPU GEMM 优化方案](04-cpu-gemm-optimization.md) 只负责预留 driver/packing 边界，不应猜测格式。

### 5.4 P1：模型身份与可复现性

为 packed artifact、KV prefix（长期）和诊断建立稳定 model identity：

- architecture/config digest；
- weight file/index identity；
- tensor schema/version；
- quantization scheme；
- adapter identity（当前为 none）；
- packing recipe version。

当前产品不需要跨进程 artifact cache，但同一进程内必须能证明 packed store、execution plan 和原始 model artifact 属于同一模型实例。

### 5.5 当前不做

- tokenizer；
- MoE、encoder-decoder、sliding-window attention；
- LoRA/adapters；
- 在线模型热更新；
- 多模型 serving cache。

## 6. Operators 演进

### 6.1 建立单一 capability matrix

每个 `OpType` 至少记录：

| 维度 | 示例 |
|---|---|
| semantic dtype | FP32/FP16/BF16/I32… |
| shape/rank | static/symbolic/broadcast |
| alias policy | exact in-place/disjoint/unsupported |
| layout | contiguous/collapsible/strided |
| backend coverage | CPU reference/optimized |
| phase | Prefill/Decode/Both |
| weight format | plain/packed/quantized |

Operator inference 声明的支持范围可以宽于某个 backend，但必须在 capability matrix 中明确“semantic supported / CPU unsupported”，并在 planning 阶段给出可诊断的失败，而不是把 inference test 当作 execution support。

### 6.2 P0：修复 GraphOpBuilder 与 operator semantic 漂移

当前 `InferElementwiseMul` 和 CPU kernel 支持 broadcast，而 `AddElementwiseMul` 要求完整 `TensorSpec` 相等。推荐由 operator inference 成为 shape 语义权威：GraphOpBuilder 不重复实现更窄的 shape 规则；若产品有意禁止 graph-level broadcast，则应同步收窄 operator contract，而不是三层各自定义。

关联代码：

- [`graph_op_builder.cpp`](../../src/graph/graph_op_builder.cpp)
- [`elementwise_mul_op.cpp`](../../src/operators/ops/elementwise_mul_op.cpp)
- [CPU ElementwiseMul](../../src/backend/cpu/kernels/elementwise_mul/elementwise_mul_entry.cpp)

### 6.3 P0：Attention / KVCacheUpdate execution contract

**状态（2026-09-17）**：下列冻结项已随 reference chain 落地（`cpu::kvcache_update_f32_reference` / `cpu::attention_f32_reference` + `KVCache*Binding` 窄契约 + `KVCacheUpdateKernel.*`/`CPUKernelAttention.*` 测试）。

需要冻结：

- Q/K/V、KV cache state、mask/position 的 port 语义；
- GQA/MQA head mapping；
- Prefill 与 Decode 的 logical range；
- KV append transaction 内 pending visibility；
- supported dtype/layout；
- causal mask 与 sequence length；
- output alias/injective mapping；
- reference numerics 与 softmax stability。

state resource identity 与物理 binding 归 execution/runtime；operator 只定义语义，不出现 block table、page id 或 Manager 指针。

### 6.4 P1：消除重复校验权威

- `Infer*`：参数、dtype、rank、shape 和 deferred constraints；
- `PrepareExecutionBindings`/params builder：concrete stride、alignment、alias、storage bounds；
- kernel：只保留 cheap defense-in-depth 与 compute；
- GraphOpBuilder：只负责 port/value wiring，不复制 per-op inference。

## 7. Graph 演进

### 7.1 保持 semantic graph 边界

当前 Graph IR、GraphRewrite、GraphPassManager、ConstantFolding、DCE 和 fused semantic passes 已形成合理边界。近期不应引入：

- backend/ISA/layout/workspace 字段；
- Paged KV block/page；
- kernel sequence；
- request phase scheduler；
- packed pointer。

### 7.2 P1：pass contract 与 pipeline outcome

需要提升：

- 每个 pass 的 precondition/postcondition 与 preserved invariants；
- ambiguous multi-consumer pattern 的保守拒绝；
- fusion 后 output/consumer/residual/weight binding 完整验证；
- O0/O1/O2 的 deterministic pipeline composition；
- optimized graph dump 与 before/after structural diff；
- 每个 O-level 至少一个 production planning test，证明 pass 结果可被目标 backend 执行。

### 7.3 不做 cost-based optimizer

当前模型和 backend 单一，尚无统计信息、layout propagation 或多个等价 physical plan。不要为了“像编译器”而引入通用 cost model。只有出现两个以上真实可执行候选并能测量选择收益时再立项。

## 8. Compiler / Lowering 演进

### 8.1 保持当前 artifact 定位

当前 `LoweredGraph` 是 immutable、structurally validated compiler artifact，保留 semantic `OpType + OpParams`、dense value metadata、runtime checks 和 state aliases。它不是 Primitive IR，也不需要仅为命名完整而重写。

### 8.2 P1：`ImplementationPlan`

当第一个真实算子需要 1→N physical lowering 时，先在 execution/backend planning 边界引入局部：

```cpp
using ImplementationPlan = std::variant<SingleKernel, KernelSequence>;
```

候选场景包括：

- 一个 semantic Attention 降为多个 backend primitives；
- quantized Linear 需要 explicit dequant/compute/requant sequence；
- layout conversion 无法被 kernel 内部吸收且可复用。

在没有真实 sequence、workspace/lifetime/alias 合同和至少一个 production consumer 前，不提前实现。

### 8.3 Primitive IR 的进入门槛

只有同时满足以下条件才另立 RFC：

- 稳定且重复出现的 1→N lowering；
- 至少两个 backend 共享 primitive contract；
- 需要跨 node 的 layout/alias/resource analysis；
- Primitive IR 能减少而非复制 `OpType + OpParams` 语义。

KV Resource IR 只在真实 Paged KV/page table/opaque mutable resource 进入产品范围后考虑。

## 9. Execution Planning 演进

### 9.1 P0：ExecutableModel preparation

详细门禁见 [InferenceSession / Generate 前置闭环计划](01-inference-session-generate-readiness.md)。核心要求：

- `LoweredModelArtifact → ExecutableModel` 唯一 production preparation 入口；
- 自动构造 raw/packed external weight bindings；
- graph-driven materialization exact artifact；
- plan、packed store、raw backing 的 ownership 明确；
- preparation 失败不留下半成品 Session；
- public Generate 只接收 production-ready `ExecutableModel`。

### 9.2 P1：shape/layout-aware kernel prepare

当前 `Backend::PrepareKernel(OpType, KernelSelector, OpParams)` 看不到 concrete shape/stride/alignment，因此不能可靠完成：

- Decode `M=1` 与 Prefill blocked GEMM 区分；
- shape-dependent workspace；
- concrete layout eligibility；
- alignment/tail specialization；
- contiguous vs strided driver 选择。

推荐演进为纯数据请求：

```cpp
struct KernelPrepareRequest {
    OpType op_type;
    KernelSelector selector;
    OpParams params;
    std::span<const TensorSpec> logical_inputs;
    std::span<const TensorSpec> logical_outputs;
    // 可选：已知 concrete shape/layout specialization facts
};
```

边界要求：

- semantic validity 仍由 compiler artifact 保证；
- request 只携带规划所需纯数据，不借用 `ExecutionContext`/`Runtime`；
- dynamic shape 采用 upper-bound plan 或 per-specialization plan，必须显式选择；
- resolved result 冻结 function、metadata、workspace、packing recipe 和 layout capability。

### 9.3 P1：activation liveness planning

当前 `PrepareExecutionBindings` 为每个 activation 在单一 arena 中顺序分配，不分析 last use，因此峰值接近所有 activation byte size 之和。

建议：

1. 在 immutable plan 上计算 producer/last-consumer interval；
2. external/model output/state/weight 不参与复用；
3. 只允许 proven-disjoint lifetime interval 共用 storage；
4. 对齐、dtype、byte size 与 alias constraint 纳入 placement；
5. 输出 `ActivationPlan {offset, bytes, alignment}`；
6. binding specialization 只解析 concrete sizes并实例化 plan；
7. 用 peak bytes 与 current sequential baseline 对比，不先假设收益。

不应使用通用 allocator 在每个 tensor lifetime 点动态 allocate/free；稳态仍是一块预分配 arena。

### 9.4 P1：Prepared bindings cache policy

prepared bindings 只能在以下全部相同情况下复用：

- `ExecutionPlanBindingKey`；
- concrete shape/symbol values；
- external address、dtype、shape、stride、alignment；
- packed artifact identity；
- state/KV binding layout capability。

Prefill/Decode shape 或 address 改变时必须重新 specialize，除非计划显式设计为 upper-bound/shape-polymorphic binding。不要仅按 step count 或 plan pointer 推断兼容。

## 10. Runtime 演进

### 10.1 P0：KVCache Manager

由 [KVCache Manager 演进方案](05-kv-cache-manager-evolution.md) 定义：contiguous static baseline、lease、append transaction、owner/epoch、execution binding 和未来 Paged KV 边界。

### 10.2 P1：统一 resource plan

Session 创建前应能汇总：

```text
model raw/mapped bytes
packed weight bytes
activation peak bytes
workspace bytes
KV bytes
runtime metadata bytes
```

目标是 allocation 前报告 capacity failure，而不是在 Prefill 中途失败。resource plan 必须区分 reserved、committed、in-use、peak 和 reclaimable。

### 10.3 P1：线程模型与 topology

当前 PRD 同时出现“单线程”和“OpenMP kernel 并行”描述，而源码没有统一 production thread pool。这需要先做产品/架构裁决：

- **单请求**不等于**单执行线程**；
- 推荐同步调用线程负责 control flow，Runtime 拥有固定 worker pool；
- kernel 通过窄 parallel-for/execution-domain contract 使用线程资源；
- 禁止每个 kernel 创建私有线程或直接绑定 OpenMP runtime；
- 避免 nested parallelism、oversubscription 和不受控环境变量；
- 明确 affinity、NUMA node、determinism 和 error propagation；
- 首轮性能优化仍可保持单线程，在线程合同冻结后再并行化。

需要独立 benchmark：Decode GEMV、Prefill GEMM、Attention，不以线程数增加自动推断加速。

### 10.4 P2：多 CPU execution domain

当前一个 Runtime 的 `BackendRegistry` 只缓存一个 `DeviceType::kCPU` backend。只有出现 NUMA/异构核/不同 feature policy 的实际需求时，才演进为：

```text
BackendKey { device_type, device_index, execution_domain }
```

当前单 CPU domain 内继续用 `CpuFeatureSet` 选择 scalar/AVX2/AVX-512/AMX，不为同一 host feature variant 创建多个 backend。

## 11. Backend Dispatch 演进

### 11.1 保留 plan-time resolve

当前 `RuntimeBuilder → BackendRegistry → CpuBackend::PrepareKernel → KernelRegistry → ResolvedKernel → ExecutionPlan → LayerRunner` 路径正确。执行热路径不应重新 registry lookup 或 capability detection。

PRD 中“编译期确定 kernel_ptr”应校正为“planning-time resolve、execution-time frozen function pointer”；当前实现使用 virtual `Backend` 和全局 frozen registry，文档不能宣称纯 compile-time Concepts dispatch。

### 11.2 P1：descriptor capability 完整化

Kernel descriptor/resolve 需要逐步表达：

- logical selector：device/dtype/weight format/phase；
- ISA requirements；
- concrete layout eligibility；
- shape class/tail policy；
- exact packing recipe/version；
- workspace requirement；
- immutable metadata builder；
- params builder；
- deterministic/debug identity。

避免把所有维度都塞进 `KernelSelector`。selector 保持稳定结构请求，shape/layout specialization 通过 prepare request 和 descriptor capability 完成。

### 11.3 P1：packing service 边界

当前 `WeightPrepackPlanner` 直接构造 `CpuWeightPrepacker`，且 recipe 主要由 selector 推导。目标应是：

- backend/descriptor 决定 exact recipe；
- model/execution preparation 根据 optimized graph 的具体 `WeightBinding` 请求 materialization；
- packing service 接收 raw/composite logical view并返回 opaque artifact；
- artifact key 包含 source artifact、value/binding、selector、exact recipe/version；
- kernel 只消费与自身 resolved recipe 精确匹配的 artifact；
- 不在 `ModelLoader` prepack。

## 12. CPU Kernel 演进

### 12.1 P0：完整 reference chain

**状态（2026-09-17）：主链已齐备**——`KVCacheUpdate` 与 causal `Attention`（含 GQA、Prefill/Decode 与稳定 softmax）均已落地（`cpu::kvcache_update_f32_reference` / `cpu::attention_f32_reference`），layout/alias/bounds 与数值测试覆盖于 `KVCacheUpdateKernel.*` / `CPUKernelAttention.*`。

后续：进入 production Executor vertical slice（ExecutableModel/真实绑定/端到端数值验收，见 [01](01-inference-session-generate-readiness.md) §3.2–§3.5），再谈 fused/paged/SIMD Attention。

### 12.2 P1：性能优先级

1. Decode Linear/GEMV 与 fused QKV/Gate-Up；
2. Prefill blocked GEMM；
3. Decode Attention 的历史 KV 扫描；
4. Prefill Attention；
5. RMSNorm/AddRmsNorm/SiluMul/Embedding 等 bandwidth-bound kernels；
6. INT8/INT4 weight-only kernels；
7. 多线程与 NUMA；
8. AArch64 NEON/SVE（仅在目标硬件纳入验收后）。

GEMM 细节见 [CPU GEMM 优化方案](04-cpu-gemm-optimization.md)。

### 12.3 kernel 共同质量门禁

- params 在 binding-time 构造，执行期不重复 shape/alias 全量分析；
- exact in-place 与 partial overlap policy 明确；
- unknown/stride-hole 不伪装为 proven disjoint；
- output mapping injective；
- checked size/stride/address arithmetic；
- reference 与 optimized descriptor 的 supported contract 一致；
- optimized descriptor 内部处理合法 tail/fallback，不能依赖 registry 在 params build 失败后回退；
- benchmark 检查实际 machine code/build flags，噪声小幅变化不作为结论。

## 13. Memory / Allocator 演进

### 13.1 当前边界

通用 allocator 负责 bytes/alignment/device ownership；activation planner、workspace planner 和 KV Manager 分别负责各自领域的 logical lifetime。不要让 ammalloc 或 `Allocator` 承担 KV prefix/refcount/eviction 等上层语义。

### 13.2 P1：provider 与 arena 统一

- raw/mapped model weights：只读长期 storage；
- packed weights：ExecutableModel lifetime；
- activation arena：prepared specialization lifetime，支持 liveness offsets；
- workspace arena：execution scratch，支持 lifetime-aware reuse；
- KV storage：Session/Runtime policy lifetime；
- metadata/params arena：heap-stable cold-path storage。

各 arena 共享 allocation provider 和统计合同，但不合并语义 ownership。

### 13.3 P1：workspace lifetime 真正生效

`WorkspaceRequirement` 已有 `lifetime/reusable`，但当前 `PlanWorkspaceRequirements()` 仍为所有 requirement 顺序累加。目标 planner 应基于 execution schedule 与 lifetime scope复用兼容 slice，并保证：

- non-reusable/persistent 不重叠；
- per-operator 顺序执行可复用；
- per-layer/per-token/per-sequence 的边界有真实 schedule identity；
- alignment 与 overflow 正确；
- 规划失败不部分修改 requirement；
- peak bytes 有可解释 dump。

在没有 layer/token/sequence schedule identity 前，不要仅按 enum 名称猜测 overlap。

### 13.4 P2：NUMA/huge page

只有在 7B 真实模型 benchmark 表明 page/TLB 或 remote memory 是瓶颈后，再引入：

- execution-domain-local allocation；
- first-touch/affinity policy；
- configurable 2 MiB huge page；
- model weight interleave vs bind；
- KV/activation local placement。

## 14. Shape Inference / Base / DTypes 演进

### 14.1 Shape inference

当前 TensorSpec、ShapeSymbol、ShapeConstraint 和 concrete evaluator 足以支撑当前产品。建议提升：

- constraint diagnostics 带 op/value/port/dim context；
- symbol provenance 与冲突报告；
- specialization key 的 canonical symbol assignment；
- upper-bound vs exact-shape planning 的显式合同；
- zero-dimension/overflow 行为跨 operator 一致。

不建议当前建设通用动态 shape optimizer 或 runtime shape graph。

### 14.2 Base / TensorView / DTypes

这些模块不需要独立架构演进。继续按使用方需求加固：

- ownership 与 borrowed view 区分；
- shape/stride/address checked arithmetic；
- const correctness；
- DataType 与 C ABI code 映射稳定；
- FP16/BF16 转换与数值测试；
- 避免把 backend layout、quantization storage 或 Session state塞入通用 Tensor。

### 14.3 Containers

`amstring`/Array/Map 等通用容器不是当前推理闭环的阻塞项。除非 profile 证明它们位于模型加载或热路径瓶颈，不进行与 Generate 无关的重构。

## 15. Public C++ / C API 演进

### 15.1 C++ ownership model

推荐公开对象层次：

```text
Runtime                 long-lived resources
ExecutableModel         compiled/planned immutable model
Session                 mutable single-request generation state
GenerationConfig        max_new_tokens/eos_token_id
GenerationResult        output token ids + finish reason
```

Session 同步、非线程安全；不同 Session 是否能并发取决于未来 Runtime resource policy，当前产品不承诺。

### 15.2 C ABI

需要冻结：

- opaque handles 与 create/destroy；
- ABI version query；
- status code 与 owned error message；
- model/session/runtime lifetime；
- input/output buffer capacity 与 required-size reporting；
- null/zero-length/overflow behavior；
- no exception across ABI；
- symbol visibility/export；
- versioned compatibility tests。

现有 `include/c_api.h` 主要是通用 object refcount/error primitives，不是 PRD 中的 inference C ABI。不要直接把内部 `TensorView`、`Status`、STL container 或 graph/compiler 类型暴露到 C boundary。

### 15.3 发布前冻结顺序

1. 先通过真实 FP32 Prefill→Decode vertical slice；
2. 再冻结 C++ Session orchestration；
3. 再定义 C ABI handle/buffer/error contract；
4. 增加 ABI compatibility tests；
5. 最后承诺 v1.x compatibility。

## 16. Tests / Benchmark / Quality 演进

详细治理框架见 [工程质量体系建设方案](02-engineering-quality-system.md)。本路线图要求以下 vertical slices：

### 16.1 P0 correctness slices

- accepted HF config → exact semantic graph；
- optimized graph → complete kernel resolve；
- raw/composite weights → exact packed artifact；
- KV transaction → KVCacheUpdate → Attention；
- tiny Llama Prefill + two Decode steps；
- public Session/C API error cleanup。

### 16.2 P0/P1 hard evidence

- steady-state Decode allocation count 0；
- same-platform 100-run deterministic token/hash；
- plan/bind/execute failure injection；
- stale KV/session/resource handles rejected；
- real-model load-to-plan integration fixture；
- target build variants in CI，而不是只存在 CMake option。

### 16.3 benchmark hierarchy

```text
microkernel
  -> prepared kernel
  -> production ExecutionStep
  -> layer/block
  -> Prefill/Decode
  -> end-to-end model
```

每层明确是否包含 packing、binding、allocation、cold cache、thread startup 和 model I/O。不能用 microkernel GFLOPS 代替 tok/s、TTFT 或总内存证据。

## 17. Documentation / Product Contract 演进

### 17.1 当前需要裁决的漂移

- PRD 同时描述单线程与 OpenMP kernel 并行；
- PRD 写 compile-time/static dispatch，而实现是 planning-time registry resolve + frozen function pointer；
- PRD 承诺 INT8/INT4，但 loader/kernel 尚未支持；
- PRD 包含多 Session 隔离验收，但当前产品边界为单请求/单 active KV slot；
- 存量 KV/Graph/Dispatch 设计文档混有“已实现”和“推荐接口”。

这些项目应由 [文档系统稳定化方案](03-documentation-stabilization.md) 按“当前事实 / 当前产品目标 / 长期方向”分类，不通过修改代码去迎合过期文案。

### 17.2 capability status 自动化

长期建议从以下事实生成或校验 capability matrix：

- registered kernel descriptors；
- operator schemas/inference dtype contracts；
- CMake target/build options；
- test suite/filter existence；
- public header symbols。

自动化只报告漂移，不替代数值、性能或端到端验证。

## 18. 不建议近期实施的演进

以下项目不是“遗漏”，而是明确延后：

- standalone Primitive IR；
- generic cost-based optimizer；
- Paged KV/prefix cache/continuous batching；
- HTTP/gRPC server；
- GPU/CUDA/CANN production execution；
- MoE、multimodal、LoRA；
- 通用动态 shape runtime；
- 多模型/多租户调度；
- speculative decoding；
- distributed inference。

进入条件必须是 PRD 范围更新、独立设计评审和可验证 workload，而不是为未来预埋大量无消费者 abstraction。

## 19. 分阶段实施路线

### Batch A：correctness contract（立即）

- model `hidden_act` 支持范围与 graph builder 对齐；
- ElementwiseMul graph/operator broadcast contract 对齐；
- KV reserve/commit/owner/generation 修复；
- capability matrix 建立；
- 对现有错误路径补 regression tests。

退出条件：被 loader/graph 接受的输入不会静默编译成不同语义；KV 未写入数据不可读。

### Batch B：FP32 production vertical slice

- execution-native KV state binding；
- KVCacheUpdate/Attention FP32 reference；
- `PrepareExecutableModel`；
- automatic weight bindings/materialization；
- tiny Llama Prefill + two Decode；
- synchronous C++ Generate。

退出条件：真实 production path 数值闭环，无 fake backend/placeholder。

### Batch C：physical planning 与内存

- `KernelPrepareRequest`；
- descriptor exact packing recipe；
- activation liveness plan；
- workspace lifetime-aware planning；
- unified resource budget/dump；
- binding specialization cache contract。

退出条件：所有 physical memory/packing/kernel 选择在 execution 前冻结，Decode 无规划/分配。

### Batch D：INT8/INT4

- 量化格式合同；
- loader/validator/resolver；
- graph/lowered metadata；
- exact packer/artifact；
- reference dequant + optimized kernels；
- accuracy/memory/performance e2e。

退出条件：PRD 指定格式真实可加载和生成，误差与内存门禁通过。

### Batch E：性能与 topology

- GEMM/Attention optimized kernels；
- Runtime thread pool contract；
- measured parallelism；
- NUMA/huge-page 评估；
- AArch64 是否进入当前验收的产品裁决。

退出条件：基于目标硬件、目标模型和实际 build flags 的端到端证据。

### Batch F：API/ABI 与发布

- C++ Session 生命周期冻结；
- C ABI v1；
- compatibility/error/lifetime tests；
- capability manifest 与 release checklist；
- 文档和实现状态对齐。

退出条件：满足 PRD hard gates，并能以受支持的公开入口复现。

## 20. 专题拆分规则

本路线图是总索引，不应无限增长。满足以下任一条件时，从本文件拆出独立编号提案并反向链接：

- 修改三个以上模块的公共合同；
- 引入新的 artifact/owner/lifetime；
- 需要独立 benchmark 或数值验收；
- 存在两个以上可行架构方案；
- PRD 当前承诺但实现跨度超过一个里程碑。

下一批最可能需要独立提案的主题：

1. INT8/INT4 weight-only format and execution；
2. shape-aware kernel preparation + activation/workspace planning；
3. Runtime thread pool/topology；
4. C++ Session/C ABI v1 freeze。

在进入实现前再创建专题文档，避免仅凭路线图重复现有 01/04/05 的详细设计。

## 21. 验收标准

- 每个当前产品 capability 都有唯一 owner module、事实状态和测试证据；
- P0 缺口全部映射到现有或新建专题提案；
- semantic、compiler、execution、backend 支持范围不再相互矛盾；
- accepted model config 能闭环到 production execution，或在 load/prepare 阶段明确拒绝；
- FP32 tiny Llama Prefill→Decode 闭环后再开放 public Generate；
- INT8/INT4 只有端到端格式、kernel 和精度证据齐全后才标记支持；
- Decode steady-state 无 heap allocation、registry lookup、shape inference 或 repack；
- 性能结论有目标硬件/模型/build flags/统计方法；
- 长期方向不会泄漏到当前 Graph IR、API 或 runtime hot path。

## 22. 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-09-16 | 1.0 | 首次建立全仓库 capability gap、模块演进裁决、依赖顺序和 Batch A–F 路线图 |
| 2026-09-17 | 1.1 | 同步 Attention/KVCacheUpdate reference kernel 落地：§3 模块总表、§6.3、§12.1 状态更新 |
