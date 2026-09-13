# AetherMind 工程质量体系建设方案

- **状态**: Draft
- **版本**: 1.2
- **日期**: 2026-09-14
- **产品范围**: [AetherMind 产品需求](../products/aethermind_prd.md)
- **架构基线**: [架构总览](../designs/architecture/architecture_overview.md)
- **关联模块**: 全仓库

## 1. 结论与定位

AetherMind 的质量建设不采用“寻找单一标杆并推动全仓库代码模仿”的方式，也不开展逐文件、逐模块的统一达标运动。本方案以当前产品能力、真实执行链路和可验证风险为起点，建立以下闭环：

```text
产品能力与发布要求
    → 风险台账
    → 变更类型与适用门禁
    → 验证证据
    → 可持续回归资产
```

仓库中已经存在契约清晰、生命周期明确或验证充分的局部实现，也存在产品链路未闭环、质量工具不完整和文档状态漂移等系统性问题。质量建设的目标不是让所有文件外观一致，而是让关键风险被发现、设计约束被证明、发布门禁可重复执行，并防止后续改动降低已建立的基线。

### 1.1 Capability-driven 原则

本方案按能力和成熟度组织，不使用 `Phase 1`、`Phase 2` 等阶段编号定义架构、API、质量门禁或实施里程碑。

- 阶段编号只可作为历史产品规划语境，不得成为模块 ownership、依赖方向、类型或接口命名的一部分。
- 当前范围使用具体 capability 描述，例如“本地 CPU”“同步单请求”“Llama family”“Token ID 接口”“FP32 correctness baseline”。
- 暂不建设的能力直接列明，例如服务端调度、continuous batching、distributed execution 或 GPU backend，不归入抽象的“下一阶段”。
- 能力成熟度使用 `Experimental / Supported / Production-ready / Deprecated` 等状态表达。
- 未来能力由明确需求、架构约束和验证证据驱动，不由阶段编号驱动。

当前静态文档仍包含阶段性表述时，其有效范围约束在完成术语迁移前继续生效；术语弱化不得被解释为扩大当前实现范围。

### 1.2 目标

1. 识别并收敛影响当前核心推理链路的 correctness、lifetime、architecture、concurrency、robustness 和 performance 风险。
2. 为不同类型的变更建立风险触发式验证门禁。
3. 建立 warnings、sanitizers、fuzzing、coverage 和 benchmark 的可重复入口。
4. 保证新增代码不降低既有质量基线，并以增量方式治理历史问题。
5. 通过端到端 vertical slice 验证跨模块真实契约，而不是只验证孤立组件。
6. 使设计决策、缺陷、验证结果和文档事实能够相互追溯。

### 1.3 非目标

- 不进行全仓库风格统一式重构。
- 不要求所有组件使用相同设计模式。
- 不按 `base → operators → runtime → ...` 顺序机械翻修全部模块。
- 不以测试数量、注释数量、文档数量或全仓平均覆盖率代表质量。
- 不为启用工具而修改无关代码或第三方依赖。
- 不把 correctness baseline、public API 闭环和性能优化捆绑在同一批次。
- 不预先建设当前产品范围之外的服务化、调度、分布式或硬件后端能力。

## 2. 当前问题

### 2.1 局部质量与系统能力不均衡

部分组件已经形成可复用的工程经验，例如：

- alias/layout 分析区分事实分类与调用方策略，并保留“已证明违反”与“无法证明”的语义差异；
- compiler 使用受控 draft、结构验证和不可变 artifact 边界；
- KV cache view 通过 generation 检测失效借用；
- reference kernel 保留 correctness oracle，并区分 cold-path binding validation 与运行时可变数据验证。

这些实现可作为案例，但不是新的规范来源。能否推广必须重新分析目标模块的语义、ownership、性能和错误模型。

### 2.2 核心推理链路仍需持续闭环

质量建设必须沿真实数据流进行：

```text
HF model
  → LoadedModel
  → ModelGraph
  → LoweredModelArtifact
  → ExecutionPlan
  → PreparedExecutionBindings
  → Executor
  → KV state
  → token output
```

孤立组件测试通过不能证明该链路已经满足发布要求。必须重点验证 compiler artifact、execution plan、weight binding、state identity、KV commit 和 token result 之间的跨模块契约。

### 2.3 质量工具链与发布要求不闭环

当前顶层构建已经提供 TSAN 选项，但产品要求中出现的 ASAN 入口尚未在顶层 CMake 中形成对应能力。warnings、UBSAN、fuzzing、coverage 和静态分析也缺少统一、可重复、由项目 target 承载的标准入口。

### 2.4 文档事实可能落后于实现

文档治理与稳定化详见 [03 号方案](03-documentation-stabilization.md)。任何质量基线和实施优先级都必须以当前代码、构建配置和测试事实为准，不能直接复制历史结论。

## 3. 质量模型

### 3.1 质量属性

重要设计和代码评审按以下属性检查，但只对适用项提出要求：

| 属性 | 核心判定问题 |
|---|---|
| 语义与 correctness | 输入输出、边界条件、错误语义和 invariant 是否正确 |
| Architecture / API / ABI | 职责、依赖方向、公开接口和兼容性是否正确 |
| Ownership / lifetime | 谁拥有、谁借用、何时失效、失败路径如何清理 |
| Concurrency | 共享状态、同步、线程安全边界和内存序是否明确 |
| Performance / resource | 算法、分配、数据移动、cache、SIMD 和 workspace 是否合理 |
| Robustness / portability | 不可信输入、溢出、平台、compiler 和 ISA 差异是否处理 |
| Evolvability | 单一事实源、coupling 和扩展成本是否可控 |

质量属性描述系统最终必须具备的性质，不规定具体实现形式。

### 3.2 可选工程手段

以下手段根据问题选用，不作为逐项打勾的统一要求：

- typed contract 与强类型 identity；
- `Status` / `StatusOr` 错误传播；
- immutable artifact 与 staged builder；
- generational handle；
- RAII 与显式 borrow；
- reference implementation；
- checked arithmetic；
- prepare/run 冷热路径分离；
- narrow runtime binding；
- property-based 与 differential testing。

工程评审必须说明为什么某个手段适合当前问题，而不是因为仓库其他组件使用了它。

### 3.3 证据等级

| 等级 | 证据 | 可以支持的结论 |
|---|---|---|
| E0 | 假设、设计推演 | 候选风险或候选方案 |
| E1 | 源码、类型、依赖和控制流静态事实 | 当前机制与可达路径 |
| E2 | 聚焦单测、负面测试、property/differential test | 已覆盖输入空间内的行为 |
| E3 | ASAN、UBSAN、TSAN、静态分析、fuzzing | 对应动态或静态缺陷类别在给定环境中未复现 |
| E4 | 真实端到端执行、故障注入、生命周期测试 | 跨模块合同和失败路径成立 |
| E5 | Release benchmark、重复实验、机制分析 | 给定 workload 和环境中的性能结论 |

评审和交付报告必须区分证据等级。E1 不能证明性能收益；普通测试通过不能代替 ownership、sanitizer 或 concurrency 证据；未运行的验证不得写成通过。

## 4. 风险台账

风险台账是本方案的控制中心，而不是附属看板。每项质量工作必须能关联到真实风险、发布要求或防回退目标。

| 字段 | 要求 |
|---|---|
| ID | 使用稳定编号，便于评审、提交和验证报告引用 |
| 影响链路 | 标明 model、graph、compiler、execution、runtime、backend、API 等影响面 |
| Invariant | 写明必须始终成立的条件 |
| Failure mode | 描述违反后可能出现的错误结果、越界、UAF、数据竞争、ABI 破坏或性能退化 |
| Severity | P0 / P1 / P2，并说明判断依据 |
| Reachability | 区分当前生产路径可达、测试路径可达、未来能力相关 |
| 当前证据 | 分开记录静态事实、机制推断和动态验证 |
| 处置 | 修复、增加证明、接受风险或移出当前范围 |
| 验收门禁 | 指定所需 E1–E5 证据 |
| 状态 | Open / In Progress / Verified / Accepted |

短生命周期缺陷继续记录在 [问题跟踪](../issues.md)；跨多个里程碑的系统性风险记录在本提案或对应专题提案中。已完成项只有在验收证据可定位时才能标记为 `Verified`。

### 4.1 不采用的指标

以下数字容易被机械提高，不作为质量目标：

- 测试用例总数；
- 注释行数或文档数量；
- 全仓平均覆盖率；
- 未区分严重程度和 baseline 的 clang-tidy 违规总数；
- 未注明环境、波动和 workload 的单次 benchmark 数值。

### 4.2 保留的运营指标

- 未关闭的 P0/P1 风险及其停留状态；
- 相对冻结 baseline 的新增 compiler warning 和静态分析违规；
- 适用验证 profile 的缺失数量；
- sanitizer、fuzzing 和关键端到端测试最近一次有效结果；
- 当前核心推理链路的合同闭环状态；
- 热路径在受控实验中的统计显著回退。

指标必须能够触发具体行动。只增加数字、不改变工程决策的指标不进入门禁。

## 5. 风险触发式验证门禁

### 5.1 基础门禁

所有代码改动至少满足：

1. 明确要解决的问题、改动范围和关键 invariant；
2. 构建最小受影响 target；
3. 运行最窄的相关测试，再按风险扩大范围；
4. 检查改动文件格式和 `git diff --check`；
5. 不引入新增 compiler warning；
6. 按文档触发矩阵同步事实源；
7. 在交付报告中列明已验证、推断、未验证和残余风险。

### 5.2 条件 profile

| Profile | 触发条件 | 强制证据 |
|---|---|---|
| Public API / ABI | public header、C API、枚举数值、对象布局或导出符号变化 | API contract、兼容性分析、ABI test、公共文档 |
| Lifetime | borrowed view、handle、buffer、prepared binding、资源销毁顺序变化 | 生命周期与失效语义、负面测试、ASAN |
| Concurrency | lock、atomic、共享可变状态、线程安全声明变化 | concurrency invariant、TSAN、压力或故障路径测试 |
| Untrusted input | 文件、JSON、safetensors、外部配置 | 输入上限、checked arithmetic、负面测试、ASAN+UBSAN、fuzzing |
| Compiler / graph | rewrite、lowering、constraint proof、artifact 构建 | invalid-artifact tests、结构验证、源图/失败状态约束 |
| Kernel | tensor shape、layout、alias、dtype、SIMD | semantic oracle、边界/alias/overflow tests、sanitizer |
| Hot path | execution loop、kernel、allocator、KV 或 packing | Release benchmark、allocation/data-movement 证据 |
| Platform / ISA | compiler-specific、AVX2、NEON 或平台分支 | capability detection、fallback、目标平台构建或未验证声明 |

不适用的 profile 不需要为了“达标”而人为引入相应机制。一次改动命中多个 profile 时，门禁取并集。

## 6. 工作流

### 6.1 Workstream A：代码事实基线与风险台账

**文档治理不属于本 Workstream，见 [03 号方案](03-documentation-stabilization.md)。** 本节只保留代码侧事实基线与风险台账工作。

先建立当前代码事实，不从历史计划直接推导优先级。

交付内容：

- 当前能力与缺口清单（代码侧）；
- P0/P1 风险台账；
- README 与 CMake 构建选项一致性检查；
- 对时间敏感的 kernel、API 和执行链状态进行源码复核；
- 单独审查 `StatusCode` 的 canonical-code 表述、C ABI 数值和兼容策略。

退出条件：

- P0/P1 判断均有当前代码证据；
- 已实现能力、计划能力和未实现能力不混写；
- 每项高风险问题都有明确处置方向和验收证据。

### 6.2 Workstream B：可重复质量工具链

建立可组合的 build/validation profiles：

```text
default
warnings
asan-ubsan
tsan
coverage
fuzz
release-benchmark
```

设计约束：

- warnings 和 sanitizer 通过 project-owned target 或集中 CMake helper 传播，不使用无边界的全局编译选项；
- ASAN+UBSAN 可以组成一个 profile，TSAN 与其互斥；
- 配置阶段检查 compiler、platform 和 sanitizer capability；
- third-party target 和第三方头文件不纳入本项目 `-Werror`；
- warnings/static-analysis 先冻结 baseline，再实行零新增，最后按 target 清零；
- Doxygen 先达到零新增 warning，再评估 `WARN_AS_ERROR`；
- clang-tidy、coverage、fuzzing 均为显式 profile，不改变默认开发构建成本；
- IWYU 在收益和误报经过小范围试点前不设为公共头强制门禁。

退出条件：

- 每个 profile 有唯一、可复制的配置和运行入口；
- 至少一个聚焦测试集合在 ASAN+UBSAN、TSAN 下真实执行；
- 默认构建行为与产物不发生非预期变化；
- unsupported 配置在 configure 阶段得到明确诊断；
- 工具门禁进入正常开发或 CI 流程，而不是孤立脚本。

### 6.3 Workstream C：核心推理链路 assurance

沿生产数据流验证四个关键 contract boundary：

1. compiler artifact → execution plan；
2. raw/packed weights → immutable execution bindings；
3. execution state identity → narrow KV kernel binding；
4. Prefill → Decode → KV commit → token result。

每个边界必须回答：

- 上游交付哪些已经证明的事实；
- 下游还需要验证哪些动态条件；
- 谁拥有 backing storage，谁只借用；
- 失败是否留下部分状态；
- 状态推进和对外可见的 commit point 在哪里；
- 热路径是否重复执行本可在 prepare 阶段完成的工作。

该工作流与 [InferenceSession / Generate 前置闭环计划](01-inference-session-generate-readiness.md) 协同，但在实施前必须刷新其中时间敏感的实现状态。

退出条件：

- tiny Llama 配置通过真实 CPU Prefill 和至少两个 Decode step；
- logits、token、KV content 和 commit position 与独立 reference 一致；
- plan 执行失败不推进 KV commit；
- stale view、错误 binding 和越界 position 被稳定拒绝；
- deterministic repeat 通过；
- 验证不通过 fake backend、空 kernel 或测试 helper 绕过生产入口。

### 6.4 Workstream D：不可信模型输入

优先提取可供 production code、unit test 和 fuzz target 共用的纯解析边界，例如：

- HF config 的 `std::string_view` 解析入口；
- safetensors index 的 `std::string_view` 解析入口；
- safetensors 文件字节、header length、JSON header 与 data offsets 的 `std::span<const std::byte>` 解析入口。

目录存在性、symlink、canonical path 和 shard containment 属于文件系统策略，使用临时目录集成测试覆盖，不与高速 byte fuzz target 混合。

每个 fuzz target 必须定义：

- 输入长度与资源上限；
- seed corpus 和必要 dictionary；
- ASAN+UBSAN instrumentation；
- corpus merge/minimize；
- crash reproducer 保存方式；
- 固定时长 smoke run；
- 成功解析后的 invariant validation。

退出条件不以 target 数量计算，而以可持续运行、crash 可复现、资源受控、合法结果满足后置条件为准。

### 6.5 Workstream E：性能与资源证据

性能门禁只覆盖真实热路径和已稳定的语义：

- GEMM / Linear；
- RMSNorm；
- RoPE；
- Attention；
- Decode loop；
- workspace / KV access；
- weight packing。

实验要求：

- Release 构建；
- baseline 与 candidate 使用一致配置；
- 固定 workload、线程数、CPU affinity 和数据初始化；
- 保存原始 repetitions；
- 报告中心趋势、离散程度和环境信息；
- 同时记录 allocation、workspace 和 bytes moved 等机制指标；
- reference kernel 保留为 correctness oracle。

没有 E5 证据时只能写“候选优化”或“机制推断”，不能声称获得生产性能收益。

### 6.6 Workstream F：架构边界防回退

[仓库级执行指南](../../AGENTS.md) 的模块 ownership 表继续作为依赖规则的权威来源。近期不恢复只扫描禁止 include 的孤立脚本，也不立即为了边界检查拆分全部 CMake target。

推荐顺序：

1. 跨模块改动在 review profile 中强制进行 dependency inspection；
2. 记录真实发生的边界回退和人工检查成本；
3. 有证据表明需要自动化后，再建设基于 compile database 或 compiler dependency information 的检查器；
4. 检查器必须有明确 owner、合法依赖配置、自身测试以及标准质量入口；
5. 只有独立构建、可见性或增量编译需求成立时，才评估拆分 component targets。

## 7. 实施批次

### Batch -1：Documentation Stabilization

**详见 [03 号方案](03-documentation-stabilization.md)。** 本方案不复述 D0–D5 步骤。

Batch -1 的退出条件 E1–E9（见 [03 号方案 §9](03-documentation-stabilization.md)）是 Batch 0 的准入门禁。文档治理、清查表、阶段性术语迁移与 `verify_docs.py` 演进全部在 Batch -1 内完成。

**当前状态（2026-09-14）**：规划工件已交付（见 [03 号方案 §1.3](03-documentation-stabilization.md) 快照），D1–D5 实质执行未启动；E1–E8 失败，E9 部分完成；**Batch 0 阻塞中**。

### Batch 0：事实与风险基线

**前置依赖**：Batch -1 退出条件 E1–E9 全部满足后方可启动。

范围限定为代码事实基线与风险台账审核（文档治理已迁移至 Batch -1）：

- 生成当前能力清单和质量基线审核（代码侧）；
- 建立 P0/P1 风险台账；
- 完成 `StatusCode` API/ABI 专项审核；
- 建立工具链缺口清单。

本批次不得顺带修改业务实现。

### Batch 1：最小工具链闭环

- target-scoped warnings；
- ASAN+UBSAN profile；
- 收敛现有 TSAN 配置；
- 聚焦测试验证；
- README、构建指南和产品门禁同步。

先保证入口可用、诊断可信，再逐步提高严格度。

### Batch 2：两个质量 vertical slice

选择两类差异明显的真实风险验证质量模型：

1. state/KV lifetime slice：覆盖 ownership、stale view、commit 和 TSAN/ASAN；
2. HF parser slice：覆盖 untrusted input、checked arithmetic、UBSAN 和 fuzzing。

试点结束后审查 profile 是否遗漏风险、门禁是否成本过高，再决定是否形成稳定开发指南。

### Batch 3：核心推理链路闭环

按真实依赖推进 state binding、executable artifact、缺失 reference compute 和 direct Prefill→Decode proof。功能实施由对应专题计划管理；本方案只定义每批必须交付的质量证据。

### Batch 4：发布与性能门禁

在 correctness 和生命周期闭环后接入：

- public API/ABI profile；
- Decode 稳态分配检测；
- Release benchmark baseline；
- coverage 基线；
- Doxygen 和静态分析增量门禁；
- platform/ISA capability 验证。

### Batch 5：持续治理

- 新改动执行基础门禁和适用 profile；
- P0/P1 风险定期复核；
- 失效文档及时修订或标记 Superseded；
- 已验证、跨多个组件稳定复用的实践再写入 `docs/guides/`；
- 只有产生长期、跨版本约束的决策才新增 ADR。

## 8. 方案选择与权衡

### 8.1 推荐：风险驱动 + vertical slice + 增量门禁

优点：

- 质量工作直接服务于真实产品风险；
- 跨模块合同得到验证；
- 工具成本与改动风险匹配；
- 能通过 baseline ratchet 治理历史问题；
- 不依赖阶段编号，能够适应未来模型、backend 和部署能力扩展。

代价：

- 前期需要事实审计和风险分类；
- 不会快速得到一张“全仓达标率”表；
- 不同改动的门禁不同，需要评审者进行工程判断。

### 8.2 否决：六维度逐文件达标

该方案混合质量属性、设计模式和验证手段，容易诱导无适用性的机械重构；文件也不是 ABI、lifetime、concurrency 或端到端 correctness 的合理验收单元。

结论：不采用。

### 8.3 否决：先一次性补齐全部工具

全局 warnings、IWYU、tidy、sanitizer、coverage 和 fuzzing 同时接入，会混合工具链问题、历史问题和业务问题，无法判断失败来源，也容易污染第三方 target。

结论：不采用；按 profile 建立最小闭环后逐步扩展。

### 8.4 否决：模块顺序式全仓重构

从 base 开始逐模块翻修具有高 blast radius，且不能保证优先解决当前核心链路的真实阻塞。局部设计改善也不能证明跨模块合同成立。

结论：不采用；按风险和端到端链路选择改动。

## 9. 总体验收标准

本方案进入稳定运行状态需要满足：

1. 当前能力、计划能力和历史结论有清晰状态边界；
2. P0/P1 风险均已关闭、被明确接受或有受控处置计划；
3. default、warnings、ASAN+UBSAN、TSAN 等核心 profile 可重复运行；
4. 当前核心推理链路存在真实端到端 correctness 证据；
5. public API/ABI、lifetime、concurrency、untrusted input、kernel 和 hot path 改动能够触发对应 profile；
6. warning、static-analysis 和文档门禁至少做到零新增回退；
7. fuzz target 能持续运行并保存可复现失败；
8. 性能结论来自受控 Release 实验而非单次数字；
9. 质量工具和检查器进入正常开发或 CI 入口，不依赖无人维护的辅助脚本；
10. 新架构、API 和里程碑命名不再依赖阶段编号。

## 10. 文档落地策略

本提案在实践验证前保持 `Draft`，不立即新增一份宣称稳定的“设计品质指南”。

文档演进顺序：

1. 本提案定义目标、工作流和门禁模型；
2. `docs/reviews/` 记录事实基线与专项审核；
3. `docs/tests/` 记录重要动态验证；
4. `docs/issues.md` 跟踪短生命周期缺陷；
5. 经过至少两个 vertical slice 验证后，将稳定且通用的规则提炼到开发指南；
6. 只有涉及长期架构、ABI 或并发约束时才新增 ADR；
7. 阶段性术语迁移应单独审核，避免改变仍然有效的产品范围约束；
8. 文档治理规范以 [03 号方案](03-documentation-stabilization.md) 为准；本方案不再单独定义文档 metadata、状态或清查流程。

## 11. 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| 1.0 | 2026-09-13 | 初始版本：采用 capability-driven、风险驱动、vertical slice 和增量门禁模型 |
| 1.1 | 2026-09-13 | 拆分文档治理至 [03 号方案](03-documentation-stabilization.md)；Batch 0 依赖 Batch -1 退出条件 E1–E9；Workstream A 更名为"代码事实基线与风险台账" |
| 1.2 | 2026-09-14 | §7 Batch -1 增加当前状态指针：规划工件已交付，D1–D5 未执行，E1–E8 失败、E9 部分完成，Batch 0 阻塞中 |
