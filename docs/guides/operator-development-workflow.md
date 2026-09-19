# AetherMind 算子开发与优化工作流

- **状态**: Current
- **版本**: 1.3
- **日期**: 2026-09-19
- **适用范围**: operators / graph semantic fusion / backend kernels / execution binding / model weight packing
- **文档规范**: [AetherMind 文档系统规范](documentation-guide.md)
- **优化方法**: 见本文附录 C（原《算子优化指南》并入，独立文件已删除）

> v1.3 变更：算子开发文档收敛为**单一规范文档**——原 `operator_optimization_guide.md`（2110 行）压缩为附录 C 后删除；流程性/GPU/文档模板等重复内容删减，方法主体保留。AGENTS 与文档入口统一单指向本文。
>
> v1.2 变更：算子工作流收敛为**单一文档**——专项提案与工作文件骨架并入本文件附录 A/B，对应模板文件删除；引用统一指向本文。
>
> v1.1 变更：文档拓扑由 9 类收敛为 5 类；实验日志与验证报告合并为单一工作文件（一工作包一文件）；证据要求按 Change Profile 分级；性能元数据分为核心/全档两级。原"每次运行新建验证报告"的不可变快照机制废止。

## 1. 目的与适用边界

本文定义 AetherMind 所有算子开发、reference kernel、optimized kernel、packing/layout、fusion 和 threading 变更的统一工程流程。它冻结的是职责、证据门禁、文档角色和追溯关系，不冻结某个算子的具体优化顺序、tile、阈值或 ISA 选择。每个算子根据机制选择工作包，但不得绕过本文定义的 correctness、production-path 和证据门禁。

## 2. 核心原则

1. **语义先于实现**：先冻结 input/output、dtype、shape、stride/layout、alias、zero-size、overflow 和 numerical semantics，再实现 kernel。
2. **reference 与 optimized 分离**：reference 是 correctness oracle，不为追求性能而修改成难以审计的实现。
3. **microbenchmark 不代表产品性能**：必须同时验证真实 registered/prepared production path。
4. **性能结论必须可追溯**：绑定 commit、机器、benchmark 命令和 raw 数据；无上下文的数字不记录。
5. **失败实验也是证据**：有意义的被否定方案必须保留，避免重复试错。
6. **同一事实只在一个位置详述**：work file 是过程与结论的唯一证据载体，proposal/design/ADR 只保留状态和链接。
7. **未达到门禁不得提高 production priority，也不得把计划写成已实现能力**。

## 3. Change Profile 与证据等级

每次算子工作开始前先分类；一个变更可以命中多个 profile，证据等级取最高者。

| Profile | 示例 | 必须产物 | 等级 |
|---|---|---|---|
| Semantic | dtype/shape/alias/in-place/zero-size 合同变化 | correctness 测试 + 合同说明；无性能报告 | 轻 |
| Reference | 新 reference kernel 或 correctness 修复 | correctness 测试 + 数值 oracle + correctness 说明 | 轻 |
| Optimized | scalar/SIMD/ISA-specific kernel | proposal + work file 完整区段 + production benchmark | 标准 |
| Packing/Layout | packed weight、layout、workspace | proposal + exact recipe/ownership 评审 + integration 验证；重大决策写 ADR | 标准 |
| Fusion | QKV/Gate-Up/AddRmsNorm 等 | semantic rewrite review + consumer/port/weight binding 检查 + 端到端数值验证 | 标准 |
| Threading | task partition、thread pool、NUMA | ownership/并发 ADR + TSAN + scaling 与 oversubscription 报告 | 标准 |
| Tuning-only | tile、unroll、prefetch、threshold | work file 一条记录；采纳后更新正式 design | 轻 |

规则：

- **轻级**：只要求 correctness 测试与提交说明，不写性能报告，不建 proposal。
- **标准级**：要求 proposal（瘦身模板）+ 工作文件完整区段 + 性能结论附核心元数据（见 §6）。
- 未运行或未证明的内容必须明确标注，不得省略为留白。

## 4. 文档与证据拓扑

| 内容 | 位置 | 说明 |
|---|---|---|
| 专项路线、工作包、风险 | `docs/operators/<op>/<op>-optimization.md` | 只保留现状、方案、工作包状态和证据链接，不粘贴实验数据 |
| **过程 + 结论（唯一证据文件）** | `docs/operators/<op>/benchmarks/<work-package>.md` | 一工作包一文件；实验区追加式保留失败，结论区按 `commit@date` 冻结追加、不覆盖。按[附录 B](#附录-b算子工作文件骨架) 创建 |
| 索引与门禁状态 | `docs/operators/<op>/README.md` | 一屏汇总：工作包 × 机器 × 门禁状态 |
| 重大决策 | `docs/decisions/NNNN-*.md` | 仅 §7 判定情形 |
| 当前设计 | `docs/designs/<module>/NN-*.md` | 与代码同 PR 同步 |
| 台账 / 发布 | `docs/issues.md`、`CHANGELOG.md` | 短生命周期问题与行为可见变化 |
| 原始数据 | `benchmark-results/`（gitignored）+ CI artifact | 机器可读原始证据，不进 `docs/` |

算子目录的完整形态与各层职责以 [docs/operators/README.md](../operators/README.md) 为唯一说明，本文不重复绘制目录树。同一工作包**一个文件**：环境元数据只在 work file 出现一次；facts/inference/assumptions 只写一次。正式结论是绑定 commit、机器、环境的快照，新证据按条追加，不新建文件、不覆盖旧结论。

## 5. 工作包与门禁（O0–O6）

每步列出"做什么 / 退出条件 / 何时可跳过"。算子的机制决定选择哪些工作包，但 correctness、production-path 与追溯门禁不可绕过。

| 工作包 | 内容与退出条件 | 可跳过 |
|---|---|---|
| **O0 语义与合同** | 明确 semantic operator 与 backend primitive 边界、端口顺序、dtype/rank/shape、stride/layout、alias/in-place、empty/zero-size/overflow、数值预算与错误类别；退出 = inference、binding-time validation 与 kernel contract 不矛盾，错误行为有聚焦测试 | 不改变合同的 Tuning-only |
| **O1 Reference baseline** | 简单、可读、独立的 correctness oracle，覆盖完整合法合同；退出 = canonical/boundary/zero-size/stride/alias/overflow 与数值测试通过 | 已有 reference，仅优化现有实现 |
| **O2 Benchmark baseline** | 以 registered/resolved kernel + prepared params 的 **prepared operator** 为主门禁；microkernel 仅作诊断。计时循环外完成 correctness guard；preparation 成本单独测 | Reference/Semantic 级 |
| **O3 Optimization proposal** | 按[附录 A](#附录-a专项提案骨架) 的提案骨架记录瓶颈与假设、目标 shape/layout/ISA、方案 trade-off、fallback、workspace/packing/ownership 依赖、验收门禁 | Tuning-only、Reference |
| **O4 实验与调优** | 在 work file 追加；一次有独立假设/sweep/失败结论的工作形成一条记录，区分已验证事实/推断/假设与"采纳/拒绝/继续"；失败不得删除；纯机械重跑只保留 raw run id | Reference/Semantic 级 |
| **O5 正式结论** | 在 work file 冻结区按 `commit@date` 追加；覆盖 correctness/数值误差、layout/alias/fallback、production-path benchmark、preparation break-even、workspace/ownership、dispatch、并发、集成/端到端，以及未覆盖项与结论边界 | Tuning-only（采纳结论并入 design） |
| **O6 Production closeout** | 门禁通过后按需执行：descriptor priority、提案状态与链接、design、ADR、CHANGELOG、issues 风险登记 | — |

## 6. Benchmark 与原始数据规范

### 6.1 两级元数据

- **核心（所有性能结论必填）**：

```text
git commit and working-tree dirty state
CPU model
OS/kernel
compiler/version and build type/flags
benchmark command
raw artifact run ID/path/URL
```

- **全档（仅 end-to-end 结论、production 决策或跨机器比较需要）**，在核心之外增加：

```text
date/time and timezone      baseline/candidate identity
CPU stepping/microcode      effective CPU features
thread count and affinity   NUMA binding
governor/turbo/SMT          memory configuration
raw JSON/checksum
```

缺少核心元数据的结果只能作为本地观察，不能用于 production 决策。

### 6.2 Run ID 与目录

```text
benchmark-results/operators/<operator>/<run-id>/
├── context.json ├── baseline.json ├── candidate.json
├── comparison.txt ├── perf-stat.txt └── disassembly.txt
```

run ID 格式 `YYYYMMDDTHHMMSSZ_<git-sha>_<host-id>_<variant>`。大量 machine-specific JSON 不进入 `docs/`；需长期保留时使用 CI artifact，在 work file 中记录 artifact ID、URL、checksum 与 retention。只有稳定、极小、用于自动回归的 canonical baseline 才可经评审提交。

### 6.3 比较规则

- baseline/candidate 同机同配置，独立进程交错 A/B，保存全部 repetitions；
- Decode/Prefill、hot/streaming、single/multi-thread 分组报告；logical bytes 与实际 DRAM traffic 区分；
- 噪声或置信区间内重叠的差异不得宣称收益；microbenchmark 改善未经 production/integration 验证，不得提高 production priority。

## 7. 决策、评审与反模式

### 7.1 状态流转

```text
专项提案 Draft → In Progress → 实验记录（work file 追加）→ 正式结论（work file 冻结）
    → 门禁通过？ 否：继续实验 / Rejected / Needs More Data
                是：更新 proposal、ADR、design、CHANGELOG → Implemented / Superseded
```

### 7.2 ADR 判定

需要 ADR：semantic/physical abstraction boundary；packed-weight recipe/versioning；shape-dependent workspace contract；runtime thread-pool ownership；quantization metadata layout；对 public API/ABI 或跨模块依赖产生长期影响的选择。

不需要 ADR：`MR/NR`、unroll、prefetch distance、某 CPU 的 dispatch threshold、未采纳的局部调参结果（记入 work file）。

### 7.3 PR 评审门禁

- [ ] Change Profile 已识别；
- [ ] reference 与 optimized 独立；语义/layout/alias/zero-size/overflow 合同未被意外缩窄；
- [ ] correctness 测试覆盖目标和 fallback；steady-state 计时边界正确；
- [ ] 性能结论包含核心元数据，microbenchmark 与 production-path 证据已区分；
- [ ] proposal、work file、ADR、design、CHANGELOG 按触发条件同步；未运行或未证明内容明确标注。

### 7.4 反模式

- 直接优化 reference kernel；只保存最好数据、删除失败实验；
- 用 microkernel GFLOP/s 宣称端到端收益；在 timed loop 内做无意的 allocation/validation；
- baseline/candidate 跨机器或跨配置直接比较；把 raw JSON/长终端输出粘贴进提案；
- 没有 exact recipe/ownership 证据修改 packed layout；没有 runtime threading contract 就在 kernel 内建线程；
- 把未实现能力标成 Current，或把提案写进 `designs/`。

## 附录 A：专项提案骨架

新建 `docs/operators/<op>/<op>-optimization.md` 时复制以下骨架（原 `operator-optimization-plan.md` 模板并入本文）。

````markdown
# <算子名> 优化方案

> 存放于 `docs/operators/<op>/`。过程证据不入本文件——实验与正式结论统一记入本算子的 `benchmarks/` 工作文件（见附录 B）。流程见本文。

- **状态**: Draft / In Progress / Implemented / Superseded
- **版本**: 1.0
- **日期**: YYYY-MM-DD
- **产品边界**: [AetherMind 当前产品 PRD](../../products/aethermind_prd.md)
- **Change Profile**: Semantic / Reference / Optimized / Packing/Layout / Fusion / Threading / Tuning-only
- **关联代码**: <paths/symbols>
- **关联工作文件**: <docs/operators/<op>/benchmarks/<work-package>.md>
- **关联 ADR**: <无或链接>

## 1. 结论与范围

- 推荐方案和原因；包含与不包含的范围；明确事实、推断、假设。

## 2. 现状、瓶颈与假设

| 能力 | 当前实现 | 证据 |
|---|---|---|
| <能力> | <实现> | <代码/测试/工作文件链接> |

| 瓶颈/假设 | 类型（已验证事实 / 推断 / 假设） | 验证方式 |
|---|---|---|

## 3. 语义与合同约束

仅当本变更改变合同（port/dtype/shape/stride/alias/zero-size/overflow/数值预算/ownership）时填写；无变化写"无变化"。

## 4. 方案与备选

- 目标架构与数据流（职责边界、核心结构、cold/hot path、fallback）。
- 方案对比：`方案 | 优点 | 缺点 | 结论`。
- 重大且长期的决策另建 ADR，本节只摘要并链接。

## 5. 验证与门禁设计

- 引用工作文件：benchmark 层级（prepared operator 为主门禁）、shape/layout/cache/thread matrix、correctness guard、raw 数据保存方式。
- 验收点：correctness / architecture-API / performance / integration 各一行。

## 6. 风险、相关文档与变更记录

| 风险 | 影响 | 缓解 |
|---|---|---|

- proposal / 工作文件 / design / ADR / issues：

| 日期 | 版本 | 变更 | 原因 | 证据/PR |
|---|---|---|---|---|
````

## 附录 B：算子工作文件骨架

新建 `docs/operators/<op>/benchmarks/<work-package>.md` 时复制以下骨架（原 `operator-work-file.md` 模板并入本文）。它是工作包过程与结论的**唯一证据载体**：实验区追加式（保留失败实验），结论区按 `commit@date` 冻结追加、不覆盖。

````markdown
# <算子名> <工作包> 工作文件

- **专项提案**: <link>
- **状态**: Open / In Progress / Under review / Closed
- **门禁状态**: correctness [ ] / production-path [ ] / 可追溯 [ ]

## 1. 目标与合同不变式

- 目标（一句话）：
- 合同不变式（端口/dtype/shape/stride/alias/zero-size/overflow/数值预算）：无变化则写"无变化"；有变化必须列明并同步 OperatorSchema/测试。
- 关联代码 / 关联测试：

## 2. 实验记录（追加式，失败保留）

### YYYY-MM-DD — EXP-001：<标题>

- 假设 / 预期机制：
- 验证命令（benchmark/test）与核心元数据（commit、dirty、CPU、OS/kernel、compiler+flags、raw artifact）：

```bash
# exact commands
```

- Correctness：`测试 | 结果 | 备注`（PASS/FAIL/NOT RUN）
- 结果摘要：`Case | Baseline | Candidate | Delta | Raw artifact`（不粘贴完整 JSON/终端输出）
- 分析与决定：Accepted / Rejected / Needs More Data；下一步：

## 3. 正式结论（冻结区，按 commit@date 追加，不覆盖）

### <commit-sha>@<YYYY-MM-DD> — <结论名称>

- 判定：Accepted / Rejected / Needs More Data
- 环境快照（核心 6 项；production 结论附全档）：
- correctness / numerical error 摘要：
- production-path benchmark 摘要：
- layout/alias/fallback、workspace/ownership、dispatch、并发（适用时）：
- 未覆盖项与结论边界：

## 4. 门禁判定

- [ ] correctness 与 safety 测试通过
- [ ] production-path benchmark 已运行且附核心元数据
- [ ] 结果可追溯（commit/机器/命令/raw artifact）
- [ ] 未运行或未证明的内容已标注

## 5. 相关链接

- proposal / design / ADR / issues / 原始数据路径：
````

## 附录 C：优化方法

> 原《算子优化指南》（2110 行）于 2026-09-19 并入本文附录 C，独立文件已删除。压缩时删除了与 O0–O6 流程重复的叙述、GPU 专项、旧文档模板等内容；方法主体（Roofline、correctness 体系、内存/SIMD/threading 优化、典型算子策略）全部保留。噪声 floor 量化规范在 §C.2.4。

### C.0 总体原则

算子优化是一套闭环工程，不是"加多线程/套 SIMD"：

```text
定义契约 → 建立 reference → 建立 baseline → 理论建模 → profiling → 提出假设
    → 单点优化 → 回归验证 → 继续迭代
```

1. **先定义契约，再写代码**：输入输出语义、数值精度、layout、shape 范围、目标硬件必须先明确。
2. **先保证正确，再追求极限性能**：正确性体系覆盖边界 shape、随机数据、特殊数值、误差容忍度和端到端影响。
3. **先建立性能模型，再选择方向**：FLOPs / Bytes / 算术强度 / Roofline / cache 层级建立性能假设。
4. **用 Profiling 验证假设**：看似 memory-bound 可能实则受 shuffle、reduction、TLB、分支、spilling、同步限制。
5. **优先高层收益**：算子融合、减少 pass、减少中间 tensor、权重预打包 > 局部指令微调。
6. **面向目标 workload**：prefill / decode / 单请求 / 多请求瓶颈不同。
7. **结果可复现、可回归、可维护**：每个版本回答——为什么快、快多少、哪些 shape 快、有无 fallback、是否破坏数值稳定性。

### C.1 契约与 workload

#### C.1.1 算子契约清单

- **输入输出语义**：tensor 数量/shape/dtype/layout/stride；输出是否预分配；contiguous 要求；对齐要求；aliasing；in-place；空 tensor；batch 维参与切分；hidden/head/seq 特殊维。
- **数值语义**：bitwise equal / tolerance equal / top-k equal；累加精度（FP32 vs FP64 reference）；NaN/Inf 传播；denormal/FTZ；rounding；fast-math 是否允许近似；epsilon 位置与默认值；溢出稳定策略；端到端 logits/token 一致性。
- **性能目标**：latency vs throughput 优先；单核 vs 多核；是否要求 steady-state zero allocation；小算子 call overhead；目标 ISA 路径与必需 fallback。
- **目标硬件**：CPU 微架构（x86/ARM）；内存层级假设；WSL2/容器对性与可用 ISA 的影响（虚拟化会把 P/E 混合拓扑伪造成对称 CPU，`taskset` 仅咨询性）。

#### C.1.2 Prefill / Decode 差异

| 阶段 | 特征 | 主要优化方向 |
|---|---|---|
| Prefill | token 多、GEMM 大、复用潜力高 | GEMM packing、micro-kernel、多线程分块、blocking、KV cache 批量写、epilogue 融合 |
| Decode | 每步 1 token、GEMM→GEMV/skinny、call overhead 敏感、KV 读取重要 | small-batch GEMV 专用路径、权重预打包、KV layout、低开销 dispatch、避免小算子起线程、适度融合 RMSNorm/RoPE/elementwise |

### C.2 正确性与基准

#### C.2.1 Reference vs Baseline

- **Reference**：正确性对齐；逻辑直接；可用 double/标准库；不追求性能。
- **Baseline**：性能对比基准；朴素标量实现；不复杂优化。

#### C.2.2 正确性测试体系

- **固定用例**：最小/常见/大 shape；非 2 的幂与质数维度；SIMD 宽度整除与不整除；cache 边界附近；batch=1 与 >1；输出 LLM 常见尺寸（4096/8192/11008 等）。
- **随机测试**：uniform/normal；小幅值/大幅值/混合符号/全正/全负；sparse-like/重复值/极值。
- **特殊数值**（按 contract 决定）：`0.0`/`-0.0`/`NaN`/`±Inf`/denormal/极大极小值；fast-math 路径须明确这些值的处理。

#### C.2.3 误差评价指标

| 算子类型 | 推荐指标 |
|---|---|
| Elementwise | max abs diff、max relative diff |
| Reduction / RMSNorm | max diff、double reference 对比 |
| Softmax | max diff、sum-to-one、argmax 一致 |
| GEMM | max diff、relative diff、误差分布、必要时 ULP |
| LLM logits / greedy | max diff、top-1/top-k 一致性；采样路径看分布稳定性 |

#### C.2.4 噪声 floor、repetitions 与最小可信 delta（方法权威）

固定百分比回退阈值（如"5% 即 REGRESS"）不是可套用常数：噪声 floor 是"机器 × benchmark 组"属性，同组跨机器可差一个数量级，最差组可能完全不同。自动门禁启用前必须量化：

1. **分解方差来源**：进程内方差（repetition 抖动，每 case CV/stddev，提高 repetitions 有效）与跨进程系统偏移（换进程整体平移，提高 repetitions **无效**，只能对比多进程 median 分布）。
2. **同进程交错 A/B**：baseline/candidate 同一进程内交替执行，使共同偏移在配对差值中抵消；保存原始 per-repetition 行。
3. **用零改动样本校验阈值**：同一实现两轮采集互比，若被判 REGRESS/IMPROVE，该阈值在该组只能降级为人工判读提示；只有噪声 floor 明显低于阈值的组进自动门禁；判读用该机器该组 floor，不用全局固定值。
4. **绑定环境与拓扑**：记录 CPU/microcode/kernel/编译器/governor/SMT/NUMA/内存频率；虚拟化下拓扑可能伪造，核类别/SMT 结论仅裸机采集，虚拟化结果标注为指示性。
5. **结论**：小于该组 noise floor 的差异不是收益证据；未达门禁不得提高 descriptor priority。

### C.3 理论模型与 Roofline

- **FLOPs**：MAC 计作 2 FLOPs（1 乘 1 加），口径统一是建模与跨团队对比的前提。RMSNorm 约 `4N + 常数`。
- **Bytes**：按实际内存层级流量估算，不只张量逻辑大小（两遍扫描/写分配/非临时存储都会改变流量），必要时 PMU 验证。
- **算术强度** `AI = FLOPs / Bytes`：AI 低大概率 memory-bound；但 AI 只是第一层判断，瓶颈还可能来自 TLB、load/store port、shuffle、reduction 依赖链、branch、front-end、spilling、同步、NUMA remote。
- **Roofline**：`Attainable = min(Peak Compute, AI × Memory Bandwidth)`；机器平衡点 `= Peak Compute / Bandwidth`。不用厂商标称峰值，实测 STREAM 带宽、单核/多核带宽、实际 FMA 峰值、AVX-512 降频、AMX 吞吐。
- **层级 Roofline**：逐层自问——是否受 DRAM / L2 / L1 带宽限制？load/store port？reduction 依赖链？shuffle？线程同步？

### C.4 Profiling（CPU）

贯穿全过程：`建 baseline → profiling → 提出瓶颈假设 → 单点优化 → 再 profiling → 验证收益`。每次优化回答：优化前瓶颈、手段针对什么、指标是否改善、是否引入新瓶颈、跨 shape 是否稳定。

| 工具 | 用途 |
|---|---|
| Linux perf / pmu-tools | 采样、PMU 计数器 |
| VTune / uProf | pipeline、memory、threading |
| likwid | 拓扑、带宽、硬件计数器 |
| numactl / taskset | NUMA 绑定与验证 |

关键指标：cycles、instructions、**IPC**、L1/L2/LLC miss、DRAM 带宽、branch miss、TLB miss、vectorization ratio、stalled cycles、port pressure、register spilling、remote NUMA access。

### C.5 算法级优化与融合

优先级：`减少计算/访存次数 > 减少中间 tensor > 提高数据复用 > SIMD/指令级 > 多线程扩展`。

| 融合方向 | 目的 |
|---|---|
| GEMM + Bias/Activation/Residual（epilogue 融合） | 避免完整中间矩阵写回读回 |
| RMSNorm/Linear/RoPE 与前后算子 | 减少额外读写 |
| Attention score + mask + softmax | 减少 score 中间存储 |
| Softmax + value matmul、Dequant + MatMul | 流式/免中间反量化 tensor |

**Fast Math**：默认不全局开启；每算子显式声明是否允许近似；LLM 输出敏感路径做端到端验证；`rsqrt` 近似后通常需 Newton-Raphson；允许按精度档位提供 kernel。风险：NaN/Inf 传播、signed zero、rounding、denormal、累积误差、top-k/token 一致性。

### C.6 内存层级优化

- **连续访问优先**：cache line 利用率、硬件预取、SIMD load、TLB 压力都更优；跨步访问会导致 cache/prefetch 失效、TLB miss、gather 成本。
- **Layout**：GEMM packing 目标是连续访问+提高 cache 复用+匹配 SIMD/AMX tile+简化寻址；KV cache layout 要考虑 decode 按 head 连续读、head_dim SIMD 对齐、写入低开销、支持 paged、减少 TLB/cache miss。
- **Cache Blocking**：寄存器 → L1 → L2 → L3 → NUMA 层级；参数权衡 cache 容量/associativity/line/TLB/prefetch/寄存器/SIMD 宽度/conflict miss。
- **2 的幂次步长诅咒（Conflict Miss）**：`hidden_size = 2^N`（4096/8192）跨列访问会把数据映射到同一 cache set。解法：padding（4096→4096+δ）、swizzling、blocking 参数避开 2^N 对齐。这是 GEMM 列访问与并行列偏移的经典陷阱。
- **Alignment**：大 buffer ≥64B；AVX2 32B、AVX-512 64B；权重预打包按 cache line 对齐。但现代 CPU 对 unaligned 支持好，避免为对齐引入复杂分支。
- **Prefetch**：只对间接/跨步/可预测但硬件难识别/有明显 memory latency stall 的场景手工 prefetch；连续线性访问依赖硬件预取器。
- **False Sharing**：线程私有 reduction buffer 用 `alignas(64)` 或合理布局，避免同 cache line ping-pong。
- **Non-temporal store**：只写一次且短期不读的大输出适用；输出很快被下一算子读取则不适用。
- **量化权衡**：反量化可能使算子从 memory-bound 变 ALU-bound——边反量化边计算、SIMD unpack 解 INT4、仅反量化权重；blocking 参数（MC/NC/KC）取 `group_size` 整数倍使 Scale/Zero-point 连续复用。

### C.7 SIMD / 指令级 / Micro-kernel

- **主循环 + tail**：N 不整除 SIMD 宽度时处理剩余元素；AVX-512 用 mask tail，AVX2 选 scalar tail / masked 模拟 / over-read（需 contract 允许）/专用 remainder kernel。
- **多累加器**：拆 `sum0..3` 提升 ILP、降依赖链、提 FMA 利用率；但增加寄存器压力，过头会 spilling。
- **FMA**：更高吞吐与精度；注意与非 FMA 路径结果不 bitwise equal，contract 需明确。
- **Horizontal reduction / shuffle**：成本不可忽略；横向归约只在循环末尾做一次；对固定维度写专用 reduce；shuffle 高则重新设计 layout、减少跨 lane、用 SoA/AoS 适配；RoPE shuffle 成本受 split-half vs interleaved layout 影响。
- **Register pressure**：展开/多累加器导致 spilling 时检查汇编/compiler report，拆循环、收窄 live range、减展开因子。
- **ISA dispatch**：`Scalar fallback → SSE/NEON → AVX2 → AVX-512 → AMX`。启动检测 feature，按 dtype/shape/layout/ISA 选择，小/大 shape 分路径，预留 fallback，各路径共享测试。选择逻辑示例：AVX2 且 hidden≥阈值走 SIMD kernel，否则 reference。

### C.8 多线程与 NUMA

前提：先让单线程 kernel 达到合理效率再扩展（多线程掩盖单核低效；小算子启动成本可能大于收益）。推荐顺序：单线程 correctness → baseline → profiling → SIMD/memory 优化 → 多线程 scaling → NUMA。

- **线程池**：Executor 持有长驻 worker，支持绑核/NUMA 分组/低开销 barrier；避免每次调用创建线程与 steady-state malloc。
- **切分维度**：Elementwise 按 contiguous；Norm/Softmax 按 batch/row（head）；GEMM 按 M/N tile；GEMV 按输出行；Attention 按 batch/head/query block。原则：每线程工作量足够大、访问连续、不写同一 cache line、同步少、reduction 合并可控。
- **Load balancing**：shape 不整除/变长序列/mask 导致负载不均时用 block cyclic / work stealing / 动态调度；动态调度有开销，按 workload 选择。
- **Reduction 并行**：每线程私有 partial 并对齐、末尾归并、避免 atomic；小 N 不并行；结果可能与单线程数值不同。
- **NUMA**：thread affinity + first-touch + 按 socket 分配内存与切分任务；`numactl --hardware` / `--cpunodebind/--membind` / `numastat -p <pid>` 验证。
- **系统集成**：检查是否引入临时分配、破坏 tensor 生命周期、兼容 executor/registry/dispatch、影响异常/可测试性/tracing、decode 阶段是否划算。CPU-first LLM 引擎坚持：steady-state zero allocation、single-request low latency、predictable execution、clear fallback。

### C.9 回归、调优与停止准则

- **每次优化必做**：正确性回归、性能 benchmark、多 shape/多 ISA 验证、profiling 对比、端到端影响检查。
- **记录维度**：p50/p90/p99、GB/s、GFLOPS、speedup、scaling efficiency、decode per-token latency、端到端 token/s；**MFU**（实际 GFLOPS/硬件峰值）、**MBU**（实际 GB/s/STREAM 峰值）作为 memory-bound 算子的"终极考卷"。
- **停止准则**：达到目标延迟/吞吐、接近实测硬件上限、瓶颈已落到物理带宽或指令吞吐极限、继续优化收益低于阈值、复杂度超过维护收益、端到端收益不明显。阈值按算子/shape/平台确定，不宜机械套统一比例。
- **Autotuning**：搜索空间（blocking/展开/线程/ISA）× 目标函数（latency/throughput）× 搜索策略（穷举/随机+贝叶斯）× 缓存最优配置；在目标硬件离线运行并持久化、预热阶段轻量覆盖常用 shape、保留 fallback、定期回归防编译器/系统更新退化。

### C.10 典型算子策略速查

| 算子 | 关键优化点 |
|---|---|
| RMSNorm | 两遍扫描 SIMD 化；多累加器降 reduction 依赖；`rsqrt` 由 contract 控制；hidden 固定值专用路径；batch 并行；decode 避免过度多线程；与 residual/elementwise 融合；FP32/BF16/FP16 累加精度明确 |
| Softmax | max/exp/sum 三段 SIMD；用 `inv_sum` 代替逐元素除法；小维专用路径、大维分块；融合 mask/scale/value matmul；不能直接 `exp(x)`（溢出）；masked 行 `-inf` 与全 mask 行为定义；fast exp 验证误差 |
| GEMM/GEMV | 优化层级：loop order → packing → cache/register blocking → SIMD/AMX micro-kernel → 多线程 tile → epilogue 融合 → small shape 专用 kernel → quantized/prepack weight；decode 退化为 `[1×K][K×N]`，重点转权重带宽/cache 复用/量化权重/NUMA/避免过高线程开销 |
| RoPE | split-half layout（本项目 Llama）中 `i` 与 `half+i` 配对；sin/cos 预计算；连续读 Q/K；SIMD 处理 pair、减少 shuffle；head_dim 固定值专用；与 Q/K projection 后处理融合；decode 低开销路径 |
| Attention / KV | decode 瓶颈在 KV 读取带宽与 cache/TLB miss；block/paged cache；head_dim 对齐；K/V 分开存；prefetch 历史 block；streaming softmax；减少 score 中间存；多 head/多 query 并行合并；长短上下文分路径 |

### C.11 开发者检查清单

| 域 | 检查项 |
|---|---|
| 契约 | 输入输出 shape/dtype/layout/stride；contiguous/in-place/aliasing；NaN/Inf/fast-math/误差容忍度；目标 workload 与硬件；benchmark shape 集合 |
| Correctness | reference/baseline 完成；固定+随机+边界+SIMD tail 覆盖；NaN/Inf 按 contract；各 dtype 路径分别测；max abs/rel diff 记录；端到端 logits/token 影响评估 |
| Performance | benchmark 无 malloc/free 干扰、有 warmup 与多轮重复、记录 p50/p90/p99；固定线程/绑核/编译参数；记录 ISA 路径与 GB/s、GFLOPS；与 baseline 和理论上限对比；profiling 证明瓶颈变化 |
| Memory/SIMD | 主访问连续；对齐合理；blocking 参数合理；packing 成本可摊销；无多余中间 tensor；无 false sharing；prefetch 有依据；主循环 SIMD 化且 tail 正确；多累加器；横向归约不频发；无 spilling；dispatch/fallback 正确 |
| Threading | 不每次创建线程；chunk 合理；小 shape 不过度并行；reduction 线程私有 partial；barrier 可控；affinity/NUMA 明确；scaling 已测 |

### C.12 工程落地

- **Kernel dispatch**：按 op type → dtype/weight_format/phase 结构化过滤 → CPU 按 `effective_features ⊇ cpu_requirements` 过滤 → priority 选择 → shape 特化 kernel 内二次分发 → fallback 必须存在。`CpuFeature` 模型见 `../operators/system`。
- **Steady-state zero allocation**：临时 buffer 由 workspace 预分配；初始化阶段分配、推理稳态零分配；小临时用 stack/寄存器；大临时复用 buffer。
- **Benchmark 与 CI 三层**：Correctness CI（每次提交，覆盖各 ISA 与关键 shape）；Performance smoke（每日，宽松阈值防明显回退）；Full benchmark（定期/手动，供优化决策）。