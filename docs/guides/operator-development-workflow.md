# AetherMind 算子开发与优化工作流

- **状态**: Current
- **版本**: 1.0
- **日期**: 2026-09-17
- **适用范围**: operators / graph semantic fusion / backend kernels / execution binding / model weight packing
- **文档规范**: [AetherMind 文档系统规范](documentation-guide.md)
- **技术方法参考**: [算子优化指南](operator_optimization_guide.md)

## 1. 目的与适用边界

本文定义 AetherMind 所有算子开发、reference kernel、optimized kernel、packing/layout、fusion 和 threading 变更的统一工程流程。它冻结的是职责、证据门禁、文档角色和追溯关系，不冻结某个算子的具体优化顺序、tile、阈值或 ISA 选择。每个算子根据机制选择工作包，但不得绕过本文定义的 correctness、production-path 和证据门禁。

## 2. 核心原则

1. **语义先于实现**：先冻结 input/output、dtype、shape、stride/layout、alias、zero-size、overflow 和 numerical semantics，再实现 kernel。
2. **reference 与 optimized 分离**：reference 是 correctness oracle，不为追求性能而修改成难以审计的实现。
3. **source inspection、unit test、benchmark 与端到端证据不可互相替代**。
4. **microbenchmark 不代表产品性能**：必须同时验证真实 registered/prepared production path。
5. **不记录无上下文的性能数字**：所有性能结论必须绑定代码版本、机器、构建配置、运行命令和原始数据。
6. **失败实验也是证据**：有意义的被否定方案必须保留，避免重复试错。
7. **同一事实只在一个位置详述**：计划、过程、验证、当前设计、ADR 和原始数据各司其职。
8. **未达到门禁不得提高 production priority，也不得把计划写成已实现能力**。

## 3. 文档与证据拓扑

| 内容 | 位置 | 权威范围 | 更新方式 |
|---|---|---|---|
| 算子专项路线、工作包、风险、退出条件 | `docs/improvement-plan/NN-<operator>-optimization.md` | 未来工作与当前实施状态 | 持续更新状态和证据链接 |
| 实验过程、失败尝试、参数 sweep | `docs/tests/operators/<operator>/<work-package>-log.md` | 过程记录 | 追加式，不改写旧结论 |
| 正式 correctness/performance 验证 | `docs/tests/operators/<operator>/` | 指定 commit/环境的证据快照 | 新结果新建报告，旧报告不覆盖 |
| 重大架构/合同决策 | `docs/decisions/NNNN-*.md` | 决策与被否定方案 | 状态迁移，不删除历史 |
| 已经实现的当前设计 | `docs/designs/<module>/NN-*.md` | 当前仓库事实 | 与代码同 PR 同步 |
| 通用技术方法 | `docs/guides/operator_optimization_guide.md` | SIMD、Roofline、packing 等方法 | 方法变化时更新 |
| 短期缺陷与待办 | `docs/issues.md` | 短生命周期问题 | 完成后勾选 |
| 行为可见变化 | `CHANGELOG.md` | 发布可见变化 | 随行为变更更新 |
| JSON/perf/反汇编等原始结果 | CI artifact 或 gitignored `benchmark-results/` | 机器可读原始证据 | 每次运行产生新 run ID |

专项提案只保留工作包状态、当前结论和正式验证报告链接，不持续粘贴原始 benchmark 表格。`designs/` 只描述已经验证并落地的实现，不承载实验方案。

## 4. Change Profile

每次算子工作开始前先分类；一个变更可以命中多个 profile。

| Profile | 示例 | 必须产物 |
|---|---|---|
| Semantic | dtype/shape/alias/in-place/zero-size 合同变化 | proposal 或 ADR、inference tests、operator contract/design 更新 |
| Reference | 新 reference kernel 或 correctness 修复 | reference tests、数值 oracle、correctness 说明 |
| Optimized | scalar/SIMD/ISA-specific kernel | experiment log、microbenchmark、production benchmark、validation report |
| Packing/Layout | packed weight、layout、workspace | proposal、exact recipe/ownership 评审、integration validation；重大决策写 ADR |
| Fusion | QKV/Gate-Up/AddRmsNorm 等 | semantic rewrite review、consumer/port/weight binding 检查、端到端数值验证 |
| Threading | task partition、thread pool、NUMA | ownership/并发 ADR、TSAN、scaling 与 oversubscription 报告 |
| Tuning-only | tile、unroll、prefetch、threshold | experiment log；采纳后更新正式 validation |

## 5. 通用工作包与门禁

### O0：语义与合同

必须明确：

- semantic operator 与 backend primitive 的边界；
- input/output 端口顺序和含义；
- dtype、rank、shape 和 deferred constraint；
- stride/layout 和 output injectivity；
- alias/in-place policy；
- empty/zero-size、overflow、null pointer 行为；
- numerical accumulation、rounding 和 error budget；
- unsupported/unknown layout 的错误类别。

退出条件：operator inference、binding-time validation 和 kernel contract 不互相矛盾；错误行为有聚焦测试。

### O1：Reference baseline

- 实现简单、可读、独立的 correctness oracle；
- 覆盖完整合法合同，不以优化路径的限制反向收窄语义；
- 对 optimized implementation 使用独立的期望值或 reference 对照；
- 明确 reference 能证明什么，不能证明什么。

退出条件：canonical、boundary、zero-size、stride、alias、overflow 与数值测试通过。

### O2：Benchmark baseline

根据算子选择以下层级：

| 层级 | 用途 | 是否可单独作为产品结论 |
|---|---|---|
| Primitive/microkernel | tile、unroll、prefetch、指令吞吐诊断 | 否 |
| Prepared operator | registered/resolved kernel + prepared params | 是，单算子主门禁 |
| Preparation/packing/binding | 冷路径成本与 amortization | 作为成本证据 |
| Execution integration | ExecutionPlan/bindings/workspace | 验证框架开销 |
| End-to-end | 真实 Prefill/Decode 或产品场景 | 最终产品证据 |

计时循环外完成 correctness guard。除非 benchmark 明确测量 preparation，否则 registry lookup、validation、allocation、packing 和 params build 不得混入 steady-state compute。

退出条件：baseline 可重复，shape/cache/thread/preparation 维度可独立归因，原始结果可保存和比较。

### O3：Optimization proposal

使用 [算子优化提案模板](../templates/operator-optimization-plan.md)，至少记录：

- 已验证瓶颈与尚未验证的假设；
- 目标 shape、layout、dtype、ISA 和产品场景；
- 可选方案及 trade-off；
- fallback 与兼容策略；
- workspace、packing、ownership 和线程依赖；
- correctness/performance 验收门禁。

### O4：实验与调优

使用 [算子实验日志模板](../templates/operator-experiment-log.md)。一次有独立假设、参数 sweep 或失败结论的工作形成一个日志条目。日志必须区分：

- 已验证事实；
- 基于数据的推断；
- 尚未验证的假设；
- 采纳、拒绝或继续收集数据。

失败实验不得删除。纯机械重复运行无需新建条目，但其 raw run ID 应保留。

### O5：正式验证

使用 [算子验证报告模板](../templates/operator-validation-report.md)。报告必须覆盖适用项：

- correctness/numerical error；
- layout/alias/fallback；
- production-path benchmark；
- preparation/packing break-even；
- allocation、workspace 和 ownership；
- ISA dispatch 与 portability；
- concurrency/scaling；
- integration/end-to-end；
- 未覆盖项和结论边界。

正式报告是指定 commit、机器和环境的不可变快照；新证据新建报告，不覆盖旧报告。

### O6：Production closeout

门禁通过后按需执行：

- 调整 descriptor priority 或默认选择；
- 更新专项提案状态和证据链接；
- 更新当前 module design；
- 重大决策新增/更新 ADR；
- 行为或性能基线可见变化更新 CHANGELOG；
- 将未完成风险登记到 `docs/issues.md`。

## 6. Benchmark 与原始数据规范

### 6.1 必需元数据

所有正式性能结论必须记录：

```text
date/time and timezone
git commit and working-tree dirty state
baseline/candidate identity
CPU model/stepping/microcode
OS/kernel
compiler/version
build type and flags
effective CPU features
thread count and affinity
NUMA binding
governor/turbo/SMT
memory configuration when relevant
benchmark command
raw artifact run ID/path/URL and checksum
```

缺少关键元数据的结果只能作为本地观察，不能用于 production 决策。

### 6.2 Run ID 与目录

推荐 run ID：

```text
YYYYMMDDTHHMMSSZ_<git-sha>_<host-id>_<variant>
```

本地结果建议放在 gitignored：

```text
benchmark-results/operators/<operator>/<run-id>/
├── context.json
├── baseline.json
├── candidate.json
├── comparison.txt
├── perf-stat.txt
└── disassembly.txt
```

大量 machine-specific JSON 不进入 `docs/`。需要长期保留时使用 CI artifact/object storage，并在 validation report 中记录 artifact ID、URL、checksum 和 retention policy。只有稳定、体积小、用于自动回归的 canonical baseline 才可经评审提交到仓库。

### 6.3 比较规则

- baseline/candidate 使用同一机器、构建配置和线程/NUMA设置；
- 使用独立进程并采用交错 A/B 次序；
- 保存全部 repetitions，不用单次最优值；
- Decode/Prefill、hot/streaming、single/multi-thread 分组报告；
- logical bytes 与实际 DRAM traffic 明确区分；
- 小于系统噪声或置信区间重叠的差异不得宣称收益；
- microbenchmark 改善若未经过 production/integration 验证，不得提高 production priority。

## 7. 文档状态流转

```text
专项提案 Draft
    → In Progress
    → 实验日志（追加）
    → validation report
    → 门禁通过？
       ├── 否：继续实验 / Rejected / Needs More Data
       └── 是：更新 proposal、ADR、current design、CHANGELOG
    → Implemented 或 Superseded
```

`Implemented` 表示提案范围已落地且门禁已通过，不表示算子从此不再优化。新的目标、ISA、dtype 或 packing contract 应创建新工作包或新提案。

## 8. ADR 判定

通常需要 ADR：

- semantic/physical abstraction boundary；
- packed-weight recipe/versioning；
- shape-dependent workspace contract；
- runtime thread-pool ownership；
- quantization metadata layout；
- 对 public API/ABI 或跨模块依赖产生长期影响的选择。

通常不需要 ADR：

- `MR/NR`、unroll、prefetch distance；
- 某一 CPU 的 dispatch threshold；
- 未采纳的局部调参结果。

后者记录在 experiment log 和 validation report。

## 9. PR/评审门禁

提交算子变更时，评审者至少检查：

- [ ] Change Profile 已识别；
- [ ] reference 与 optimized implementation 独立；
- [ ] 语义、layout、alias、zero-size、overflow 合同未被意外缩窄；
- [ ] correctness 测试覆盖目标和 fallback；
- [ ] benchmark 测量对象与名称一致；
- [ ] steady-state 计时边界正确；
- [ ] 性能结论包含环境、原始数据与统计依据；
- [ ] microbenchmark 与 production-path 证据已区分；
- [ ] packing/workspace/ownership/concurrency 风险已处理；
- [ ] proposal、validation、ADR、current design 和 CHANGELOG 按触发条件同步；
- [ ] 未运行或未证明的内容明确标注。

## 10. 反模式

- 直接优化 reference kernel，导致 correctness oracle 与 candidate 共用复杂逻辑；
- 只保存最终最好数据，删除失败实验；
- 用 microkernel GFLOP/s 宣称端到端收益；
- 在 benchmark timed loop 内做无意的 allocation、validation 或 cache clearing；
- baseline/candidate 来自不同机器或配置却直接比较；
- 把 raw JSON、大段终端输出粘贴进 improvement plan；
- 把提案写进 `designs/`，或把未实现能力标成 Current；
- 没有 exact recipe/ownership 证据就修改 packed layout；
- 没有 runtime threading contract 就在 kernel 内创建线程。

