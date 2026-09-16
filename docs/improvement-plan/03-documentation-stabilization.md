# AetherMind 文档系统稳定化方案

- **状态**: In Progress
- **版本**: 2.1
- **日期**: 2026-09-17
- **产品范围**: [AetherMind 产品需求](../products/aethermind_prd.md)
- **架构基线**: [架构总览](../designs/architecture/architecture_overview.md)
- **文档规范**: [文档系统规范](../guides/documentation-guide.md)
- **关联模块**: 全仓库文档
- **配套清查表**: [03-documentation-stabilization-inventory.md](03-documentation-stabilization-inventory.md)

## 1. 结论与定位

本方案是 [02 号工程质量体系建设方案](02-engineering-quality-system.md) 的 **Batch -1 前置里程碑**：在启动 Batch 0（事实与风险基线）之前，先把文档系统从"目录已建立、内容治理未完成"迁移到"核心事实可信、剩余债务可清点"。

当前状态定性：

> 文档目录体系已经建立，但内容治理和事实可信度尚未完成迁移。

其后果是：风险台账、验证门禁与架构约束若直接建立在既有文档之上，会继承文档漂移带来的错误事实。因此本方案不追求一次性重写全部约 2.7 万行文档（那会形成新的长期阻塞项），而是采用两步走：

1. **先建立可信核心**：D0 冻结文档角色，D1 核验 9 篇权威文档，D2 建立全仓库清查表；
2. **再渐进迁移其余文档**：D3 按风险而非目录顺序处置，D4 分类弱化阶段术语，D5 演进校验工具但仍拒绝宣称"事实正确"。

本方案的交付边界：Batch -1 完成时，可信核心已核验，所有非归档文档有明确类型与状态，已知错误已修复或显著标记，剩余债务有清单可渐进处理。**不要求所有文档达到最终形态**。

### 1.1 与既有文档规范的关系

[文档系统规范](../guides/documentation-guide.md) 是文档分类、命名、交叉引用与质量标准的**唯一规范来源**。本方案是迁移实施计划，不是规范文档。本方案自身是 `improvement-plan/` 下的 Draft 提案，**不是文档治理权威**。

本方案的职责：

- 描述从当前状态迁移到符合 documentation-guide.md 的实施步骤（D0–D5）；
- 为存量文档建立一次性清查表，让迁移进度可见；
- 定义 Batch -1 的退出条件，作为 02 号方案 Batch 0 的准入门禁。

**规范变更路径**：本方案提出的状态模型扩展（§3.2）若与 documentation-guide.md 现有定义不同，必须先通过 ADR 或直接修订 documentation-guide.md 接纳后，才能在文档头部落地。在接纳前，清查表中的相关字段仅作为工作标签（working label），不改变文档本身的规范状态。

### 1.2 非目标

- 不改写全部文档正文；
- 不批量重命名或移动文件（避免破坏交叉引用）；
- 不宣称 `tools/verify_docs.py` 能证明"文档事实正确"；
- 不引入新的阶段编号（`Phase1/Phase2`）作为架构、API、模块、里程碑或质量门禁的命名依据；
- 不删除历史文档（archive/ 与 Historical Snapshot 保留）。

### 1.3 当前执行状态（2026-09-17 快照）

**Batch -1 主要实施动作已完成，但最终收口存在 correctness 问题，当前不能认定为 `Implemented`。**

#### 1.3.1 已完成的实施动作

| 动作 | 完成日期 | 证据 |
|---|---|---|
| D0 角色冻结规则定义 | 2026-09-13 | 本方案 §3 |
| D1 核心 9 篇结构核验 | 2026-09-14/17 | §4.4 核验记录 |
| D2 清查表建立（144 行） | 2026-09-13 | inventory |
| D3 P0 处置（01 §2.2 / code_review_guide / issues.md） | 2026-09-14 | E3/E5/E6 已满足 |
| D3 P1 处置（amstring plan 归档） | 2026-09-14 | E4 已满足 |
| D4 核心 9 篇术语迁移 | 2026-09-17 | E7 已满足 |
| D5a verify_docs.py 扩展（389 行） | 2026-09-14 | E8 已满足 |

#### 1.3.2 未完成的收口动作

| 动作 | 状态 | 说明 |
|---|---|---|
| D0 frontmatter 落地到具体文档 | 未执行 | 规则已定义，但 144 篇文档均未添加 YAML frontmatter |
| D1 深度内容核验 | 部分完成 | 结构核验已做（§4.4），逐段对照代码的深度核验未做 |
| D3 P1/P2 处置 | 未启动 | 25 篇 designs/ 命名迁移、4 篇 review 重命名、4 篇超长拆分 |
| D4 非核心文档术语迁移 | 未启动 | designs/、guides/、reviews/ 中仍有 Phase 术语 |
| D5b metadata schema | 未启动 | 依赖 D0 frontmatter 落地 |

#### 1.3.3 E1–E9 退出条件状态

| 编号 | 当前结果 | 证据 |
|---|---|---|
| E1 | 部分满足 | 核心 9 篇结构核验完成（§4.4），但缺乏独立可追溯的深度核验记录 |
| E2 | 已满足（修订后） | 核心 9 篇状态列无 Unverified；其余 27 篇治理范围文档保持 Unverified 但均有处置+优先级 |
| E3 | 已满足 | 01 §2.2 Linear/RoPE/Argmax 已更新（2026-09-14） |
| E4 | 已满足 | 3 篇 plan 已归档（2026-09-14） |
| E5 | 已满足 | issues.md 已登记真实问题（2026-09-14） |
| E6 | 已满足 | code_review_guide.md 已标 Deprecated（2026-09-14） |
| E7 | 已满足 | 8 篇计数为 0；issues.md 仅保留自身跟踪条目标题中的 2 处元描述（2026-09-17） |
| E8 | 已满足 | verify_docs.py D5a 全部 7 项实现（2026-09-14） |
| E9 | 已满足 | 清查表 + issues.md 跟踪剩余债务 |

#### 1.3.4 结论

- **Batch -1 implementation substantially complete，final acceptance blocked**；
- 阻塞项：E1 需要补充可追溯的深度核验记录；
- **02 号方案 Batch 0 仍阻塞**，直到 E1 完全满足；
- 本方案状态：`In Progress`（主要实施完成，收口待验收）。

## 2. 当前问题（证据锚定）

以下每条陈述都指向可复核的仓库事实，避免主观判断。

### 2.1 `docs/issues.md` 未反映真实未解决问题

[docs/issues.md](../issues.md) 第 9 行显示未解决条目为 `- （无）`。这与本方案 §2.2–§2.5 列出的已知缺口不一致，也无法作为短生命周期缺陷的权威跟踪源。

### 2.2 `tools/verify_docs.py` 只覆盖有限结构条件

[tools/verify_docs.py](../../tools/verify_docs.py) 报 `0 problems` 不能证明文档事实正确，其覆盖范围明确排除以下内容：

- 第 7 行注释：`anchors are not validated`（不校验 heading anchor）；
- 第 39 行：`EXCLUDED_PARTS = ("templates", "archive", "agent", "improvement-plan")`（改进计划与归档完全不参与检查）；
- 第 111-112 行：symbol 校验只作用于 `docs/designs/<module>/NN-*.md` 命名的少数文档；
- 第 124-134 行：symbol 校验为 loose 匹配——类名与成员名各自独立出现在 `include/` 或 `src/` 中即通过，不要求 `Class::Member` 作为整体存在；
- 第 155-158 行：`check_index` 允许目录级链接覆盖整个子树，掩盖单篇文档未登记的情况。

### 2.3 演进提案与当前实现漂移

[docs/improvement-plan/01-inference-session-generate-readiness.md](01-inference-session-generate-readiness.md) §2.2 表格将 Linear、RoPE、Argmax 标注为"无 kernel / 阻塞"，但仓库中实际存在：

- [src/backend/cpu/kernels/linear/linear_f32_reference.cpp](../../src/backend/cpu/kernels/linear/linear_f32_reference.cpp)
- [src/backend/cpu/kernels/rope/rope_f32_reference.cpp](../../src/backend/cpu/kernels/rope/rope_f32_reference.cpp)
- [src/backend/cpu/kernels/argmax/argmax_f32_reference.cpp](../../src/backend/cpu/kernels/argmax/argmax_f32_reference.cpp)

这是典型的"文档事实落后于实现"，且直接位于 02 号方案 Batch 0 的输入侧。

### 2.4 开发指南含失效命令与专项残留

[docs/guides/code_review_guide.md](../guides/code_review_guide.md) 存在多类问题：

- 第 49 行：`make --build build --target aethermind_unit_tests` 语法错误（应为 `cmake --build build`）；
- 第 52 行：`--gtest_filter="*SizeClass*:*Span*"` 引用 ammalloc 专项测试，但 ammalloc 已移出仓库；
- 第 32、199、216-217、312、482-698 行：37+ 处 `ammalloc / am_malloc / SizeClass / Span` 专项规则，与当前仓库无对应代码；
- 第 26 行：审查时间的硬阈值（5-10 min / 30-60 min / 2-4 hours）无依据支撑；
- 第 55、58 行：`clang-format` 与 `clang-tidy` 收集逻辑与 [AGENTS.md §6](../../AGENTS.md) 不一致。

该文档在 [docs/README.md](../README.md) 中被登记为 `Current`，作为门禁传播会放大错误。

### 2.5 `docs/designs/` 存在分类与命名违规

按 [文档系统规范 §1 准入条件](../guides/documentation-guide.md)："`designs/` 下所有文档只描述已验证实现；草稿/提案不得进入"。当前违规条目：

- [docs/archive/designs-legacy/amstring_development_plan.md](../archive/designs-legacy/amstring_development_plan.md)：development plan 而非设计（已归档 2026-09-14）；
- [docs/archive/designs-legacy/amstring_milestones.md](../archive/designs-legacy/amstring_milestones.md)：milestones 而非设计（已归档 2026-09-14）；
- [docs/archive/designs-legacy/amstring_task_checklist.md](../archive/designs-legacy/amstring_task_checklist.md)：task checklist 而非设计（已归档 2026-09-14）；
- 命名违规统计（分母重新定义）：`docs/designs/` 现有 27 篇文件（原 30 篇，3 篇 plan 已归档），其中 canonical module design 1 篇（`model/01-model-loader.md`）、存量例外 1 篇（`architecture/architecture_overview.md`，见 [documentation-guide.md §2](../guides/documentation-guide.md)）、**真正待迁移的命名违规 25 篇**；
- 中文文件名：`status设计方案.md`、`已证明约束的执行阶段保障方案.md`、`kernel_dev/*.md` 全部；
- 同名冲突：[docs/designs/graph_compilation_flow.md](../designs/graph_compilation_flow.md) 与 [docs/reviews/graph_compilation_flow.md](../reviews/graph_compilation_flow.md) 并存，容易混淆权威来源。

### 2.6 超长文档违反可读性标准

按 [文档系统规范 §4](../guides/documentation-guide.md)："单文档 ≤ 1000 行"。当前超限（非归档）：

- [docs/guides/operator_optimization_guide.md](../guides/operator_optimization_guide.md) 2110 行；
- [docs/designs/model_graph_design.md](../designs/model_graph_design.md) 1706 行；
- [docs/designs/dispatch_design.md](../designs/dispatch_design.md) 1061 行；
- [docs/designs/graph_lowering_design.md](../designs/graph_lowering_design.md) 1005 行。

归档区超限文档（`docs/archive/kernel_dev/*.md` 等）不作为 Batch -1 优先项，仅在清查表登记。

### 2.7 大量文档无明确状态

`docs/designs/`、`docs/guides/`、`docs/reviews/` 中多数文档头部无 `状态: Current / Draft / Historical Snapshot / Superseded / Deprecated` 字段，或状态与内容不匹配（例如 code_review_guide.md 标 Current 但含失效命令）。无法从文档本身判定当前性。

### 2.8 阶段性术语散布

`Phase 1 / Phase 2` 仍出现在部分设计、评审、指南和测试记录中。2026-09-16 已完成 PRD、AGENTS.md、根 README、文档索引、架构总览、public API 与相关 active improvement plans 的首批迁移；其余存量文档必须区分当前规范与历史快照，不能机械替换。02 号方案 §1.1 已确立 capability-driven 原则。

## 3. D0：冻结文档角色

**先确定文档类别与事实权威，不改写正文。**

### 3.1 事实 → 权威来源映射

| 事实 | 权威来源 | 备注 |
|---|---|---|
| 当前产品能力与范围 | [docs/products/aethermind_prd.md](../products/aethermind_prd.md) | 唯一权威 |
| 模块 ownership 与依赖 | [AGENTS.md](../../AGENTS.md) §2.1 | 设计文档引用不复制 |
| 当前架构 | [docs/designs/architecture/architecture_overview.md](../designs/architecture/architecture_overview.md) | 全系统唯一权威总览 |
| Public API 语义 | 头文件 Doxygen + [docs/api/public-api.md](../api/public-api.md) | 同源同步 |
| 构建选项与运行配置 | 根 [README.md](../../README.md) + CMake | 单一事实源 |
| 未来工作与演进提案 | [docs/improvement-plan/](../improvement-plan/) | 编号专题 |
| 已知短生命周期缺陷 | [docs/issues.md](../issues.md) | 每条目 `[x]`/`[ ]` |
| 历史架构决策 | [docs/decisions/](../decisions/) | ADR，编号全局只增 |
| 历史审查结果 | [docs/reviews/](../reviews/) | 必须注明快照日期 |
| 文档分类与命名规范 | [docs/guides/documentation-guide.md](../guides/documentation-guide.md) | 元规范 |

### 3.2 状态取值（清查表工作标签）

以下状态取值用于清查表"状态"列，作为迁移期间的工作标签。文档头部的正式状态字段仍遵循 [documentation-guide.md §1](../guides/documentation-guide.md) 的按类型定义。

| 清查表标签 | 含义 | 对应 documentation-guide.md 状态 |
|---|---|---|
| `Current` | 内容与当前代码事实一致 | 各类型的 `Current` |
| `Draft` | 内容尚未稳定 | 演进提案的 `Draft` |
| `Historical Snapshot` | 特定时间点的事实快照 | 评审报告的 `历史快照` |
| `Superseded` | 已被后续文档取代 | 演进提案/决策记录的 `Superseded` |
| `Deprecated` | 内容失效 | 各类型的 `Deprecated` |
| `Unverified` | 尚未核验，不能判定当前性 | 无对应（清查表专用） |
| `Out of Scope` | 独立子系统，不参与本治理 | 无对应（仅用于 `docs/agent/`） |

**规范变更需求**：若要将 `Historical Snapshot`、`Unverified` 或 `Out of Scope` 正式引入 documentation-guide.md 的状态枚举，需先通过 ADR 或直接修订该规范。在接纳前，这些标签仅在清查表内部使用。

### 3.3 硬约束

- **无法证明当前性的文档默认不能标为 `Current`**；未核验时统一填 `Unverified`（清查表用）或 `Draft`（文档头部用）；
- `designs/` 中的 `Draft` 内容必须迁移到 `improvement-plan/` 或标记为调研备忘；
- `reviews/` 中的 `Historical Snapshot` 必须在头部注明快照日期；
- `Current` 状态的文档在事实变化时必须同 PR 更新（[文档系统规范 §6.1](../guides/documentation-guide.md)）。

## 4. D1：可信核心文档集

优先核验，而不是全面重写。以下 9 篇文档在 Batch -1 退出前必须完成人工审查：

| 编号 | 文档 | 角色 |
|---|---|---|
| 1 | [AGENTS.md](../../AGENTS.md) | 仓库级执行指南、模块 ownership |
| 2 | [README.md](../../README.md) | 构建选项与运行配置 |
| 3 | [docs/README.md](../README.md) | 文档唯一入口、索引与术语表 |
| 4 | [docs/guides/documentation-guide.md](../guides/documentation-guide.md) | 文档治理唯一规范 |
| 5 | [docs/products/aethermind_prd.md](../products/aethermind_prd.md) | 产品能力与验收标准 |
| 6 | [docs/designs/architecture/architecture_overview.md](../designs/architecture/architecture_overview.md) | 当前架构 |
| 7 | [docs/api/public-api.md](../api/public-api.md) | Public API 汇总 |
| 8 | [docs/issues.md](../issues.md) | 已知问题跟踪 |
| 9 | [docs/improvement-plan/README.md](README.md) | 演进提案索引 |

**排除**：02 号方案与 03 号方案（本文档）均为 `improvement-plan/` 下的 Draft 提案，属于"待审提案"，不作为当前事实权威。其正确性在 Batch -1 执行过程中通过 §1.3 快照跟踪，不纳入 D1 核心核验集。

### 4.1 核验维度

每篇文档必须验证：

- 当前能力与计划能力明确区分；
- 模块边界与 [AGENTS.md](../../AGENTS.md) §2.1 一致；
- 构建命令真实可执行（在干净环境或 CI 中至少一次通过）；
- 不再使用阶段编号决定架构；
- 没有已知的事实冲突（与其他核心文档、与代码）；
- 相互之间只引用，不重复定义同一事实。

### 4.2 核验方式

- 人工审查为主，`tools/verify_docs.py` 只能证明结构不变量；
- 核验记录写入清查表 "当前性" 列（`已核验 / 部分核验 / 未核验`）；
- 发现的事实冲突记入 [docs/issues.md](../issues.md) 或本方案 §2 增补。

### 4.3 已知需修复项（首批）

- [docs/issues.md](../issues.md)：从 `- （无）` 更新为反映 §2.3–§2.8 列出的已知缺口；
- [docs/improvement-plan/01-inference-session-generate-readiness.md](01-inference-session-generate-readiness.md) §2.2：Linear、RoPE、Argmax 三行改为反映当前实现状态；
- [docs/guides/code_review_guide.md](../guides/code_review_guide.md)：§2.4 列出的失效命令与 ammalloc 专项残留必须在 Batch -1 内修复或标为 `Deprecated`；
- [docs/README.md](../README.md)：术语表与索引行的 `Phase 1` 表述按 §7 D4 规则处理。

### 4.4 D1 核验记录（2026-09-14/17，修正于 2026-09-17）

以下为核心 9 篇的结构核验记录。每篇按 §4.1 的 6 个维度检查，记录实际执行的验证动作与结果。**本轮为结构核验（目录存在性、链接有效性、依赖规则、构建 configure、关键事实抽检），不是逐段对照代码的深度内容核验。**

| 文档 | 核验动作 | 结果 | 残留风险 |
|---|---|---|---|
| AGENTS.md | §2.1 模块表 16 个目录存在性检查；5 条依赖约束 include 抽检（operators→shape_inference、graph↛compiler/execution/backend/model、runtime↛execution/compiler/graph/model、compiler↛execution/backend/runtime、backend↛graph/model） | 全部通过，无违规 | 未逐行核验 §3–§13 文本与代码一致性 |
| README.md | `cmake -S . -B /tmp/am_verify_build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` 实际执行 | configure 成功（19.1s），AVX2/FMA 检测通过 | **未执行 `cmake --build` 和单元测试**；第 113–157 行的 build/test/benchmark/TSAN 命令未验证 |
| docs/README.md | verify_docs.py 链接检查通过；03–06 条目已登记；Phase 术语已迁移（3→0） | 通过 | 术语表内容未逐项对照代码 |
| documentation-guide.md | 全文审阅（161 行）；内部自洽性检查；§1 分类表与实际目录结构对照 | 规则清晰、自洽 | 执行落地率未量化 |
| aethermind_prd.md | 第 145 行 kernel 覆盖结论抽检："现有 kernel 覆盖不能闭环真实 Llama Prefill→Decode"——当前仅 Attention 缺失（SiluMul、KVCacheUpdate 已实现），结论仍成立；Phase 术语已迁移（31→0） | 结论正确 | 未逐条核验 §3–§7 验收标准与代码一致性 |
| architecture_overview.md | 模块地图 vs `src/` 目录对照（9 个核心模块全部提及且存在）；Phase 术语已迁移（27→0） | 通过 | 未逐节核验数据流图与实现一致性 |
| public-api.md | `am_session_generate` vs `include/c_api.h` 对照；确认文档正确标注"尚未实现"；Phase 术语计数 0（已全部迁移） | 无漂移 | 未逐个核验 C++ 构建块签名与头文件一致性 |
| issues.md | 本轮更新后 10 条未解决 + 8 条已解决；链接有效性由 verify_docs.py 确认 | 通过 | 无 |
| improvement-plan/README.md | 6 篇文件存在性检查（01–06）；索引完整性 | 通过 | 无 |

**核验级别与 E1 差距说明**：

以上为结构核验。E1 要求"可信核心文档集已经核验"，其充分条件尚未明确定义。当前残留差距：

1. README.md 的 build/test 命令未实际执行（仅 configure）；
2. PRD 验收标准未逐条对照代码；
3. architecture_overview 数据流图未逐节核验；
4. public-api C++ 构建块签名未逐个对照头文件；
5. AGENTS.md §3–§13 文本未逐行核验。

E1 完全满足需要：(a) 补充上述深度核验，或 (b) 明确裁决结构核验 + 本记录即为 Batch -1 的充分条件，深度核验作为后续持续治理项。

## 5. D2：清查表规范

清查表本体：[03-documentation-stabilization-inventory.md](03-documentation-stabilization-inventory.md)。

### 5.1 字段

严格 8 列：

| 字段 | 内容 |
|---|---|
| 路径 | 相对仓库根的路径 |
| 类型 | design / guide / plan / review / test / archive / index / api / prd / handoff / memory / template / decision / log |
| 状态 | Current / Draft / Historical Snapshot / Superseded / Deprecated / Unverified |
| 事实权威 | 是 / 否；若是，注明对应 §3.1 表中哪一类事实 |
| 当前性 | 已核验 / 部分核验 / 未核验 |
| 问题 | 漂移 / 重复 / 分类错误 / 过长 / 术语过时 / 命名违规 / 失效命令 / 无 frontmatter / `-` |
| 处置 | 保留 / 修订 / 拆分 / 移动 / 归档 / 待 D3 排序确认 |
| 优先级 | P0 / P1 / P2（按 §6 D3 映射） |

### 5.2 清查范围

- **包含（治理范围）**：仓库根 `README.md`、`AGENTS.md`、`CHANGELOG.md`；`docs/` 下全部 .md（含 `archive/`、`improvement-plan/`、`templates/`）；
- **包含（仅登记，Out of Scope）**：`docs/agent/` 下全部 .md——按 [documentation-guide.md §1 准入条件](../guides/documentation-guide.md)，`docs/agent/` 是独立子系统，本规范不适用于其内部文档。清查表为全景统计列出这些文件，但**排除在 E1–E9、状态迁移、命名和 metadata 门禁之外**。唯一交叉点是 ADR 互链（documentation-guide.md §3.3）；
- **排除**：`3rdparty/`、`build*/`、`.models/`、`.qoder/`、`.trae/`、`.omo/`、`.sisyphus/`、`.codex/`、`.gemini/`、`.idea/`、`.vscode/`、`.cache/`、`.agents/`、`tools/agent_memory/`（外部代码、构建产物、工具元数据、历史草稿）。

### 5.3 硬约束

**inventory 阶段不批量重命名或移动文件**——避免破坏大量交叉引用。所有"移动 / 归档 / 拆分"处置只登记在清查表，实际执行放到 D3 及后续 workstream。

### 5.4 首次填写规则

- "状态" 未核验时统一填 `Unverified`（不填 `Current`，遵守 §3.3 硬约束）；
- "事实权威" 只对 §3.1 表中列出的 10 类事实来源填"是"，其余填"否"；
- "当前性" 首次填 `未核验`；
- "问题" 依据 §2 已核验事实与行数、命名、重复等结构化证据填写，无问题填 `-`；
- "处置" 首次给出建议值并注明"待 D3 排序确认"；
- "优先级" 按 §6 D3 8 级映射到 P0 / P1 / P2。

## 6. D3：风险优先整理顺序

按风险而非目录顺序处置。以下 8 级从高到低：

| 级别 | 处置对象 | 优先级 |
|---|---|---|
| 1 | 错误描述当前实现的文档 | P0 |
| 2 | 被开发者当作执行规范的 guide | P0 |
| 3 | architecture / API / build 等权威事实源 | P0 |
| 4 | 名为 design、实际是未来计划的文档 | P1 |
| 5 | 未注明日期和状态的 review | P1 |
| 6 | 重复或互相冲突的文档 | P1 |
| 7 | 超长文档和命名问题 | P2 |
| 8 | 纯历史材料 | P2 |

**元规则**：内容正确但格式旧的文档优先级低于内容已经错误的文档。

### 6.1 P0 首批对象（依据 §2 证据）

- [docs/improvement-plan/01-inference-session-generate-readiness.md](01-inference-session-generate-readiness.md) §2.2（错误描述当前实现）；
- [docs/guides/code_review_guide.md](../guides/code_review_guide.md)（失效命令 + ammalloc 残留 + 被当作执行规范）；
- [docs/issues.md](../issues.md)（未反映真实问题，权威事实源）；
- [docs/README.md](../README.md)（索引与术语表，权威事实源）；
- [docs/designs/architecture/architecture_overview.md](../designs/architecture/architecture_overview.md)（当前架构权威）。

### 6.2 P1 首批对象

- `docs/archive/designs-legacy/amstring_development_plan.md`、`amstring_milestones.md`、`amstring_task_checklist.md`（名为 design、实际是计划；已归档 2026-09-14）；
- [docs/reviews/graph_compilation_flow.md](../reviews/graph_compilation_flow.md)、`model_graph_data_structure_review.md`、`operator_semantic_layer_review.md`、`prd/prd_review.md`（未注明日期）；
- [docs/designs/graph_compilation_flow.md](../designs/graph_compilation_flow.md) 与 reviews 同名冲突。

### 6.3 P2 首批对象

- [docs/guides/operator_optimization_guide.md](../guides/operator_optimization_guide.md) 2110 行、[docs/designs/model_graph_design.md](../designs/model_graph_design.md) 1706 行、`dispatch_design.md` 1061 行、`graph_lowering_design.md` 1005 行（超长，内容正确性待核验后决定是否拆分）；
- 中文文件名、非 canonical 命名（待 D3 排序后统一处理）。

## 7. D4：阶段术语弱化

**不进行简单的全局替换。** 每处阶段性表述分成三类处理：

| 类别 | 处理 |
|---|---|
| 历史记录 | 保留，并明确是历史语境（例如 `reviews/` 快照、`archive/` 文档、`logs/`） |
| 当前范围 | 改成具体 capability，例如"本地 CPU"、"同步单请求"、"Llama family"、"Token ID 接口"、"FP32 correctness baseline" |
| 未来规划 | 改成具体能力及其准入条件，例如"服务端调度（准入：本地 Generate 端到端通过 + public API 稳定）" |

### 7.1 硬约束

**类型名、API、模块名、里程碑和质量门禁不得再引入新的 `Phase1 / Phase2` 命名。** 该约束与 [02 号方案 §1.1](02-engineering-quality-system.md) 一致。

### 7.2 迁移顺序

1. D1 可信核心 9 篇（§4）优先；
2. 权威事实源（architecture_overview、public-api、README、AGENTS）；
3. guides/ 与 improvement-plan/；
4. designs/ 与 reviews/；
5. archive/ 保留历史语境，不改写。

### 7.3 术语迁移审核

阶段性术语迁移应单独审核，避免改变仍然有效的产品范围约束（[02 号方案 §10](02-engineering-quality-system.md)）。术语弱化不得被解释为扩大当前实现范围。

## 8. D5：校验工具演进

`tools/verify_docs.py` 逐步扩展。**工具永远不宣称"文档事实正确"；事实核验必须由代码审查完成。**

### 8.1 D5a：Batch -1 内可完成

- 取消 `improvement-plan/` 在链接与索引检查中的排除；
- 增加 heading anchor 校验（当前第 7 行显式声明不校验）；
- 增加状态值合法性检查（针对已加 frontmatter 的文档，取值必须在 §3.2 枚举内）；
- `designs/` 中 Draft / 计划类内容 report-only 诊断（依据文件名与头部状态）；
- canonical `NN-*.md` 命名覆盖率报告（`docs/designs/<module>/` 下）；
- README 与子索引一致性检查（`docs/README.md` 与各子目录 `README.md`）；
- public API symbol 精确定位（`Class::Member` 作为整体匹配，替换第 128-134 行的 loose 检查）。

新增检查默认以 **report-only** 模式运行（输出诊断但不影响退出码）。通过 `--strict-<check>` 将相应诊断升级为失败（非零退出码）。完成存量治理后再将对应检查改为默认 gate。保留 `--no-*` 仅用于临时关闭已默认启用的检查。

### 8.2 D5b：后续（需先落地 metadata schema）

- 文档类型对应的必需 metadata 校验（例如 `reviews/` 必须有 `snapshot_date`）；
- 重复事实报告（同一事实在多篇文档中被定义而非引用）；
- 阶段性术语 report-only 诊断（`Phase 1 / Phase 2` 出现位置与上下文分类）。

D5b 依赖 D1 核心 9 篇先落地 frontmatter，再扩展到其余文档。

### 8.3 工具能力边界

`verify_docs.py` 能证明的：

- 相对链接指向存在的文件或目录；
- heading anchor 指向存在的章节；
- 文档头部 metadata 符合 schema；
- 命名符合 canonical pattern；
- 索引覆盖完整。

`verify_docs.py` 不能证明的：

- 文档内容与代码一致；
- 构建命令可执行；
- 事实权威映射正确；
- 阶段术语迁移完成。

以上必须由 D1 人工核验与代码审查承担。

## 9. 退出条件

Batch -1 完成、02 号方案 Batch 0 可启动的判定条件：

| 编号 | 退出条件 | 可验证锚点 |
|---|---|---|
| E1 | 可信核心文档集已经核验 | 清查表中 §4 列出的 9 篇文档 "当前性" 列全部为 `已核验` |
| E2 | 可信核心状态明确，其余有处置跟踪 | 核心 9 篇（§4）状态列无 `Unverified`；其余治理范围文档允许保持 `Unverified`，前提是清查表"处置"与"优先级"列有值（`archive/` 与 `docs/agent/` OOS 除外） |
| E3 | 已知错误的当前实现描述已修复或显著标记 | [01 号方案 §2.2](01-inference-session-generate-readiness.md) Linear/RoPE/Argmax 三行已更新 |
| E4 | 未来计划不再放在 `designs/` 中冒充现状 | `docs/designs/` 中不再有 `development_plan / milestones / task_checklist` 命名的文件，或已迁移到 `improvement-plan/` |
| E5 | `docs/issues.md` 能反映真实未解决问题 | 未解决节至少列出 §2.3–§2.8 中未在本 Batch 内修复的条目 |
| E6 | `code_review_guide.md` 不再作为错误门禁传播 | 该文档头部标 `Deprecated` 或已修订；`docs/README.md` 索引状态同步 |
| E7 | 阶段编号不再决定架构和里程碑 | D1 核心 9 篇中不再出现 `Phase 1 / Phase 2` 作为架构、API、模块、里程碑或门禁命名依据 |
| E8 | 文档校验覆盖改进计划和 metadata | `verify_docs.py` 已实现 §8.1 D5a 全部项，`improvement-plan/` 不再在 `EXCLUDED_PARTS` 中 |
| E9 | 剩余文档债务有明确清单，可以渐进处理 | 清查表 "处置" 与 "优先级" 列全部有值；P1 / P2 条目进入后续 workstream 计划 |

**不要求全部文档达到最终形态。** 满足 E1–E9 即可启动 02 号方案 Batch 0 及后续实施。

## 10. 与 02 号方案的接口

### 10.1 依赖顺序

```text
Batch -1  文档系统稳定化（本方案）
    ↓
Batch 0   事实与风险基线（02 号方案）
    ↓
Batch 1   最小工具链闭环
    ↓
Batch 2   质量 vertical slice
    ↓
后续核心链路与发布门禁
```

02 号方案 Batch 0 首行声明依赖："Batch -1 退出条件 E1–E9 全部满足后方可启动"。

### 10.2 职责划分

| 内容 | 归属 |
|---|---|
| 文档角色冻结、清查表、状态与类型规范 | 本方案（03） |
| 阶段术语弱化规则与迁移顺序 | 本方案（03） |
| `verify_docs.py` 演进 | 本方案（03） |
| 代码事实基线（capability 清单、kernel 覆盖、API 现状） | 02 号方案 Workstream A |
| P0 / P1 风险台账 | 02 号方案 §4 |
| 验证 profile 与门禁矩阵 | 02 号方案 §5 |
| Vertical slice 与核心链路 assurance | 02 号方案 Workstream C |

02 号方案 §6.1 Workstream A 中原有的文档校准、阶段性术语迁移清单、`docs/issues.md` 校准等条目移交本方案；Workstream A 更名为 "代码事实基线与风险台账"。

### 10.3 03 号方案自身状态

- 本方案当前状态为 `In Progress`（主要实施完成，收口待验收）；
- E1 完全满足后，状态迁移到 `Implemented`；
- 后续若文档治理规则需要长期约束，另建 ADR，本方案转为 `Superseded` 并保留链接。

## 11. 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| 1.0 | 2026-09-13 | 初始版本：Batch -1 前置里程碑，D0–D5 与 E1–E9 退出条件 |
| 1.1 | 2026-09-14 | 新增 §1.3 "当前执行状态"快照，明确区分规划工件交付与 D0–D5 实质执行；E1–E8 当前失败，E9 部分完成；02 号方案 Batch 0 阻塞中 |
| 1.2 | 2026-09-14 | 修正 6 条自相矛盾：§1.1 明确从属 documentation-guide.md；§3.2 改为清查表工作标签；§4 核心集替换为 9 篇权威文档（排除 02/03 Draft 提案）；§5.1 移除 authority 类型；§5.2 docs/agent/ 标记 Out of Scope；§8.1 修正开关语义为 report-only + --strict-*；§9 E2 排除 agent/；§2.5 命名违规计数修正为 25 篇 |
| 2.0 | 2026-09-17 | E1–E8 实施完成；状态 Draft → In Progress；E2 修订为“核心无 Unverified + 其余有处置跟踪”；回退未经核验的 designs/ 批量 Current 赋值 |
| 2.1 | 2026-09-17 | 修正 §1.3 内部矛盾（D1/D3/D5a 实际已执行但快照写“未启动”）；状态回退 Implemented → In Progress；Batch 0 重新标记为阻塞；补充 §4.4 D1 核验记录 |
