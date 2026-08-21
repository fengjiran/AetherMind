# AetherMind 文档索引

本文档是 AetherMind 项目文档的唯一入口。文档分类体系、命名与交叉引用规则、质量标准与维护流程见 [文档系统规范](guides/documentation-guide.md)；新建文档从 [文档模板](templates/module-design.md) 复制填充。

> 迁移状态：本索引按 [文档系统设计方案](guides/documentation-guide.md) 落地；`docs/designs/` 存量文档正处于渐进式迁移中，未完成迁移的文档在清单中标注"待迁移"。

## 文档分类总览

| 类型 | 目录 | 状态字段 | 职责边界 |
|---|---|---|---|
| 架构总览 | [designs/architecture/architecture_overview.md](designs/architecture/architecture_overview.md) | Current | 全系统唯一权威总览：分层架构、硬性约束、跨层流程、模块地图、性能基线 |
| 模块设计 | [designs/](designs/) | Current / Deprecated | 单模块深入设计：数据结构、并发模型、接口、边界、权衡；按 AGENTS.md §2.1 模块表分子目录 |
| 调研备忘 | [designs/research/](designs/research/) | 无状态，头部标注日期与可信度 | 未验证/未采纳的技术调研，不承诺实现，禁止当作事实引用 |
| 演进提案 | [improvement-plan/](improvement-plan/) | Draft / In Progress / Implemented / Superseded | 未来要做什么：路线图与专题提案 |
| 开发指南 | [guides/](guides/) | Current / Deprecated | "怎么做"的过程规范：编码、注释、测试、评审、文档 |
| API 参考 | [api/public-api.md](api/public-api.md) | Current | 公共 API 语义（与头文件 Doxygen 同源同步） |
| 决策记录 | [decisions/](decisions/) | Proposed / Accepted / Deprecated / Superseded | 已作出/被否决的架构决策及其理由（ADR），每篇至少 1 个被否定备选方案 |
| 评审报告 | [reviews/](reviews/) | Current / 历史快照 | 代码/设计评审结论；过时快照头部加"已过时"警示 |
| 验证报告 | [tests/](tests/) | Current | 测试/基准验证结果与结论 |
| 问题跟踪 | [issues.md](issues.md) | 每条目 `[x]`/`[ ]` | 已知缺陷与优化待办，短生命周期 |
| 变更记录 | [../CHANGELOG.md](../CHANGELOG.md) | 无 | 行为可见变更，按语义化版本 |
| 开发日志 | [logs/development_log.md](logs/development_log.md) | 无 | 过程记录（开发黑匣子），记录"为什么"而非"做了什么" |
| 产品需求 | [products/aethermind_prd.md](products/aethermind_prd.md) | Current | Phase 1 产品范围、验收标准（唯一权威） |

**职责边界判定**：描述"代码里现在是什么" → `designs/`；"将来要做什么" → `improvement-plan/`；"曾经怎么决策的" → `decisions/`；"怎么干活" → `guides/`；"API 怎么用" → `api/`；"已知缺陷/待办" → `issues.md`。

**独立子系统**：`docs/agent/`（Agent 记忆与 handoff 系统）不并入本体系，不参与编号与索引；与人工文档的唯一交叉点是决策记录（ADR 双向链接，见 [文档系统规范 §3.3](guides/documentation-guide.md#33-单一事实源与-docsagent-互链)）。

## 文档清单

### 架构与模块设计（docs/designs/）

#### 架构总览

| 文档 | 定位 | 状态 | 最后更新 |
|---|---|---|---|
| [architecture/architecture_overview.md](designs/architecture/architecture_overview.md) | 全系统唯一权威总览（六层架构、依赖规则、产物所有权、Phase 1 边界） | Current | 2026-08-21 |

#### 模块设计（按模块子目录）

| 文档 | 定位 | 状态 |
|---|---|---|
| [model/01-model-loader.md](designs/model/01-model-loader.md) | ModelLoader 模块：HF I/O/校验/权重 resolve（P3 新建，基于当前头文件） | Current |
| [amstring/](designs/amstring/) | amstring 字符串库模块设计（存量，待按 NN- 编号重排） | 待迁移 |
| [kernel_dev/](designs/kernel_dev/) | 算子契约（LinearOp/RMSNorm 契约、算子系统设计；存量，待归位 operators/backend） | 待迁移 |
| [graph_compilation_flow.md](designs/graph_compilation_flow.md) | 图编译流程追踪（存量，待归位 compiler/） | 待迁移 |
| [graph_lowering_design.md](designs/graph_lowering_design.md) | 图降低设计（存量，待归位 compiler/） | 待迁移 |
| [graph_invariants_and_validator_architecture.md](designs/graph_invariants_and_validator_architecture.md) | 图不变量与验证器架构（存量，待归位 graph/） | 待迁移 |
| [backend_design.md](designs/backend_design.md) | Backend 层设计（存量，待归位 backend/） | 待迁移 |
| [dispatch_design.md](designs/dispatch_design.md) | Dispatch/KernelRegistry 设计（存量，待归位 backend/） | 待迁移 |
| [executor_design.md](designs/executor_design.md) | Executor 设计（存量，待归位 execution/） | 待迁移 |
| [kv_cache_design.md](designs/kv_cache_design.md) | KV Cache 设计（存量，待归位 execution/） | 待迁移 |
| [op_evaluator.md](designs/op_evaluator.md) | 算子求值设计（存量，待归位 operators/） | 待迁移 |
| [operator_contract_design.md](designs/operator_contract_design.md) | 算子契约设计（存量，待归位 operators/） | 待迁移 |
| [model_graph_design.md](designs/model_graph_design.md) | ModelGraph 设计（存量，待归位 graph/） | 待迁移 |
| [tensor_view_design.md](designs/tensor_view_design.md) | TensorView 设计（存量，待归位 base/） | 待迁移 |
| [unified_allocator_design.md](designs/unified_allocator_design.md) | 统一分配器设计（存量，待归位 memory/） | 待迁移 |
| [status设计方案.md](designs/status设计方案.md) | Status 错误模型设计（存量，待归位 base/） | 待迁移 |
| [已证明约束的执行阶段保障方案.md](designs/已证明约束的执行阶段保障方案.md) | 形状约束证明与执行期保障（存量，待归位 compiler/或 shape_inference/） | 待迁移 |

#### 历史与过期文档（docs/archive/）

| 文档 | 原位置 | 状态 |
|---|---|---|
| [aethermind_arch_design.md](archive/aethermind_arch_design.md) | `docs/designs/` | Deprecated（有效内容并入架构总览） |
| [model_loader/](archive/model_loader/)（4 篇） | `docs/designs/model_loader/` | Deprecated（历史快照，已被当前实现取代） |
| [kernel_dev/](archive/kernel_dev/)（8 篇） | `docs/designs/kernel_dev/` | Deprecated（历史方案/审查记录；契约类保留原位） |
| [designs-legacy/](archive/designs-legacy/)（14 篇） | `docs/designs/` 顶层 | Deprecated（历史计划/评审快照） |
| [archive/](archive/) | 归档区索引见 archive/README.md | Deprecated |

### 演进提案（docs/improvement-plan/）

待建立。存量计划类文档（phase1_*、backend_phase1_*、路线图等）迁移后在此登记。

### 开发指南（docs/guides/）

| 文档 | 定位 | 状态 |
|---|---|---|
| [documentation-guide.md](guides/documentation-guide.md) | 文档系统规范：命名、交叉引用、质量、维护流程 | Current |
| [operator_optimization_guide.md](guides/operator_optimization_guide.md) | 算子优化指南（由 kernel_dev/算子开发指南.md 归位） | Current |
| [cpp_coding_style_guidelines.md](guides/cpp_coding_style_guidelines.md) | C++ 编码风格 | Current |
| [cpp_comment_guidelines.md](guides/cpp_comment_guidelines.md) | 注释与 Doxygen 规范 | Current |
| [test_writing_guidelines.md](guides/test_writing_guidelines.md) | GoogleTest 测试编写规范 | Current |
| [code_review_guide.md](guides/code_review_guide.md) | 代码审查方法（风险分级驱动） | Current |

### 架构决策记录（docs/decisions/）

| 文档 | 决策内容 | 状态 |
|---|---|---|
| [0001-documentation-system.md](decisions/0001-documentation-system.md) | 文档系统落地：分类/模板/命名/质量/维护规范与存量渐进式迁移 | Accepted |

### API 参考（docs/api/）

- [api/public-api.md](api/public-api.md)：公共 API（C ABI + 公开 C++ 构建块）语义参考，与头文件 Doxygen 同源同步。

### 评审报告（docs/reviews/）

| 文档 | 定位 | 状态 |
|---|---|---|
| [graph_compilation_flow.md](reviews/graph_compilation_flow.md) | 图编译功能历史审查快照（已过时，头部有警示） | 历史快照 |
| [model_graph_data_structure_review.md](reviews/model_graph_data_structure_review.md) | ModelGraph 数据结构评审 | Current |
| [operator_kernel_architecture_review_2026-07-18.md](reviews/operator_kernel_architecture_review_2026-07-18.md) | 算子内核架构评审 | Current |
| [operator_semantic_layer_review.md](reviews/operator_semantic_layer_review.md) | 算子语义层评审 | Current |
| [prd/prd_review.md](reviews/prd/prd_review.md) | PRD 评审 | Current |

### 验证报告（docs/tests/）

| 文档 | 定位 | 状态 |
|---|---|---|
| [ammalloc_benchmark_rigorous_20260303.md](tests/ammalloc_benchmark_rigorous_20260303.md) | ammalloc 基准严格验证 | Current |
| [size_class_benchmark_20260310.md](tests/size_class_benchmark_20260310.md) | size_class 基准验证 | Current |
| [amstring_m6_validation_20260428.md](tests/amstring_m6_validation_20260428.md) | amstring M6 里程碑验证 | Current |
| [amstring_charlayout_m7_validation_20260429.md](tests/amstring_charlayout_m7_validation_20260429.md) | amstring CharLayout M7 验证 | Current |
| [amstring_sso_boundary_validation_20260502.md](tests/amstring_sso_boundary_validation_20260502.md) | amstring SSO 边界验证 | Current |

### 问题跟踪（docs/issues.md）

已知缺陷与优化待办清单，条目格式：`- [ ] 背景与方案简述`。

## 术语表

以下术语在全仓库文档中统一使用，含义以本表为准。

| 术语 | 定义 |
|---|---|
| ModelLoader | 模型加载前端：仅负责 HF I/O、校验与权重 resolve，返回 `LoadedModel` |
| LoadedModel | 模型加载产物：config + resolved raw weights + backing storage，构造后只读，由 `LoweredModelArtifact` 持有 |
| ModelGraphBuilder | HF → 语义图唯一转换权威（`BuildLlamaDense`）；显式拒绝 HF-only RoPE scaling types |
| ModelGraph | backend-independent 语义 DAG（graph 模块产物） |
| ModelCompiler | 编译阶段编排：`BuildLlamaDense` → `OptimizeModelGraph` → `LowerModelGraph` |
| OptimizeModelGraph | 语义优化入口（O0/O1/O2+ passes） |
| LowerModelGraph | 语义图降低入口，产出 `LoweredGraph` |
| LoweredGraph | 不可变、结构验证过的编译产物（`LoweredStepSpec[]` + dense value metadata） |
| LoweredModelArtifact | 编译产物容器：owns `LoadedModel` + `LoweredGraph` |
| ExecutionPlan | 不可变执行计划（`ExecutionStep[]`，kernel fn 已 resolve、weight ptr 已绑定、workspace req 已冻结） |
| ExecutionStep | 单步执行描述（kernel 函数指针 + 参数 + 绑定信息） |
| StateAliasPlan | state alias 的 runtime 表示（execution 层，由 LoweredGraph 的 alias 信息转换） |
| ExecutionPlanBuilder | 计划构建器：消费 LoweredGraph，kernel resolve + workspace planning |
| KernelRegistry | 全局内核注册表（singleton + `AM_REGISTER_KERNEL` 静态注册），冻结后不可变 |
| KernelSelector | base 层纯数据契约内核匹配器（backend 据此匹配内核，不依赖 Graph IR 语义细节） |
| ResolvedKernel | plan-build-time 冻结的 kernel 描述（`fn` 函数指针 + 参数） |
| LayerRunner | 逐步骤执行器：workspace 绑定、shape 校验后调用算子 kernel |
| WorkspaceArena | 可复用 workspace buffer（Bind/Reset），Session 生命周期 |
| KVCacheManager | 静态 KV 缓存管理（Init + ReserveForSession） |
| KVCacheView | KV 缓存逻辑访问视图（可复制，底层状态可变） |
| semantic pass | backend-independent 图变换（ConstantFolding / DCE / SiluMul / AddRmsNorm / QkvLinearFusion） |
| GraphRewrite / GraphPassManager | 语义 pass 框架（graph 模块） |
| OpType / OperatorSchema / OpParams | 算子语义契约层：端口顺序为语义 ABI；OpParams 为 typed variant |
| TensorSpec / ShapeSymbol / ShapeConstraint | 形状推导基础设施（shape_inference 模块） |
| PackedWeightStore | packed weights 存储（legacy backend artifact，调用方持有） |
| RuntimeBuilder / RuntimeContext | 运行时装配与上下文（AllocatorRegistry + BackendRegistry + KVCacheManager） |
| ammalloc | 自研用户态分配器（ThreadCache/CentralCache/PageCache） |
| Argmax | 贪婪采样（Phase 1 唯一采样策略） |
