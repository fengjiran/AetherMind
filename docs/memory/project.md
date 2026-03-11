---
scope: global
module: project
parent: none
depends_on: []
adr_refs: []
last_verified: 2026-03-10
owner: team
status: active
---

# 项目级记忆

> 用途：记录跨模块稳定事实与默认约束。代理执行细则仍以 `AGENTS.md` 为准。

## 模块范围
### 职责
- 负责：记录项目目标、目录约定、构建与验证基线，以及跨模块协作规则。
- 不负责：单个模块实现细节、一次性会话状态、未验证讨论。

### 边界
- 上游输入：`AGENTS.md`、`docs/aethermind_prd.md`、已验证代码/CMake/测试事实。
- 下游输出：模块记忆、子模块记忆、模块 ADR 与 handoff 的共识基线。
- 不直接管理：模块内部私有接口、临时调试记录、局部试验方案。

### 子模块划分
- `include/` + `src/`：核心运行时公共接口与实现。
- `ammalloc/`：内存池子系统及其独立实现。
- `tests/unit/` + `tests/benchmark/`：单元测试与性能基准入口。
- `docs/`：架构文档、memory 体系与提示词。

## 已确认事实
- 项目当前阶段是 Phase 1：CPU 嵌入式推理运行时，目标模型族为 Llama，主接口形态为 C/C++ API；产品边界见 `docs/aethermind_prd.md`。
- 语言与构建基线：C++20、CMake >= 3.28；核心库目标为 `AetherMind`，内存池目标为 `ammalloc`。
- 验证入口：`aethermind_unit_tests` 使用 GoogleTest，`aethermind_benchmark` 使用 Google Benchmark。
- 目录约定以 `AGENTS.md` 为索引：公共头文件放在 `include/`，核心实现放在 `src/`，第三方代码放在 `3rdparty/` 且默认不改。
- 稳定记忆统一放在 `docs/memory/`；会话交接统一遵循 `docs/prompts/handoff.md`，并放在`docs/handoff/`。

## 核心抽象
### 关键抽象
- `AetherMind`：项目主共享库，承载推理运行时对外能力。
- `ammalloc`：项目独立内存池子系统，需单独关注所有权与性能约束。
- `memory hierarchy`：`project.md`、`module.md`、`submodule.md` 与 handoff 组成的长期上下文体系。

### 数据流
- 输入：用户需求、PRD、已验证代码事实与既有设计文档。
- 转换：代码实现、聚焦构建/测试/benchmark、必要时补充 ADR。
- 输出：可执行代码、已验证结论、记忆文档增量与会话 handoff。
- 未涉及：无。

## 对外接口
- 公共 API 头文件：`include/`。
- 内存池公共头文件：`ammalloc/include/ammalloc/`。
- 构建与验证入口：`AetherMind`、`ammalloc`、`aethermind_unit_tests`、`aethermind_benchmark`。
- 详细命令与非默认配置要求：见 `AGENTS.md` 的构建、测试、benchmark 与 TSAN 章节。

## 不变量
- 已验证代码事实与用户显式指令优先于任何 memory、prompt 或 ADR 草案。
- 跨模块稳定结论写入 `docs/memory/`；handoff 只保留当前会话状态，不充当长期事实源。
- 公共接口与实现边界保持清晰：公共 API 进入头文件目录，测试与 benchmark 不混写。
- 性能敏感路径必须显式避免隐藏 `O(N^2)`、不必要分配、复制和锁竞争。

## 所有权与生命周期
- 项目默认遵循 RAII、标准库设施和显式所有权；避免 raw owning pointer。
- 跨模块传递对象时，必须让拥有者和借用者可从接口语义中直接判断。
- 稳定结论先验证再回写 memory；不要把生命周期假设直接留在 handoff 中长期漂移。
- 细则见 `docs/cpp_coding_style_guidelines.md` 与 `docs/cpp_comment_guidelines.md`。

## 并发约束
- 若模块存在线程亲和、锁顺序、原子语义或可重入性要求，必须下沉到对应模块或子模块记忆中维护。
- 改动并发相关宏、构建选项或同步策略时，至少补做一次相关的非默认验证；可选项优先参考 TSAN 流程。
- 没有明确并发契约前，不把对象默认视为可跨线程共享。
- 未涉及：无。

## 性能约束
- 热路径优先约束常数项、分配次数、复制次数和锁竞争，而不是抽象上的“可能优化”。
- 内存池或性能敏感改动需要运行聚焦 benchmark；普通改动先跑最小相关构建与测试。
- 验证顺序遵循“先最小目标，再扩大范围”；没有执行的验证在 handoff 中必须明确标记。
- 未涉及：无。

## 已否决方案
- 方案：把项目级事实、模块细节和 handoff 混写在同一记忆文件。
  - 原因：会扩大冲突面，降低可验证性，并让模块边界失真。
- 无。

## 未决问题
- 哪些活跃能力域应优先补齐独立 `module.md`，仍需结合后续任务密度逐步确定。
- 无。

## 待办事项
- [ ] 为高频修改模块建立 `docs/memory/modules/<module>/module.md`。
- [ ] 当主模块内部出现独立并发或性能边界时，再拆分 `submodules/<submodule>.md`。
- 无。
