---
title: AetherMind Phase 1 M1 执行清单
status: draft
version: v0.1
date: 2026-03-26
author: OpenCode / Sisyphus
source_documents:
  - docs/designs/phase1_development_plan_draft.md
  - docs/designs/phase1_implementation_breakdown.md
---

# AetherMind Phase 1 M1 执行清单

**状态**: Draft  
**版本**: v0.1  
**日期**: 2026-03-26  
**作者**: OpenCode / Sisyphus

---

## 1. 目标

M1 的目标是冻结核心接口与基础骨架，为后续 Loader、Reference Kernels 与 Executor 提供稳定边界。

M1 不追求端到端推理能力，只追求以下结果：

- contract 明确
- 对象边界明确
- 构建系统可承载后续模块
- 基础测试脚手架建立

---

## 2. 完成标准

M1 完成应满足：

- `Status` / `StatusOr` 可用
- `TensorView` / `MutableTensorView` 可用
- `GenerationConfig` / `OpContext` 已定义
- `RuntimeContext` / `LoadedModel` / `SessionState` / `SessionExecutionState` 骨架已落地
- `DispatchTable` 骨架已定义
- 新目录结构已纳入构建
- 对应单元测试已添加并可运行

---

## 3. 执行清单

### 3.1 预备决策

- [ ] 确认 `Status` / `StatusOr` 作为 Phase 1 主错误返回模型
- [ ] 确认现有 `Error` 保留为异常 / 调试设施，不作为主 contract
- [ ] 确认新增 `include/aethermind/...` 分层目录
- [ ] 确认 Phase 1 初期仍挂在 `AetherMind` 主库，不拆独立 target

### 3.2 目录与文件骨架

- [ ] 创建 `include/aethermind/base/`
- [ ] 创建 `include/aethermind/model/`
- [ ] 创建 `include/aethermind/ops/`
- [ ] 创建 `include/aethermind/execution/`
- [ ] 创建 `include/aethermind/api/`
- [ ] 创建对应 `src/...` 目录或占位实现文件
- [ ] 确认 `src/CMakeLists.txt` 能自动纳入新增源文件

### 3.3 Base Contract

- [ ] 定义 `StatusCode`
- [ ] 定义 `Status`
- [ ] 定义 `StatusOr<T>`
- [ ] 定义 `TensorView`
- [ ] 定义 `MutableTensorView`
- [ ] 定义 `GenerationConfig`
- [ ] 定义 `OpContext`
- [ ] 评估是否需要补充推理侧 shape / quantization 基础枚举

### 3.4 Runtime / Model Skeleton

- [ ] 定义 `ModelConfig` 只读结构
- [ ] 定义 `LoadedModel` 骨架
- [ ] 定义 `RuntimeContext` 骨架
- [ ] 定义 `SessionState` 骨架
- [ ] 定义 `SessionExecutionState` 骨架
- [ ] 定义 `DispatchTable` 骨架
- [ ] 明确各对象所有权与生命周期边界

### 3.5 最低测试脚手架

- [ ] 新增 `test_status.cpp`
- [ ] 新增 `test_tensor_view.cpp`
- [ ] 新增 `test_runtime_contract.cpp`
- [ ] 新增 `test_dispatch_table.cpp`
- [ ] 为每个新测试文件加入最小正向用例

### 3.6 最低验证命令

- [ ] 构建主库：`cmake --build build --target AetherMind -j`
- [ ] 构建单测：`cmake --build build --target aethermind_unit_tests -j`
- [ ] 运行 `Status` 相关测试
- [ ] 运行 `TensorView` 相关测试
- [ ] 运行 Runtime contract 相关测试
- [ ] 运行 DispatchTable 相关测试

---

## 4. 实施顺序建议

建议以如下顺序执行：

1. 先冻结错误模型与目录结构决策
2. 先落 base contract
3. 再落 runtime / model skeleton
4. 最后补测试与构建验证

不要在 M1 阶段提前引入：

- Loader 实现
- Reference kernels 实现
- KV Cache 实现
- Executor 真正执行逻辑
- C ABI 扩展

---

## 5. 风险提示

- 如果 `Status` 与 `Error` 边界不清，后续 Loader 与 C ABI 会返工
- 如果 `TensorView` 设计过重，后续算子热路径可能出现额外开销
- 如果 `SessionState` 与 `SessionExecutionState` 边界不清，M3 会出现状态膨胀
- 如果 `DispatchTable` 在 M1 就耦合过多细节，M2/M3 会难以演进

---

## 6. 建议验收口径

M1 可以视为完成，当且仅当：

- 新增 contract 与 skeleton 均已落地
- 头文件边界清晰
- 构建通过
- 新增测试可独立运行
- 没有为了先通过而引入临时性、不可维护接口
