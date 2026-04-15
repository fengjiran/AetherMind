# AetherMind Backend Phase 1 开发步骤文档

**版本**: v1.0  
**日期**: 2026-04-13  
**作者**: AetherMind Team

---

## 目录

1. [文档目的与范围](#1-文档目的与范围)
2. [输入设计与约束来源](#2-输入设计与约束来源)
3. [当前基线](#3-当前基线)
4. [总体开发原则](#4-总体开发原则)
5. [分阶段开发步骤](#5-分阶段开发步骤)
6. [阶段间依赖关系](#6-阶段间依赖关系)
7. [TDD 与验证策略](#7-tdd-与验证策略)
8. [风险与禁止事项](#8-风险与禁止事项)
9. [完成标准](#9-完成标准)

---

## 1. 文档目的与范围

本文档将 `docs/designs/backend_design.md` 中冻结的 backend 架构，转化为可执行的 Phase 1 开发步骤。

本文档只回答以下问题：

- 开发顺序应如何安排
- 每个阶段应该产出什么
- 每个阶段应如何验证
- 哪些工作必须前置，哪些工作必须后置

如需查看按 checklist 粒度展开的实施顺序、串行/并行边界与推荐批次，参考：

- `docs/designs/backend_phase1_implementation_plan.md`

本文档**不重新定义架构**，也不替代以下设计文档：

- `docs/designs/backend_design.md`
- `docs/designs/backend_phase1_implementation_plan.md`
- `docs/designs/executor_design.md`
- `docs/designs/loaded_model_design.md`
- `docs/designs/kv_cache_design.md`
- `docs/designs/operator_contract_design.md`

---

## 2. 输入设计与约束来源

Backend Phase 1 开发必须遵守以下已冻结约束：

1. `RuntimeContext` 持有 `BackendRegistry`，禁止全局 backend singleton。
2. `RuntimeBuilder` 是 backend 装配入口。
3. Phase 1 以 CPU-first 为主，不以 GPU/异步语义驱动设计。
4. kernel resolve 只能发生在注册期或计划构建期，不能发生在热路径。
5. `ExecutionPlan` 只保存静态执行信息，不包含 request/session 动态绑定。
6. `PackedWeights` 由 `ModelInstance` 的 backend sidecar 持有，而不是由 backend 持有。
7. workspace 必须满足 steady-state zero allocation 约束。
8. `OpKernelContext` 必须保持窄接口，不得把整个 runtime/service 宽对象下放到 kernel 调用。

---

## 3. 当前基线

当前代码库已经具备的基础包括：

- `RuntimeContext`
- `RuntimeBuilder`
- `RuntimeOptions`
- `AllocatorRegistry`
- `Buffer` / `MemoryHandle`
- `Tensor`
- `Device` / `DeviceType`

当前尚未落地或尚未完整落地的 backend 主线包括：

- `BackendFactory`
- `Backend`
- `BackendRegistry`
- `CpuBackend`
- `KernelRegistry`
- `ResolveKernel(...)`
- `ExecutionPlanBuilder`
- `ExecutionPlan`
- `OpExec`
- `RuntimeBindingContext`
- `OpKernelContext`
- packed weight sidecar
- executor backend path integration

---

## 4. 总体开发原则

### 4.1 推荐主顺序

必须遵守以下顺序：

1. 先搭 runtime/backend 骨架
2. 再实现 CPU backend 能力与资源
3. 再打通 kernel resolve 与 execution plan 冻结
4. 再接入 executor 热路径
5. 最后补测试和回归约束

### 4.2 禁止倒序推进

禁止以下开发顺序：

- 先在 `Executor` 中写 runtime resolve，再回头补 `ExecutionPlanBuilder`
- 先写 CUDA/CANN 形状，再让 CPU 适配它
- 先让 kernel 直接调用 allocator，再回头重构 workspace arena
- 先把 session 动态状态塞进 `ExecutionPlan`，再试图拆回去

### 4.3 热路径原则

Phase 1 中必须始终保持：

- init-time / plan-build time resolve
- hot-path direct call
- no runtime registry lookup in hot path
- no fallback decision in hot path
- no temporary heap allocation in decode steady-state path

---

## 5. 分阶段开发步骤

## 5.1 Phase 0：术语与目录冻结

### 目标

在开始代码开发前冻结 backend 相关术语与目录结构，避免边开发边漂移。

### 主要工作

- 建立 `include/aethermind/backend/` 与 `src/backend/` 目录
- 为 CPU backend 建立 `include/aethermind/backend/cpu/` 与 `src/backend/cpu/` 目录
- 冻结以下术语：
  - `BackendFactory`
  - `Backend`
  - `BackendRegistry`
  - `ExecutionPlanBuilder`
  - `ExecutionPlan`
  - `OpExec`
  - `RuntimeBindingContext`
  - `OpKernelContext`
  - `WorkspaceRequirement`
  - `WorkspaceBinding`
  - `WorkspaceArena`

### 阶段退出条件

- 目录结构明确
- 术语不再混用旧版本命名

---

## 5.2 Phase 1：Runtime / Backend 骨架接入

### 目标

让 backend 层正式进入 runtime 组装路径。

### 主要工作

- 定义 `BackendFactory`
- 定义 `Backend`
- 定义 `BackendRegistry`
- 为 `RuntimeContext` 增加 `BackendRegistry`
- 为 `RuntimeContext` 增加 `GetBackend(DeviceType)`
- 为 `RuntimeBuilder` 增加 backend 注册与构建逻辑
- 默认只接入 CPU backend factory

### 阶段退出条件

- `RuntimeBuilder().Build()` 可以得到带 backend registry 的 `RuntimeContext`
- `RuntimeContext::GetBackend(DeviceType::kCPU)` 可以工作
- 无全局 backend registry

---

## 5.3 Phase 2：CPU Backend 执行基础设施

### 目标

建立 Phase 1 唯一真实 backend：CPU backend。

### 主要工作

- 实现 `CpuCapabilities`
- 实现 `CpuBackend`
- 实现 `CpuBackendFactory`
- 实现 `CpuExecutionResources`
- 实现 `CpuWorkspaceArena`
- 建立 CPU backend 的最小 capability 查询逻辑

### 阶段退出条件

- `CpuBackend` 可从 runtime 正常获得
- `capabilities()` 返回稳定结果
- `CpuWorkspaceArena` 可基于预分配区域完成运行期绑定

---

## 5.4 Phase 3：Dispatch 主线建立（按 batch 推进）

### 目标

建立 backend-owned 的 dispatch 主线基线，并按预先冻结的 batch 顺序推进新旧 dispatch 体系迁移。

### 主要工作

- Batch 1：冻结 dispatch 新旧主线边界，并定义 `OpType`、`KernelSelector`、`KernelDescriptor`、`ResolvedKernel`
- Batch 1：将 `KernelKey` / `dispatcher_bridge` / `OperatorName` 标注为迁移期保留，而非未来主线核心
- Batch 2：把 backend 内部 `KernelRegistry` 演进为 selector-based resolve，并让 `CpuBackend` 收敛到 backend-owned registry
- Batch 3（对齐 `ExecutionPlanBuilder` 阶段）：定义 `KernelResolver` 或等价的计划构建期 resolve 逻辑，并冻结 `ResolvedKernel` / `OpExec`
- Batch 4（对齐 executor 接入阶段）：让执行期只消费已冻结 kernel，并正式冻结旧 `Dispatcher / DispatchKeySet` 体系
- 冻结 fallback 行为：只能在 plan-build 阶段决定，不能进入热路径

### 阶段退出条件

- Batch 1 完成后：新 dispatch 主线的最小类型与迁移边界已冻结，且不破坏现有编译/测试
- Batch 2 完成后：CPU kernel 可按 selector 注册，`ResolveKernel(...)` 可根据 selector 与 capability 返回正确 `KernelFn`
- Batch 3 完成后：计划构建期可冻结 `ResolvedKernel`
- Batch 4 完成后：执行期不再触碰 registry，旧 `Dispatcher / DispatchKeySet` 不再承接新功能

---

## 5.5 Phase 4：Packed Weight Sidecar

### 目标

把逻辑权重与 backend packed 权重分开，并固定所有权模型。

### 主要工作

- 为 `ModelInstance` 或等价对象引入 backend sidecar
- 实现 `CpuWeightPrepacker`
- 生成 CPU packed artifacts
- 让 packed 数据由 model sidecar 持有
- 明确 backend 只定义格式与构建逻辑，不持有 packed 数据本体

### 阶段退出条件

- packed weights 与逻辑权重生命周期分离清晰
- packed params 可被执行计划安全引用

---

## 5.6 Phase 5：ExecutionPlanBuilder / ExecutionPlan / OpExec

### 目标

真正落实 “init-time resolve, hot-path direct call”。

### 主要工作

- 定义 `WorkspaceRequirement`
- 定义 `WorkspaceBinding`
- 定义 `OpExec`
- 定义只读 `ExecutionPlan`
- 实现 `ExecutionPlanBuilder`
- 让 `ExecutionPlanBuilder` 成为唯一 kernel resolve 发起方
- 规划 workspace requirement / offset

### 阶段退出条件

- 每个 op 在计划阶段都能冻结成 `OpExec`
- `ExecutionPlan` 只保存静态执行信息
- 动态地址未进入 `ExecutionPlan`

---

## 5.7 Phase 6：RuntimeBindingContext

### 目标

把 request/session 级动态绑定从 `ExecutionPlan` 中隔离出来。

### 主要工作

- 定义 `RuntimeBindingContext`
- 接入 workspace base / binding state
- 接入 KV views
- 接入临时输出缓冲
- 将 `WorkspaceRequirement` 通过 `WorkspaceArena::Bind(...)` 变成 `WorkspaceBinding`

### 阶段退出条件

- request/session 动态状态不进入 `ExecutionPlan`
- runtime binding 可为 plan 中各 op 提供运行期地址绑定

---

## 5.8 Phase 7：OpKernelContext 与执行上下文收窄

### 目标

为热路径定义稳定且受控的 kernel 调用上下文。

### 主要工作

- 定义 `OpKernelContext`
- 接入 `Device`
- 接入最小 `Stream` 占位
- 接入 `TracingSink`
- 接入 `BackendCapabilities`
- 接入 `BackendExecutionResources`
- 严格限制 `opaque_backend_resources` 的使用边界

### 阶段退出条件

- kernel 调用仅依赖 `OpKernelContext + WorkspaceBinding`
- 宽 runtime/service 对象不进入热路径

---

## 5.9 Phase 8：CPU Kernel 最小闭环

### 目标

先打通一个最小可执行 backend 闭环，而不是一次铺满所有算子。

### 推荐闭环

优先选择 1~2 个基础算子，例如：

- RMSNorm
- Linear
- Embedding

### 主要工作

- 注册 CPU kernel
- 让 `ResolveKernel(...)` 可以解析到它
- 让 `ExecutionPlanBuilder` 生成对应 `OpExec`
- 在测试路径中直接执行 `KernelFn`

### 阶段退出条件

- 至少一个 op 能完成 plan-build + direct call 闭环
- 热路径中无 registry 查找

---

## 5.10 Phase 9：Executor / LayerRunner 接入

### 目标

让 execution 层真正消费 backend execution plan。

### 主要工作

- 让 executor 拿到 `ExecutionPlan`
- 让 executor 拿到 `RuntimeBindingContext`
- 让 `LayerRunner` 遍历 `OpExec`
- 调用 `KernelFn(invocation, op_ctx, workspace_binding)`
- 禁止执行期 resolve/fallback/动态申请临时内存

### 阶段退出条件

- executor 热路径只消费 plan
- 计划阶段与执行阶段职责不再混合

---

## 5.11 Phase 10：测试与回归收口

### 目标

为 backend 主线建立最小而稳定的回归保护。

### 主要工作

- runtime/backend registry 测试
- CPU backend capability 测试
- `ResolveKernel(...)` 测试
- `ExecutionPlanBuilder` 测试
- workspace 稳态地址测试
- packed weight ownership 测试
- executor direct call 路径测试

### 阶段退出条件

- backend 主线具备 focused test coverage
- 关键 ownership / hot-path / zero-allocation 约束得到验证

---

## 6. 阶段间依赖关系

### 6.1 强依赖链

- Phase 1 依赖 Phase 0
- Phase 2 依赖 Phase 1
- Phase 3 依赖 Phase 2
- Phase 5 依赖 Phase 3 与 Phase 4
- Phase 6 依赖 Phase 5
- Phase 9 依赖 Phase 6、7、8

### 6.2 可局部并行的部分

在边界冻结后，可有限并行推进：

- `CpuCapabilities` 与 `CpuWorkspaceArena`
- packed weight sidecar 与部分 kernel registration
- `RuntimeBindingContext` 与 executor 接入前的测试准备

但不得打乱主依赖关系。

---

## 7. TDD 与验证策略

### 7.1 Red-Green-Refactor

每个阶段推荐遵循：

1. **Red**：先补最窄失败测试
2. **Green**：写最小实现通过测试
3. **Refactor**：清理 ownership、命名、const-correctness、热路径约束

### 7.2 每阶段通用验证

- 构建最窄受影响目标
- 跑最窄 gtest filter
- 检查没有新引入 LSP diagnostics
- 复核是否破坏以下约束：
  - no runtime hot-path resolve
  - no plan mutation at execution time
  - no packed weight ownership ambiguity
  - no workspace allocation in steady-state decode

### 7.3 重点验证点

- `ExecutionPlanBuilder` 必须是唯一 resolve 发起方
- `Executor` 不得直接接触 `KernelRegistry`
- `WorkspaceArena::Bind(...)` 不得触发底层堆分配
- `opaque_backend_resources` 不得成为绕过窄接口的逃逸口

---

## 8. 风险与禁止事项

### 8.1 主要风险

- 把 runtime 动态状态重新塞回 `ExecutionPlan`
- 把 packed weight 所有权错误地放进 backend
- 在 `Executor` 中偷偷重做 resolve 或 fallback
- 让 workspace 退化为 `Allocate(Buffer)` 风格
- 让 `opaque_backend_resources` 逐步膨胀为宽对象通道

### 8.2 明确禁止事项

- 禁止 GPU/异步语义反向主导 CPU Phase 1 设计
- 禁止热路径 registry 查找
- 禁止热路径 capability 判断
- 禁止热路径 fallback 决策
- 禁止 decode steady-state 临时堆分配

---

## 9. 完成标准

当以下条件同时满足时，backend Phase 1 主线开发可视为完成：

1. `RuntimeContext` 已正式持有 `BackendRegistry`
2. `CpuBackend` 已可通过 runtime 获取
3. `ResolveKernel(...)` 可在 plan-build 阶段完成解析
4. `ExecutionPlanBuilder` 可生成只读 `ExecutionPlan`
5. `RuntimeBindingContext` 可完成运行期 workspace/KV 绑定
6. `Executor` 热路径只消费 `ExecutionPlan`
7. 至少一条 CPU kernel 路径完成 direct call 闭环
8. 所有关键约束都有 focused tests 保护

---

## 结语

Backend Phase 1 的开发重点不是“做出一个看起来可扩展的抽象框架”，而是按正确顺序把 CPU backend 主线做实：

- runtime 装配
- backend 能力
- kernel resolve
- packed weight sidecar
- execution plan 冻结
- runtime binding
- executor direct call

只有这条主线先稳定下来，后续的多后端、stream/event、异步执行和更细粒度 specialization 才有正确的扩展基础。
