# AetherMind Dispatch 设计执行任务清单

**版本**: v1.0\
**日期**: 2026-04-15\
**作者**: AetherMind Team

***

## 1. 文档目的

本文件将 `docs/designs/dispatch_design.md` 中定义的 **backend-owned、plan-build-time dispatch 主线方案**，进一步拆解成可执行的、按文件粒度组织的 implementation task list。

本清单的定位是：

- 指导 dispatch 新主线如何逐步落地；
- 明确哪些文件需要新增、改造、冻结；
- 明确每一批的完成标准与验证重点；
- 避免旧 `Dispatcher / DispatchKeySet` 在迁移期继续膨胀。

***

## 2. 总体迁移目标

把当前代码从：

```text
KernelKey + OperatorName + old Dispatcher / DispatchKeySet
```

迁移到：

```text
OpType + KernelSelector + KernelDescriptor + backend-owned KernelRegistry
+ plan-build-time resolve + ResolvedKernel
```

并保证：

- `ExecutionPlanBuilder` 是唯一正式 resolve 发起方
- `Executor` 只消费冻结后的 kernel 函数指针
- 执行期不做字符串查找 / hash dispatch / registry resolve
- 旧 `Dispatcher / DispatchKeySet` 冻结，不再扩展

***

## 3. 批次划分

建议按 4 个 batch 推进：

| Batch       | 覆盖阶段              | 核心目标                                                 |
| ----------- | ----------------- | ---------------------------------------------------- |
| **Batch 1** | Phase A + Phase B | 冻结边界，落最小新类型                                          |
| **Batch 2** | Phase C + Phase D | selector-based `KernelRegistry` + `CpuBackend` 注册/解析 |
| **Batch 3** | Phase E           | 计划构建期 resolve 与 `ResolvedKernel` 冻结                  |
| **Batch 4** | Phase F + Phase G | Executor direct call + 旧 dispatch 体系冻结               |

***

## 4. Batch 1：冻结边界 + 落最小新类型 ✅

### 4.1 目标

建立新 dispatch 主线的最小类型系统，并明确旧 dispatch 体系不再作为未来主线继续扩展。

### 4.2 文件任务清单

#### 文档冻结

- [x] `docs/designs/dispatch_design.md`: 冻结 dispatch 主线基线
- [x] `docs/designs/backend_phase1_development_steps.md`: 与新 dispatch 主线保持一致
- [x] `docs/designs/backend_phase1_implementation_plan.md`: 与新 dispatch 主线保持一致
- [x] `docs/designs/backend_phase1_implementation_checklist.md`: 与新 dispatch 主线保持一致

#### 新增核心类型

- [x] `include/aethermind/operators/op_type.h`: 定义 `OpType`
- [x] `include/aethermind/backend/kernel_selector.h`: 定义 `KernelSelector`、`IsaLevel`、`ExecPhase`、`WeightFormat`
- [x] `include/aethermind/backend/kernel_descriptor.h`: 定义 `KernelDescriptor`
- [x] `include/aethermind/backend/resolved_kernel.h`: 定义 `ResolvedKernel`

#### 迁移辅助（已退场）

- [x] ~~`include/operator_name.h`~~: 保留用于 `ToOpType` 过渡映射
- [x] ~~`include/aethermind/backend/kernel_key.h`~~: **已删除** - 迁移完成后退场
- [x] ~~`include/aethermind/backend/dispatcher_bridge.h`~~: **已删除** - 迁移完成后退场

### 4.3 验证重点

- 新类型头文件能独立通过编译
- 不破坏现有 Batch A / Batch B 测试
- 文档口径统一为 backend-owned、plan-build-time dispatch

***

## 5. Batch 2：selector-based KernelRegistry + CpuBackend 收敛 ✅

### 5.1 目标

把 `KernelRegistry` 从"按 key 查表"演进为"按 selector resolve"，并让 `CpuBackend` 成为新 dispatch 主线的真正 owner。

### 5.2 文件任务清单

#### 核心 registry

- [x] `include/aethermind/backend/kernel_registry.h`: 新增 `Register(const KernelDescriptor&)` 与 `Resolve(OpType, KernelSelector, ...)`
- [x] `src/backend/kernel_registry.cpp`: 实现 selector-based resolve、`kBoth` 匹配、ISA 兼容过滤、priority 选择

#### 迁移兼容（已退场）

- [x] ~~`include/aethermind/backend/kernel_key.h`~~: **已删除**
- [x] ~~`src/backend/dispatcher_bridge.cpp`~~: **已删除**

#### CpuBackend 收敛

- [x] `include/aethermind/backend/cpu/cpu_backend.h`: 明确 backend 内持有 `KernelRegistry`
- [x] `src/backend/cpu/cpu_backend.cpp`: 在初始化路径中显式注册 builtin kernels，并接入 selector-based `ResolveKernel(...)`

### 5.3 建议测试

- [x] `tests/unit/test_kernel_registry.cpp`: 扩展为 descriptor 注册与 resolve 覆盖
- [x] `tests/unit/test_kernel_selector.cpp`: 新增 selector 匹配测试
- [x] `tests/unit/test_kernel_registry_resolve.cpp`: 新增 `kBoth`、ISA、priority resolve 测试
- [x] `tests/unit/test_cpu_resolve_kernel.cpp`: 扩展 builtin kernel 注册与 selector resolve 覆盖

### 5.4 验证重点

- `KernelRegistry` 可按 selector resolve
- `ExecPhase::kBoth` 可匹配 `kPrefill` / `kDecode`
- `priority` 只在“多个可用实现”中排序
- `CpuBackend` 使用 backend-owned registry，不引入全局 singleton

***

## 6. Batch 3：计划构建期 resolve 与 ResolvedKernel 冻结 ✅

### 6.1 目标

把 resolve 明确拉进计划构建期，使 `ExecutionPlanBuilder` 成为唯一正式 resolve 发起方。

### 6.2 文件任务清单

#### 执行计划构建侧

- [x] `include/aethermind/execution/execution_plan_builder.h`: 定义 `ResolveKernelForNode(...)`，`ExecutionPlanNodeSpec` 使用 `OpType`
- [x] `src/execution/execution_plan_builder.cpp`: 基于 `OpType + KernelSelector + Backend::ResolveKernel(...)` 完成计划构建期 resolve

#### 冻结结果类型

- [x] `include/aethermind/backend/resolved_kernel.h`: 补齐 `attrs` / `debug_name` 生命周期约束说明
- [x] `include/aethermind/execution/execution_plan.h`: 接入 `ResolvedKernel`，`ExecutionStep` 只存 kernel
- [x] `src/execution/execution_plan.cpp`: 落实冻结结果存储，attrs 复制到 plan-owned storage

#### 迁移辅助

- [x] `include/aethermind/operators/op_type.h`: `ToOpType` 实现挪回 operators 模块

### 6.3 建议测试

- [x] `tests/unit/test_execution_plan_builder.cpp`: 覆盖 plan-build resolve
- [x] `tests/unit/test_execution_plan.cpp`: 覆盖 `ResolvedKernel` 冻结结果

### 6.4 验证重点

- `ExecutionPlanBuilder` 是唯一 resolve 发起方
- 计划构建期可冻结 `ResolvedKernel`
- attrs 生命周期覆盖整个 execution plan
- 执行节点不再依赖 runtime lookup

***

## 7. Batch 4：Executor direct call + 旧体系冻结 ✅

### 7.1 目标

让 `Executor` 热路径只做函数指针调用，并正式冻结旧 `Dispatcher / DispatchKeySet` 体系。

### 7.2 文件任务清单

#### Executor / LayerRunner

- [x] `include/aethermind/execution/executor.h`: 接入 `ResolvedKernel` / `ExecutionPlan`
- [x] `src/execution/executor.cpp`: 执行期 direct call，不做 runtime resolve
- [x] `include/aethermind/execution/layer_runner.h`: 接入已冻结 kernel 执行接口
- [x] `src/execution/layer_runner.cpp`: 遍历节点并 direct call `KernelFn`

#### 旧 dispatch 体系冻结/删除

- [x] ~~`include/dispatcher.h`~~: **已删除** - legacy dispatcher 退场
- [x] ~~`src/dispatcher.cpp`~~: **已删除**
- [x] `include/dispatch_key.h`: 标注冻结/待退场（文件保留，仅标注 deprecated）
- [x] `include/dispatch_key_set.h`: 标注冻结/待退场，不再扩展

#### 迁移辅助退场

- [x] ~~`include/aethermind/backend/dispatcher_bridge.h`~~: **已删除**
- [x] ~~`src/backend/dispatcher_bridge.cpp`~~: **已删除**
- [x] ~~`include/aethermind/backend/kernel_key.h`~~: **已删除**
- [x] ~~`tests/unit/test_dispatcher_bridge.cpp`~~: **已删除**

### 7.3 建议测试

- [x] `tests/unit/test_executor_backend_path.cpp`: 覆盖 executor direct call
- [x] `tests/unit/test_no_hotpath_resolve.cpp`: 断言执行路径不做 registry / dispatcher lookup

### 7.4 验证重点

- executor 热路径 direct call
- 无 runtime resolve
- 无 hot-path registry access
- 新功能全部走新主线

***

## 8. 风险控制 ✅

### 8.1 不要做的事（已遵守）

- [x] 不要一次性删除 `OperatorName` — `ToOpType` 仍保留过渡映射
- [x] 不要先删旧 dispatcher 再补新主线 — Batch 1-3 先立新主线，Batch 4 再冻结删除
- [x] 不要把全局 singleton registry 引回来 — `KernelRegistry` 由 Backend 持有
- [x] 不要把 `RuntimeContext*` 直接传给 kernel — `KernelFunc` 保持窄签名
- [x] 不要让 executor 在迁移期继续做 hash/string lookup — Executor 只消费冻结 kernel

### 8.2 必须守住的约束（已守住）

- [x] backend-owned ownership
- [x] attrs 生命周期明确（plan-owned storage）
- [x] resolve 只发生在 plan-build time

***

## 9. 一句话执行策略

> **先立新主线，再迁移 resolve，最后冻结旧 dispatcher。**

**状态：已完成 ✅**

Dispatch 新主线已落地：
- `OpType + KernelSelector + KernelDescriptor` 作为核心类型
- `Backend-owned KernelRegistry` 完成 selector-based resolve
- `ExecutionPlanBuilder` 是唯一 resolve 发起方
- `Executor` 只消费冻结后的 `ResolvedKernel`
- Legacy `Dispatcher` / `dispatcher_bridge` / `kernel_key` 已退场

