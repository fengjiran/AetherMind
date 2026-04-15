# AetherMind Backend Phase 1 Implementation Checklist

**版本**: v1.0  
**日期**: 2026-04-13  
**作者**: AetherMind Team

---

## 目录

1. [使用说明](#1-使用说明)
2. [设计追踪来源](#2-设计追踪来源)
3. [Phase 0：目录与接口冻结](#3-phase-0目录与接口冻结)
4. [Phase 1：Runtime / Backend 骨架接入](#4-phase-1runtime--backend-骨架接入)
5. [Phase 2：CPU Backend 基础设施](#5-phase-2cpu-backend-基础设施)
6. [Phase 3：Kernel 注册与解析](#6-phase-3kernel-注册与解析)
7. [Phase 4：Packed Weight Sidecar](#7-phase-4packed-weight-sidecar)
8. [Phase 5：ExecutionPlan / OpExec / PlanBuilder](#8-phase-5executionplan--opexec--planbuilder)
9. [Phase 6：RuntimeBindingContext](#9-phase-6runtimebindingcontext)
10. [Phase 7：OpKernelContext 与执行接入](#10-phase-7opkernelcontext-与执行接入)
11. [Phase 8：测试与回归](#11-phase-8测试与回归)
12. [验证命令建议](#12-验证命令建议)
13. [Done Criteria](#13-done-criteria)

---

## 1. 使用说明

本文件是 backend Phase 1 的**按文件粒度拆解**的 implementation checklist。

使用规则：

- 每个 checkbox 只对应一个文件和一个主要职责
- 优先先补测试，再补实现
- 所有实现必须服从 `docs/designs/backend_design.md`
- 实施顺序、串并行边界与推荐批次参考 `docs/designs/backend_phase1_implementation_plan.md`
- 所有 checklist 项均应保持“可 review、可验证、可回归”

单项格式：

```text
[ ] <path>: <single responsibility>
```

---

## 2. 设计追踪来源

本 checklist 主要对应以下设计文档：

- `docs/designs/backend_design.md`
- `docs/designs/backend_phase1_development_steps.md`
- `docs/designs/backend_phase1_implementation_plan.md`
- `docs/designs/executor_design.md`
- `docs/designs/loaded_model_design.md`
- `docs/designs/kv_cache_design.md`
- `docs/designs/operator_contract_design.md`

---

## 3. Phase 0：目录与接口冻结

### 文档与接口边界确认

- [x] `docs/designs/backend_design.md`: 复核术语与最新代码接口一致
- [x] `docs/designs/backend_phase1_development_steps.md`: 复核阶段顺序与当前实现计划一致

### 目录结构

- [x] `include/aethermind/backend/`: 建立 backend 公共头目录
- [x] `src/backend/`: 建立 backend 实现目录
- [x] `include/aethermind/backend/cpu/`: 建立 CPU backend 公共头目录
- [x] `src/backend/cpu/`: 建立 CPU backend 实现目录

---

## 4. Phase 1：Runtime / Backend 骨架接入

### 测试先行

- [x] `tests/unit/test_backend_registry.cpp`: 增加 factory 注册、lazy backend 创建、缓存行为测试
- [x] `tests/unit/test_runtime_backend_integration.cpp`: 增加 `RuntimeBuilder` / `RuntimeContext` backend 装配测试

### Backend 基础接口

- [x] `include/aethermind/backend/backend_fwd.h`: 声明 backend 相关前向声明
- [x] `include/aethermind/backend/kernel_types.h`: 声明 `KernelFn`、`KernelKey` 所需基础类型占位
- [x] `include/aethermind/backend/backend_capabilities.h`: 定义 `BackendCapabilities` 基础接口或基础结构
- [x] `include/aethermind/backend/backend.h`: 定义 `Backend` 接口，包括 `ResolveKernel(...)`
- [x] `include/aethermind/backend/backend_factory.h`: 定义 `BackendFactory` 接口
- [x] `include/aethermind/backend/backend_registry.h`: 定义 `BackendRegistry` 接口
- [x] `src/backend/backend_registry.cpp`: 实现 `BackendRegistry` 注册、覆盖、延迟实例化与缓存逻辑

### Runtime 集成

- [x] `include/aethermind/runtime/runtime_context.h`: 增加 `BackendRegistry` 成员与 `GetBackend(DeviceType)`
- [x] `src/runtime/runtime_context.cpp`: 实现 `GetBackend(DeviceType)`
- [x] `include/aethermind/runtime/runtime_builder.h`: 增加 backend factory 注册入口与 build 钩子
- [x] `src/runtime/runtime_builder.cpp`: 实现 `BuildBackendRegistry()` 和默认 CPU backend 注册

### Phase 1 退出条件验证

- [x] `RuntimeContext` 正式持有 `BackendRegistry`
- [x] `RuntimeContext::GetBackend(DeviceType::kCPU)` 可用
- [x] 无全局 backend registry

---

## 5. Phase 2：CPU Backend 基础设施

### 测试先行

- [x] `tests/unit/test_cpu_backend.cpp`: 增加 `CpuBackend` 基础行为测试
- [x] `tests/unit/test_cpu_capabilities.cpp`: 增加 CPU capability 可见性测试
- [x] `tests/unit/test_cpu_workspace_arena.cpp`: 增加 workspace 绑定与 reset 测试

### CPU Backend 核心类型

- [x] `include/aethermind/backend/cpu/cpu_capabilities.h`: 定义 `CpuCapabilities`
- [x] `src/backend/cpu/cpu_capabilities.cpp`: 实现 CPU ISA / capability 探测（当前为占位实现，ISA 探测延后到优化阶段）
- [x] `include/aethermind/backend/cpu/cpu_execution_resources.h`: 定义 `CpuExecutionResources`（空占位结构，Phase 1 同步执行不需要）
- [ ] `include/aethermind/backend/cpu/cpu_thread_pool.h`: 定义 CPU backend 所需线程池接口或适配层（**延后**：Phase 1 同步执行，线程池属于后续优化阶段）
- [ ] `src/backend/cpu/cpu_thread_pool.cpp`: 实现最小线程池或线程池适配逻辑（**延后**：同上）
- [x] `include/aethermind/backend/cpu/cpu_workspace_arena.h`: 定义 `CpuWorkspaceArena`
- [x] `src/backend/cpu/cpu_workspace_arena.cpp`: 实现 arena 绑定逻辑
- [x] `include/aethermind/backend/cpu/cpu_backend.h`: 定义 `CpuBackend` 与 `CpuBackendFactory`
- [x] `src/backend/cpu/cpu_backend.cpp`: 实现 `CpuBackend` 基础行为

### Phase 2 退出条件验证

- [x] `CpuBackend` 可从 runtime 正常获得（测试验证：`RuntimeBuilder().Build().GetBackend(kCPU)`）
- [x] `capabilities()` 返回稳定结果（测试验证：`test_cpu_backend.cpp` + `test_cpu_capabilities.cpp`）
- [x] `CpuWorkspaceArena` 可基于预分配区域完成运行期绑定（测试验证：`test_cpu_workspace_arena.cpp` 11 个测试）

---

## 6. Phase 3：Dispatch 主线建立（按 batch 推进）

### Batch 1：设计基线与最小新类型

- [x] `docs/designs/dispatch_design.md`: 冻结 backend-owned、plan-build-time dispatch 主线方案
- [x] `docs/designs/backend_phase1_development_steps.md`: 对齐 dispatch batches 与 backend Phase 1 阶段边界
- [x] `docs/designs/backend_phase1_implementation_plan.md`: 对齐 dispatch batches 与实施顺序
- [x] `include/aethermind/operators/op_type.h`: 定义 `OpType` 与 `OperatorName -> OpType` 过渡入口声明
- [x] `include/aethermind/backend/kernel_selector.h`: 定义 `KernelSelector`、`IsaLevel`、`ExecPhase`、`WeightFormat`
- [x] `include/aethermind/backend/kernel_descriptor.h`: 定义 `KernelDescriptor`
- [x] `include/aethermind/backend/resolved_kernel.h`: 定义 `ResolvedKernel`
- [x] `include/aethermind/backend/kernel_key.h`: 标注为迁移期保留类型，而非未来主线核心
- [x] `include/aethermind/backend/dispatcher_bridge.h`: 标注为迁移辅助，不再承接未来主线职责
- [x] `include/operator_name.h`: 保持过渡兼容，并为 `OperatorName -> OpType` 映射预留入口

### Batch 2：selector-based resolve

### 测试先行

- [x] `tests/unit/test_kernel_registry.cpp`: 增加 kernel 注册与查询测试
- [x] `tests/unit/test_cpu_resolve_kernel.cpp`: 增加 `ResolveKernel(...)` 解析测试

### Kernel 解析核心

- [x] `include/aethermind/backend/kernel_registry.h`: 定义 backend-owned `KernelRegistry`
- [x] `src/backend/kernel_registry.cpp`: 实现 backend 内部 kernel registry
- [x] `include/aethermind/backend/kernel_invocation.h`: 定义 `KernelInvocation` 基础结构
- [x] `src/backend/dispatcher_bridge.cpp`: 实现迁移期桥接逻辑
- [x] `src/backend/cpu/cpu_backend.cpp`: 接入 `ResolveKernel(...)`

### Batch 4：旧分发设施冻结与迁移边界

- [ ] `include/dispatcher.h`: 明确旧全局 dispatcher 不再作为新算子实现主线
- [ ] `src/dispatcher.cpp`: 明确不承接 runtime resolve 职责
- [ ] `include/dispatch_key.h`: 明确旧 `DispatchKey` 体系不再作为新主线基础
- [ ] `include/dispatch_key_set.h`: 冻结/待退场，不再扩展

---

## 7. Phase 4：Packed Weight Sidecar

### 测试先行

- [ ] `tests/unit/test_cpu_weight_prepacker.cpp`: 增加 packed weight 构建测试
- [ ] `tests/unit/test_backend_sidecar_ownership.cpp`: 增加 packed weight ownership 测试

### Packed Weight 路径

- [ ] `include/aethermind/backend/packed_weights.h`: 定义 packed weight 基础抽象或公共类型
- [ ] `include/aethermind/backend/cpu/cpu_weight_prepacker.h`: 定义 `CpuWeightPrepacker`
- [ ] `src/backend/cpu/cpu_weight_prepacker.cpp`: 实现逻辑权重到 packed 权重转换
- [ ] `include/aethermind/model/backend_sidecar.h`: 定义 `ModelInstance` backend sidecar 类型
- [ ] `src/model/backend_sidecar.cpp`: 实现 sidecar 生命周期与访问逻辑
- [ ] `include/aethermind/model/model_instance.h`: 接入 backend sidecar 持有关系
- [ ] `src/model/model_instance.cpp`: 落实 packed weight sidecar 生命周期管理

---

## 8. Phase 5：ExecutionPlan / OpExec / PlanBuilder

### 测试先行

- [ ] `tests/unit/test_execution_plan_builder.cpp`: 增加计划构建与 resolve 唯一入口测试
- [ ] `tests/unit/test_execution_plan.cpp`: 增加 plan 只读视图与 layer op 序列测试
- [ ] `tests/unit/test_workspace_requirement_planning.cpp`: 增加 workspace requirement/offset 规划测试

### 计划构建核心

- [ ] `include/aethermind/backend/workspace_types.h`: 定义 `WorkspaceRequirement` 与 `WorkspaceBinding`
- [ ] `include/aethermind/backend/execution_plan.h`: 定义 `ExecutionPlan` 与 `OpExec`
- [ ] `src/backend/execution_plan.cpp`: 实现 `ExecutionPlan` 存储与只读视图
- [ ] `include/aethermind/backend/execution_plan_builder.h`: 定义 `ExecutionPlanBuilder`
- [ ] `src/backend/execution_plan_builder.cpp`: 实现计划构建、resolve 与 workspace 规划

### 相邻集成

- [ ] `include/aethermind/operator/op_kind.h`: 如需要，定义 `OpKind`
- [ ] `include/operator_name.h`: 过渡期提供 `OperatorName -> OpType` 映射，不再继续扩展旧 dispatcher 语义
- [ ] `include/function_schema.h`: 如需要，补计划构建阶段所需 schema 查询入口

---

## 9. Phase 6：RuntimeBindingContext

### 测试先行

- [ ] `tests/unit/test_runtime_binding_context.cpp`: 增加 session/request 动态绑定测试
- [ ] `tests/unit/test_workspace_runtime_binding.cpp`: 增加 `WorkspaceRequirement -> WorkspaceBinding` 绑定测试

### 动态绑定核心

- [ ] `include/aethermind/backend/runtime_binding_context.h`: 定义 `RuntimeBindingContext`
- [ ] `src/backend/runtime_binding_context.cpp`: 实现 workspace/KV/临时输出绑定逻辑
- [ ] `include/aethermind/backend/workspace_arena.h`: 定义 `WorkspaceArena` 公共接口
- [ ] `src/backend/cpu/cpu_workspace_arena.cpp`: 接入 `Bind(...)` 与 `Reset()`

### KV 相关对齐

- [ ] `include/aethermind/execution/kv_cache_view.h`: 校验并接入 runtime binding 所需 KV view 接口
- [ ] `include/aethermind/execution/kv_cache_manager.h`: 校验逻辑视图与 backend 物理契约边界

---

## 10. Phase 7：OpKernelContext 与执行接入

### 测试先行

- [ ] `tests/unit/test_op_kernel_context.cpp`: 增加窄上下文约束测试
- [ ] `tests/unit/test_executor_backend_path.cpp`: 增加 executor 通过 plan direct-call 的最小路径测试

### 执行上下文

- [ ] `include/aethermind/backend/op_kernel_context.h`: 定义 `BackendExecutionResources` 与 `OpKernelContext`
- [ ] `include/aethermind/backend/stream.h`: 定义最小 `Stream` 接口
- [ ] `include/aethermind/backend/tracing_sink.h`: 定义最小 `TracingSink` 接口

### Executor / LayerRunner 接入

- [ ] `include/aethermind/execution/executor.h`: 接入 `ExecutionPlan` 与 `RuntimeBindingContext`
- [ ] `src/execution/executor.cpp`: 实现基于 plan 的执行入口
- [ ] `include/aethermind/execution/layer_runner.h`: 接入 `OpExec` 执行接口
- [ ] `src/execution/layer_runner.cpp`: 遍历 `OpExec` 并 direct call `KernelFn`

### CPU 最小闭环 Kernel

- [ ] `include/aethermind/backend/cpu/kernels/cpu_rmsnorm_kernel.h`: 定义一个最小 CPU kernel 示例
- [ ] `src/backend/cpu/kernels/cpu_rmsnorm_kernel.cpp`: 实现最小 CPU kernel 示例
- [ ] `src/backend/cpu/cpu_backend.cpp`: 将最小 kernel 注册进 CPU backend 并支持 resolve

---

## 11. Phase 8：测试与回归

### 约束型测试

- [ ] `tests/unit/test_no_hotpath_resolve.cpp`: 断言执行路径不做 registry 查找
- [ ] `tests/unit/test_execution_plan_immutability.cpp`: 断言执行期不修改 `ExecutionPlan`
- [ ] `tests/unit/test_decode_workspace_stability.cpp`: 断言 decode 稳态 workspace 地址稳定
- [ ] `tests/unit/test_backend_resource_escape_hatch.cpp`: 断言 `opaque_backend_resources` 不回传宽对象指针

### 集成回归

- [ ] `tests/unit/test_runtime_cpu_backend_e2e.cpp`: 最小 runtime -> backend -> plan -> executor 路径测试

---

## 12. 验证命令建议

推荐按最窄目标先验证：

```bash
cmake --build build --target AetherMind -j
cmake --build build --target aethermind_unit_tests -j

./build/tests/unit/aethermind_unit_tests --gtest_filter=BackendRegistry.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=CPUBackend.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=ExecutionPlan.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=RuntimeBindingContext.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=ExecutorBackendPath.*
```

若某阶段只改动单个 backend 模块，应优先补对应 focused filter，而不是全量跑整个测试集。

---

## 13. Done Criteria

只有当以下条件全部满足时，backend Phase 1 checklist 才能视为完成：

- [ ] `RuntimeContext` 正式持有 `BackendRegistry`
- [ ] `CpuBackend` 可被 runtime 查询并返回稳定 capability
- [ ] `ExecutionPlanBuilder` 是唯一 kernel resolve 发起方
- [ ] `KernelRegistry` 由 backend 持有，而不是全局 singleton
- [ ] `ExecutionPlan` 不包含 request/session 动态绑定
- [ ] `PackedWeights` 由 `ModelInstance` backend sidecar 持有
- [ ] `WorkspaceArena::Bind(...)` 不触发底层堆分配
- [ ] `Executor` 只消费 plan，不做热路径 resolve
- [ ] 至少一个 CPU kernel 完成 end-to-end direct call 闭环
- [ ] 关键 ownership / hot-path / zero-allocation 约束有 focused tests 覆盖
