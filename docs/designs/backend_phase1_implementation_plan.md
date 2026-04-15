# AetherMind Backend Phase 1 实施计划文档

**版本**: v1.0  
**日期**: 2026-04-14  
**作者**: AetherMind Team

---

## 目录

1. [文档目的](#1-文档目的)
2. [输入基线](#2-输入基线)
3. [总体实施原则](#3-总体实施原则)
4. [主依赖链与串行边界](#4-主依赖链与串行边界)
5. [分阶段实施计划](#5-分阶段实施计划)
6. [并行化窗口](#6-并行化窗口)
7. [推荐实施批次](#7-推荐实施批次)
8. [每阶段验证门槛](#8-每阶段验证门槛)
9. [关键风险与实施注意事项](#9-关键风险与实施注意事项)

---

## 1. 文档目的

本文档将以下文档中已经冻结的 backend Phase 1 设计与 checklist，进一步整理为一份可直接执行的实施计划：

- `docs/designs/backend_design.md`
- `docs/designs/backend_phase1_development_steps.md`
- `docs/designs/backend_phase1_implementation_checklist.md`

本文档重点回答以下问题：

- backend Phase 1 应按什么顺序落地
- 哪些 checklist 项必须串行推进
- 哪些 checklist 项可以有限并行推进
- 每个阶段建议以什么粒度组织提交与验证

本文档不重新定义 backend 架构，不替代设计文档与 checklist，只负责把实施顺序和工程执行方式固化下来。

---

## 2. 输入基线

当前仓库已经具备以下可复用基础设施：

- `RuntimeContext`
- `RuntimeBuilder`
- `RuntimeOptions`
- `AllocatorRegistry`
- `Buffer` / `MemoryHandle`
- `Tensor`
- `Device` / `DeviceType`
- `Status` / `StatusOr`

当前 backend Phase 1 主线相关对象尚未正式落地或未形成闭环，包括：

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

因此，backend Phase 1 的实施应从目录与骨架阶段开始，而不是从 executor 或 kernel 末端倒推。

---

## 3. 总体实施原则

### 3.1 主原则

- 先冻结 contract，再写实现
- 先让 runtime 持有 backend，再做 CPU backend 细节
- 先在计划阶段 resolve，再让执行期 direct call
- 先隔离静态计划与动态绑定，再接入 executor
- 先跑通最小闭环，再扩展算子与测试覆盖

### 3.2 测试优先原则

每个阶段均建议遵循以下顺序：

1. 先补最窄失败测试
2. 再补头文件与实现
3. 再跑最窄构建目标与 gtest filter
4. 最后更新 checklist / 进入下一阶段

### 3.3 热路径保护原则

整个实施过程中，必须持续复核以下约束没有被破坏：

- no runtime hot-path resolve
- no runtime registry lookup in hot path
- no fallback decision in hot path
- no temporary heap allocation in steady-state decode
- no request/session dynamic state stored inside `ExecutionPlan`

---

## 4. 主依赖链与串行边界

backend Phase 1 的主链必须严格遵循以下顺序：

```text
Phase 0
  → Phase 1
  → Phase 2
  → Phase 3
  → Phase 4
  → Phase 5
  → Phase 6
  → Phase 7
  → Phase 8
```

### 4.1 不能打乱的强依赖

- `Phase 1` 依赖 `Phase 0`
- `Phase 2` 依赖 `Phase 1`
- `Phase 3` 依赖 `Phase 2`
- `Phase 4` 与 `Phase 3` 共同构成 `Phase 5` 的前置条件
- `Phase 6` 依赖 `Phase 5`
- `Phase 7` 依赖 `Phase 6`
- `Phase 8` 依赖 `Phase 7`

### 4.2 必须串行推进的核心原因

- `RuntimeContext` 必须先具备 backend ownership，后续 CPU backend 才有稳定挂载点
- `CpuBackend` 必须先成型，`ResolveKernel(...)` 才有真实承载对象
- `ExecutionPlanBuilder` 必须建立在 kernel resolve 与 packed params 都已冻结的前提上
- `RuntimeBindingContext` 必须在 `ExecutionPlan` 静态化之后出现，否则容易把动态状态重新塞回 plan
- `Executor` 接入必须晚于 plan 与 binding 语义冻结，否则热路径职责会再次混杂

---

## 5. 分阶段实施计划

## 5.1 Phase 0：目录与接口冻结

### 目标

建立 backend 目录结构，冻结后续实现中的术语与文件落点。

### Checklist 项

#### 串行项

1. `docs/designs/backend_design.md`: 复核术语与最新代码接口一致
2. `docs/designs/backend_phase1_development_steps.md`: 复核阶段顺序与当前实现计划一致

#### 可并行项

- `include/aethermind/backend/`: 建立 backend 公共头目录
- `src/backend/`: 建立 backend 实现目录
- `include/aethermind/backend/cpu/`: 建立 CPU backend 公共头目录
- `src/backend/cpu/`: 建立 CPU backend 实现目录

### 阶段出口

- 目录结构明确
- backend 术语不再漂移
- 后续文件命名与模块边界可稳定展开

---

## 5.2 Phase 1：Runtime / Backend 骨架接入

### 目标

让 backend 层正式进入 runtime 装配路径。

### 建议顺序

#### 第一步：测试先行（可并行）

- `tests/unit/test_backend_registry.cpp`
- `tests/unit/test_runtime_backend_integration.cpp`

#### 第二步：Backend 基础接口（建议串行）

1. `include/aethermind/backend/backend_fwd.h`
2. `include/aethermind/backend/kernel_types.h`
3. `include/aethermind/backend/backend_capabilities.h`
4. `include/aethermind/backend/backend.h`
5. `include/aethermind/backend/backend_factory.h`
6. `include/aethermind/backend/backend_registry.h`

#### 第三步：Registry 实现（串行）

- `src/backend/backend_registry.cpp`

#### 第四步：Runtime 集成（建议串行）

1. `include/aethermind/runtime/runtime_context.h`
2. `src/runtime/runtime_context.cpp`
3. `include/aethermind/runtime/runtime_builder.h`
4. `src/runtime/runtime_builder.cpp`

### 为什么这里必须以串行为主

- `Backend` / `BackendFactory` / `BackendRegistry` 的依赖层次很强
- `RuntimeContext` 与 `RuntimeBuilder` 的 ownership 关系必须一次性收敛
- 如果先写 builder 再补 context，容易产生临时接口与返工

### 阶段出口

- `RuntimeContext` 正式持有 `BackendRegistry`
- `RuntimeContext::GetBackend(DeviceType::kCPU)` 可用
- 无全局 backend registry

---

## 5.3 Phase 2：CPU Backend 基础设施

### 目标

建立 Phase 1 唯一真实 backend：CPU backend。

### 建议顺序

#### 第一步：测试先行（可并行）

- `tests/unit/test_cpu_backend.cpp`
- `tests/unit/test_cpu_capabilities.cpp`
- `tests/unit/test_cpu_workspace_arena.cpp`

#### 第二步：头文件冻结（建议串行）

1. `include/aethermind/backend/cpu/cpu_capabilities.h`
2. `src/backend/cpu/cpu_capabilities.cpp`
3. `include/aethermind/backend/cpu/cpu_execution_resources.h`
4. `include/aethermind/backend/cpu/cpu_thread_pool.h`
5. `include/aethermind/backend/cpu/cpu_workspace_arena.h`
6. `include/aethermind/backend/cpu/cpu_backend.h`

#### 第三步：实现层并行窗口

**可并行组 A：能力与线程池**

- `src/backend/cpu/cpu_capabilities.cpp`
- `src/backend/cpu/cpu_thread_pool.cpp`

**可并行组 B：workspace arena**

- `src/backend/cpu/cpu_workspace_arena.cpp`

#### 第四步：CpuBackend 汇总接入（串行）

- `src/backend/cpu/cpu_backend.cpp`

### 阶段出口

- `CpuBackend` 可通过 runtime 获取
- `capabilities()` 返回稳定结果
- `CpuWorkspaceArena` 可基于预分配区域完成运行期绑定

---

## 5.4 Phase 3：Dispatch 主线建立（按 batch 推进）

### 目标

按 dispatch 设计文档中预先冻结的 batches，逐步建立 backend-owned、plan-build-time 的 dispatch 主线。

### 建议顺序

#### 第一步：设计基线与边界冻结（建议串行）

- 冻结 `docs/designs/dispatch_design.md` 作为 Phase 1 dispatch 主线基线
- 明确本阶段必须按 Batch 1 → Batch 2 → Batch 3 → Batch 4 推进，不跨 batch 混做

#### 第二步：Batch 1 最小类型落地（建议串行）

- `include/aethermind/operators/op_type.h`
- `include/aethermind/backend/kernel_selector.h`
- `include/aethermind/backend/kernel_descriptor.h`
- `include/aethermind/backend/resolved_kernel.h`
- `include/aethermind/backend/kernel_key.h`（迁移期保留；明确不再是未来主线核心）
- `include/aethermind/backend/dispatcher_bridge.h`（迁移辅助；不承接未来主线职责）
- `include/operator_name.h`（保留兼容，并为 `OperatorName -> OpType` 过渡映射预留入口）

#### 第三步：测试先行（可并行）

- `tests/unit/test_kernel_registry.cpp`
- `tests/unit/test_cpu_resolve_kernel.cpp`

#### 第四步：Batch 2 selector-based resolve（建议串行）

1. `include/aethermind/backend/kernel_registry.h`
2. `src/backend/kernel_registry.cpp`
3. `include/aethermind/backend/kernel_invocation.h`
4. `src/backend/dispatcher_bridge.cpp`
5. `src/backend/cpu/cpu_backend.cpp`

#### 第五步：Batch 3 / Batch 4 的后续落点（本阶段只冻结边界）

- Batch 3 对齐 `ExecutionPlanBuilder` 阶段：把 resolve 明确拉进计划构建期，并冻结 `ResolvedKernel`
- Batch 4 对齐 executor 接入阶段：切到 direct call，并冻结旧 `dispatcher.h` / `dispatch_key*.h` 体系

### 实施注意事项

- backend 设计要求最终设备族内解析只能通过 `Backend::ResolveKernel(...)` 发起
- `KernelRegistry` 应由具体 backend 持有，而不是退回全局 singleton
- `ExecutionPlanBuilder` 是唯一正式 resolve 发起方
- 不允许把 resolve 留到 executor 热路径
- 不再继续扩展 `Dispatcher / DispatchKeySet` 作为新主线；旧设施在 Batch 4 正式冻结前仅可维持兼容，不得承接新功能

### 阶段出口

- Batch 1：新 dispatch 主线的最小类型与迁移边界完成落地，且不破坏现有编译/测试
- Batch 2：CPU kernel 可注册，`ResolveKernel(...)` 可根据 selector 与 capability 返回正确 `KernelFn`
- Batch 3：计划构建期可冻结 `ResolvedKernel`
- Batch 4：执行期不再触碰 registry

---

## 5.5 Phase 4：Packed Weight Sidecar

### 目标

把逻辑权重与 backend packed 权重彻底分离，并冻结所有权模型。

### 建议顺序

#### 第一步：测试先行（可并行）

- `tests/unit/test_cpu_weight_prepacker.cpp`
- `tests/unit/test_backend_sidecar_ownership.cpp`

#### 第二步：公共类型与 CPU 预打包接口（建议串行）

1. `include/aethermind/backend/packed_weights.h`
2. `include/aethermind/backend/cpu/cpu_weight_prepacker.h`

#### 第三步：实现层局部并行

- `src/backend/cpu/cpu_weight_prepacker.cpp`
- `include/aethermind/model/backend_sidecar.h`

#### 第四步：模型侧生命周期接入（串行）

1. `src/model/backend_sidecar.cpp`
2. `include/aethermind/model/model_instance.h`
3. `src/model/model_instance.cpp`

### 阶段出口

- packed weights 与逻辑权重生命周期分离明确
- packed params 可被执行计划安全引用
- backend 不持有 packed 数据本体

---

## 5.6 Phase 5：ExecutionPlan / OpExec / PlanBuilder

### 目标

真正落实 “init-time resolve, hot-path direct call”。

### 建议顺序

#### 第一步：测试先行（可并行）

- `tests/unit/test_execution_plan_builder.cpp`
- `tests/unit/test_execution_plan.cpp`
- `tests/unit/test_workspace_requirement_planning.cpp`

#### 第二步：核心类型定义（建议严格串行）

1. `include/aethermind/backend/workspace_types.h`
2. `include/aethermind/backend/execution_plan.h`
3. `src/backend/execution_plan.cpp`
4. `include/aethermind/backend/execution_plan_builder.h`
5. `src/backend/execution_plan_builder.cpp`

#### 第三步：相邻接口对齐（可局部并行审视）

- `include/aethermind/operator/op_kind.h`（如需新增）
- `include/operator_name.h`（过渡期可保留；建议提供 `OperatorName -> OpType` 映射而不是继续扩展旧 dispatcher 路线）
- `include/function_schema.h`

### 为什么这是核心串行阶段

- `OpExec` 依赖 workspace 类型、kernel 类型与 packed params 约定
- `ExecutionPlanBuilder` 依赖 resolve 路径与 packed weight sidecar 都已冻结
- 如果提前并行推进 builder 与相邻接口，很容易重复改签名

### 阶段出口

- 每个 op 在计划阶段都能冻结成 `OpExec`
- `ExecutionPlan` 只保存静态执行信息
- 动态地址未进入 `ExecutionPlan`

---

## 5.7 Phase 6：RuntimeBindingContext

### 目标

把 request/session 级动态绑定从 `ExecutionPlan` 中隔离出去。

### 建议顺序

#### 第一步：测试先行（可并行）

- `tests/unit/test_runtime_binding_context.cpp`
- `tests/unit/test_workspace_runtime_binding.cpp`

#### 第二步：核心接口（建议串行）

1. `include/aethermind/backend/runtime_binding_context.h`
2. `src/backend/runtime_binding_context.cpp`
3. `include/aethermind/backend/workspace_arena.h`

#### 第三步：CPU workspace 接入（串行）

- `src/backend/cpu/cpu_workspace_arena.cpp`

#### 第四步：KV 对齐（可并行审视）

- `include/aethermind/execution/kv_cache_view.h`
- `include/aethermind/execution/kv_cache_manager.h`

### 阶段出口

- request/session 动态状态不进入 `ExecutionPlan`
- runtime binding 可为 plan 中各 op 提供运行期地址绑定

---

## 5.8 Phase 7：OpKernelContext 与执行接入

### 目标

为热路径定义稳定的 kernel 调用上下文，并接入 executor direct-call 路径。

### 建议顺序

#### 第一步：测试先行（可并行）

- `tests/unit/test_op_kernel_context.cpp`
- `tests/unit/test_executor_backend_path.cpp`

#### 第二步：执行上下文定义（建议串行）

1. `include/aethermind/backend/op_kernel_context.h`
2. `include/aethermind/backend/stream.h`
3. `include/aethermind/backend/tracing_sink.h`

#### 第三步：Executor / LayerRunner 接入（建议串行）

1. `include/aethermind/execution/executor.h`
2. `src/execution/executor.cpp`
3. `include/aethermind/execution/layer_runner.h`
4. `src/execution/layer_runner.cpp`

#### 第四步：CPU 最小闭环 kernel

- `include/aethermind/backend/cpu/kernels/cpu_rmsnorm_kernel.h`
- `src/backend/cpu/kernels/cpu_rmsnorm_kernel.cpp`
- `src/backend/cpu/cpu_backend.cpp`: 注册最小 kernel 并支持 resolve

### 阶段出口

- kernel 调用仅依赖 `OpKernelContext + WorkspaceBinding`
- executor 热路径只消费 plan
- 至少一个 CPU kernel 完成 direct-call 闭环

---

## 5.9 Phase 8：测试与回归收口

### 目标

为 backend 主线建立 focused regression protection，并验证关键架构约束。

### 建议顺序

#### 第一步：约束型测试（可并行）

- `tests/unit/test_no_hotpath_resolve.cpp`
- `tests/unit/test_execution_plan_immutability.cpp`
- `tests/unit/test_decode_workspace_stability.cpp`
- `tests/unit/test_backend_resource_escape_hatch.cpp`

#### 第二步：集成回归（最后串行）

- `tests/unit/test_runtime_cpu_backend_e2e.cpp`

### 阶段出口

- backend 主线具备 focused tests 覆盖
- ownership / hot-path / zero-allocation 约束可回归验证
- runtime → backend → plan → executor 最小链路打通

---

## 6. 并行化窗口

### 6.1 可以并行推进的典型类型

- 同一阶段内的多个测试文件
- 头文件冻结后的多个独立实现文件
- CPU backend 阶段中 capability / thread pool 与 workspace arena 的独立实现
- packed weight 阶段中的 prepacker 与 sidecar 类型定义
- RuntimeBindingContext 阶段中的 KV 对齐审视工作
- 最终约束型测试的补齐工作

### 6.2 不应并行推进的类型

- phase 主依赖链本身
- `Backend` 抽象层次的签名与 ownership 定义
- `ExecutionPlanBuilder` 主体实现
- `Executor` / `LayerRunner` 责任边界定义
- `CpuBackend` 中 `ResolveKernel(...)` 与最小闭环 kernel 的最终汇总接入

### 6.3 推荐的阶段内并行策略

每个阶段建议采用：

1. 先串行冻结接口
2. 再并行补实现与测试
3. 最后串行汇总验证

这样可以降低签名返工与交叉冲突风险。

---

## 7. 推荐实施批次

为便于 code review、验证与回滚，建议按以下批次推进：

| Batch | 覆盖阶段 | 核心目标 | 建议验收点 |
|------|----------|----------|-----------|
| **Batch A** | Phase 0 + Phase 1 | Runtime 正式持有 backend registry | `RuntimeContext::GetBackend(DeviceType::kCPU)` 可用 |
| **Batch B** | Phase 2 | CPU backend 基础能力与资源站起来 | `CpuBackend` / `capabilities()` / `CpuWorkspaceArena` 可用 |
| **Batch C** | Phase 3 | backend-owned kernel 注册与 plan-build resolve 基线跑通 | CPU kernel 可注册，`ResolveKernel(...)` 可基于 selector 返回稳定 `KernelFn`，旧 `Dispatcher / DispatchKeySet` 进入冻结态 |
| **Batch D** | Phase 4 | packed weight sidecar 所有权冻结 | packed 数据与逻辑权重生命周期分离明确 |
| **Batch E** | Phase 5 | `ExecutionPlanBuilder` 成为唯一 resolve 入口 | `OpExec` / `ExecutionPlan` / workspace planning 跑通 |
| **Batch F** | Phase 6 | 动态绑定从 plan 中剥离 | `RuntimeBindingContext` 可提供运行期地址绑定 |
| **Batch G** | Phase 7 | executor direct-call 最小闭环 | 至少一个 CPU kernel 路径完成 end-to-end direct call |
| **Batch H** | Phase 8 | 约束测试与 e2e 回归收口 | backend focused regression tests 通过 |

---

## 8. 每阶段验证门槛

### 8.1 通用验证门槛

每个阶段完成后，至少执行以下动作：

1. 构建最窄受影响目标
2. 运行最窄 gtest filter
3. 检查没有新引入 diagnostics
4. 复核 checklist 对应项是否已满足

### 8.2 推荐验证命令

```bash
cmake --build build --target AetherMind -j
cmake --build build --target aethermind_unit_tests -j

./build/tests/unit/aethermind_unit_tests --gtest_filter=BackendRegistry.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=RuntimeBackendIntegration.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=CPUBackend.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=CPUCapabilities.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=CPUWorkspaceArena.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=KernelRegistry.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=CPUResolveKernel.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=ExecutionPlan.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=ExecutionPlanBuilder.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=RuntimeBindingContext.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=ExecutorBackendPath.*
```

### 8.3 阶段出口复核问题

每阶段结束时，建议统一复核以下问题：

- 是否把本应静态冻结的信息放进了执行期？
- 是否把本应动态绑定的信息塞进了 `ExecutionPlan`？
- 是否在 executor 热路径重新引入了 registry lookup / resolve / fallback？
- 是否破坏了 packed weight sidecar 的所有权边界？
- 是否破坏了 workspace steady-state zero allocation 约束？

---

## 9. 关键风险与实施注意事项

### 9.1 旧 Dispatcher 相关风险

当前代码库中的旧 dispatcher 已存在，但新版 dispatch 主线不再以它为核心。实施时必须避免：

- 让旧 dispatcher 继续承担最终设备族内 resolve
- 在 executor 热路径中直接访问 dispatcher 或 registry
- 用 dispatcher 逃避 `ExecutionPlanBuilder` 作为唯一 resolve 发起方的约束
- 继续扩展 `DispatchKeySet` 体系并把它重新变成新主线基础

### 9.2 ExecutionPlan 漂移风险

最常见风险是把 workspace base、KV view、临时输出缓冲等动态状态重新塞回 `ExecutionPlan`。一旦发生，会直接破坏：

- 计划静态性
- session/request 隔离
- direct-call 执行边界

### 9.3 Backend 私有资源逃逸风险

`opaque_backend_resources` 只用于承载 backend-private execution resources。必须避免：

- 借此回传整个 runtime 宽对象
- 借此访问 allocator registry / model registry / dispatcher 等上层对象
- 逐步把窄接口演变成隐式的宽对象通道

### 9.4 实施节奏建议

- 不要跨阶段同时开大规模实现
- 不要在 `ExecutionPlanBuilder` 未冻结前提前改 executor 热路径
- 不要为了“先跑通”把 fallback 决策留到执行期
- 不要一次铺开多个 CPU kernel，先完成单个最小闭环

---

## 结语

backend Phase 1 的工程目标不是尽快堆出一套“看起来可扩展”的后端抽象，而是按正确顺序稳定打通这条 CPU-first 主线：

- runtime 装配
- backend capability
- kernel resolve
- packed weight sidecar
- execution plan 冻结
- runtime binding
- executor direct call
- focused regression

只要严格按本文档与 checklist 的串并行边界推进，就可以在保持可验证、可回滚、可 review 的前提下完成 backend Phase 1 主线落地。
