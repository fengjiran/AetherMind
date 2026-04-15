# AetherMind Dispatch 设计执行任务清单

**版本**: v1.0  
**日期**: 2026-04-15  
**作者**: AetherMind Team

---

## 1. 文档目的

本文件将 `docs/designs/dispatch_design.md` 中定义的 **backend-owned、plan-build-time dispatch 主线方案**，进一步拆解成可执行的、按文件粒度组织的 implementation task list。

本清单的定位是：

- 指导 dispatch 新主线如何逐步落地；
- 明确哪些文件需要新增、改造、冻结；
- 明确每一批的完成标准与验证重点；
- 避免旧 `Dispatcher / DispatchKeySet` 在迁移期继续膨胀。

---

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

---

## 3. 批次划分

建议按 4 个 batch 推进：

| Batch | 覆盖阶段 | 核心目标 |
|------|----------|----------|
| **Batch 1** | Phase A + Phase B | 冻结边界，落最小新类型 |
| **Batch 2** | Phase C + Phase D | selector-based `KernelRegistry` + `CpuBackend` 注册/解析 |
| **Batch 3** | Phase E | 计划构建期 resolve 与 `ResolvedKernel` 冻结 |
| **Batch 4** | Phase F + Phase G | Executor direct call + 旧 dispatch 体系冻结 |

---

## 4. Batch 1：冻结边界 + 落最小新类型

### 4.1 目标

建立新 dispatch 主线的最小类型系统，并明确旧 dispatch 体系不再作为未来主线继续扩展。

### 4.2 文件任务清单

#### 文档冻结

- [ ] `docs/designs/dispatch_design.md`: 冻结 dispatch 主线基线
- [ ] `docs/designs/backend_phase1_development_steps.md`: 与新 dispatch 主线保持一致
- [ ] `docs/designs/backend_phase1_implementation_plan.md`: 与新 dispatch 主线保持一致
- [ ] `docs/designs/backend_phase1_implementation_checklist.md`: 与新 dispatch 主线保持一致

#### 新增核心类型

- [ ] `include/aethermind/operators/op_type.h`: 定义 `OpType`
- [ ] `include/aethermind/backend/kernel_selector.h`: 定义 `KernelSelector`、`IsaLevel`、`ExecPhase`、`WeightFormat`
- [ ] `include/aethermind/backend/kernel_descriptor.h`: 定义 `KernelDescriptor`
- [ ] `include/aethermind/backend/resolved_kernel.h`: 定义 `ResolvedKernel`

#### 迁移辅助

- [ ] `include/operator_name.h`: 保持过渡兼容，并为 `OperatorName -> OpType` 映射预留入口
- [ ] `include/aethermind/backend/kernel_key.h`: 标注为迁移期保留类型，而非未来主线核心
- [ ] `include/aethermind/backend/dispatcher_bridge.h`: 标注为迁移辅助，不再承接未来主线职责

### 4.3 验证重点

- 新类型头文件能独立通过编译
- 不破坏现有 Batch A / Batch B 测试
- 文档口径统一为 backend-owned、plan-build-time dispatch

---

## 5. Batch 2：selector-based KernelRegistry + CpuBackend 收敛

### 5.1 目标

把 `KernelRegistry` 从“按 key 查表”演进为“按 selector resolve”，并让 `CpuBackend` 成为新 dispatch 主线的真正 owner。

### 5.2 文件任务清单

#### 核心 registry

- [ ] `include/aethermind/backend/kernel_registry.h`: 新增 `Register(const KernelDescriptor&)` 与 `Resolve(OpType, KernelSelector, ...)`
- [ ] `src/backend/kernel_registry.cpp`: 实现 selector-based resolve、`kBoth` 匹配、ISA 兼容过滤、priority 选择

#### 迁移兼容

- [ ] `include/aethermind/backend/kernel_key.h`: 若短期保留旧接口，明确其仅用于迁移桥接
- [ ] `src/backend/dispatcher_bridge.cpp`: 若保留桥接，实现旧路径到新 `OpType / KernelSelector` 的辅助转换

#### CpuBackend 收敛

- [ ] `include/aethermind/backend/cpu/cpu_backend.h`: 明确 backend 内持有 `KernelRegistry`
- [ ] `src/backend/cpu/cpu_backend.cpp`: 在初始化路径中显式注册 builtin kernels，并接入 selector-based `ResolveKernel(...)`

### 5.3 建议测试

- [ ] `tests/unit/test_kernel_registry.cpp`: 扩展为 descriptor 注册与 resolve 覆盖
- [ ] `tests/unit/test_kernel_selector.cpp`: 新增 selector 匹配测试
- [ ] `tests/unit/test_kernel_registry_resolve.cpp`: 新增 `kBoth`、ISA、priority resolve 测试
- [ ] `tests/unit/test_cpu_resolve_kernel.cpp`: 扩展 builtin kernel 注册与 selector resolve 覆盖

### 5.4 验证重点

- `KernelRegistry` 可按 selector resolve
- `ExecPhase::kBoth` 可匹配 `kPrefill` / `kDecode`
- `priority` 只在“多个可用实现”中排序
- `CpuBackend` 使用 backend-owned registry，不引入全局 singleton

---

## 6. Batch 3：计划构建期 resolve 与 ResolvedKernel 冻结

### 6.1 目标

把 resolve 明确拉进计划构建期，使 `ExecutionPlanBuilder` 成为唯一正式 resolve 发起方。

### 6.2 文件任务清单

#### 执行计划构建侧

- [ ] `include/aethermind/backend/execution_plan_builder.h`: 定义 `ResolveKernelForNode(...)` 或等价接口
- [ ] `src/backend/execution_plan_builder.cpp`: 基于 `OpType + KernelSelector + Backend::ResolveKernel(...)` 完成计划构建期 resolve

#### 冻结结果类型

- [ ] `include/aethermind/backend/resolved_kernel.h`: 补齐 `attrs` / `attrs_size` / `debug_name` 生命周期约束说明
- [ ] `include/aethermind/backend/execution_plan.h`: 接入 `ResolvedKernel` 或与 `OpExec` 做等价衔接
- [ ] `src/backend/execution_plan.cpp`: 落实冻结结果存储

#### 迁移辅助

- [ ] `include/operator_name.h` 或相邻映射文件：提供 `OperatorName -> OpType` 过渡映射

### 6.3 建议测试

- [ ] `tests/unit/test_execution_plan_builder.cpp`: 覆盖 plan-build resolve
- [ ] `tests/unit/test_execution_plan.cpp`: 覆盖 `ResolvedKernel` / `OpExec` 冻结结果

### 6.4 验证重点

- `ExecutionPlanBuilder` 是唯一 resolve 发起方
- 计划构建期可冻结 `ResolvedKernel`
- attrs 生命周期覆盖整个 execution plan
- 执行节点不再依赖 runtime lookup

---

## 7. Batch 4：Executor direct call + 旧体系冻结

### 7.1 目标

让 `Executor` 热路径只做函数指针调用，并正式冻结旧 `Dispatcher / DispatchKeySet` 体系。

### 7.2 文件任务清单

#### Executor / LayerRunner

- [ ] `include/aethermind/execution/executor.h`: 接入 `ResolvedKernel` / `ExecutionPlan`
- [ ] `src/execution/executor.cpp`: 执行期 direct call，不做 runtime resolve
- [ ] `include/aethermind/execution/layer_runner.h`: 接入已冻结 kernel 执行接口
- [ ] `src/execution/layer_runner.cpp`: 遍历节点并 direct call `KernelFn`

#### 旧 dispatch 体系冻结

- [ ] `include/dispatcher.h`: 标注文档与职责边界，明确不再作为新算子实现主线
- [ ] `src/dispatcher.cpp`: 不再增加 runtime resolve 责任
- [ ] `include/dispatch_key.h`: 标注冻结/待退场
- [ ] `include/dispatch_key_set.h`: 标注冻结/待退场，不再扩展

### 7.3 建议测试

- [ ] `tests/unit/test_executor_backend_path.cpp`: 覆盖 executor direct call
- [ ] `tests/unit/test_no_hotpath_resolve.cpp`: 断言执行路径不做 registry / dispatcher lookup

### 7.4 验证重点

- executor 热路径 direct call
- 无 runtime resolve
- 无 hot-path registry access
- 新功能全部走新主线

---

## 8. 风险控制

### 8.1 不要做的事

- [ ] 不要一次性删除 `OperatorName`
- [ ] 不要先删旧 dispatcher 再补新主线
- [ ] 不要把全局 singleton registry 引回来
- [ ] 不要把 `RuntimeContext*` 直接传给 kernel
- [ ] 不要让 executor 在迁移期继续做 hash/string lookup

### 8.2 必须守住的约束

- [ ] backend-owned ownership
- [ ] narrow `KernelContext`
- [ ] attrs 生命周期明确
- [ ] resolve 只发生在 plan-build time

---

## 9. 一句话执行策略

> **先立新主线，再迁移 resolve，最后冻结旧 dispatcher。**
