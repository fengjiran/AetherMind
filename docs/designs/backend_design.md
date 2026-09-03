# AetherMind Backend 层设计文档

**版本**: v1.2  
**日期**: 2026-09-01  
**作者**: AetherMind Team

---

## 目录

1. [概述](#1-概述)
2. [职责与范围](#2-职责与范围)
3. [设计目标与核心原则](#3-设计目标与核心原则)
4. [分层定位与核心关系](#4-分层定位与核心关系)
5. [核心对象与所有权模型](#5-核心对象与所有权模型)
6. [三阶段执行模型：注册、构建与执行](#6-三阶段执行模型注册构建与执行)
7. [ExecutionPlan 与 ResolvedKernel 设计](#7-executionplan-与-resolvedkernel-设计)
8. [Backend 核心接口设计](#8-backend-核心接口设计)
9. [CPU Backend 内部组成](#9-cpu-backend-内部组成)
10. [Workspace、Stream 与 KV 契约](#10-workspace-stream-与-kv-契约)
11. [Phase 1 落地策略与演进边界](#11-phase-1-落地策略与演进边界)
12. [验证与测试建议](#12-验证与测试建议)

---

## 1. 概述

### 1.1 文档目的

本文档定义 AetherMind 在 `Runtime`、`AllocatorRegistry`、`Buffer`、`Tensor` 之后的 Backend 层设计，用于冻结以下内容：

- Backend 层在整体架构中的职责边界
- Runtime 与 Backend 的所有权关系
- **三阶段执行模型**（注册、计划构建、执行）
- **ExecutionPlan / ResolvedKernel** 核心抽象
- **CPU Backend** 落地组成与权重预打包模型
- Workspace、Stream、KV Cache 物理契约

### 1.2 一句话定义

**Backend 层是 Runtime 持有的设备族计算执行层，负责为特定设备族提供 capability、kernel 集合、权重预处理、workspace 与执行资源接缝，并在初始化/计划构建阶段完成 kernel 解析，使执行热路径退化为直接函数调用；Backend 不拥有 Tensor/Buffer 生命周期，也不承担推理控制流。**

---

## 2. 职责与范围

### 2.1 Backend 层核心职责

- 表达某个设备族的执行能力（CPU、CUDA、CANN 等）
- 提供 Kernel 集合注册入口与设备能力探测（Capabilities）
- 负责权重的预打包（Weight Prepacking），以适应特定硬件访问模式
- 定义 KV cache 的物理布局与 kernel 访问契约
- 在计划构建阶段（Build-time）参与 Kernel 解析与资源绑定
- 暴露最小执行接缝（Stream, Workspace Slice）给 Kernel

### 2.2 Backend 层非职责

- 不持有 `Tensor` 或 `Buffer` 的生命周期
- 不替代 `AllocatorRegistry`（仅通过 service 消费）
- 不直接承担完整的推理控制流（推理流控属于 `Executor`）
- 不使用全局静态单例维护注册表
- 不在热路径中引入字符串查表或层层虚函数分发
- 不隐式完成跨 backend 的 fallback

---

## 3. 设计目标与核心原则

### 3.1 设计目标

#### 3.1.1 Runtime 持有，显式装配
Backend 层必须由 `RuntimeBuilder` 显式装配，并由 `Runtime` 持有 `BackendRegistry`。当前 `KernelRegistry::Global()` 是冻结后的 process-wide descriptor catalog；它只在 backend 的 resolve 路径使用，不能被混同为 Runtime 的资源 owner 或进入执行热路径。

#### 3.1.2 CPU First，稳态零分配
Phase 1 优先支持 CPU Backend。Workspace 接口必须支持预分配切片借用语义，以兼容 decode 阶段稳态零分配约束。

#### 3.1.3 Init-time Resolve, Hot-path Direct Call
Kernel 的最终执行形态必须是 plan-build time resolve 后的函数指针，执行阶段严禁查表。

#### 3.1.4 窄执行上下文
不把整个 `RuntimeServices` 下放到 Kernel。单次调用只携带执行所需的窄资源（Stream、Workspace 切片、params/attrs）。

### 3.2 核心原则
- **Runtime owns registries**
- **Builder assembles, PlanBuilder builds, Backend executes**
- **Tensor is data, Backend is execution capability**
- **Explicit resolution, implicit performance**
- **CPU fallback must be explicit, not implicit**

---

## 4. 分层定位与核心关系

### 4.1 关键关系

1. `RuntimeBuilder` 注册 `BackendFactory`。
2. `Runtime` 持有 `BackendRegistry`。
3. `BackendRegistry` 延迟创建并缓存各 `DeviceType` 的 `Backend` 实例。
4. `ExecutionPlanBuilder`（或等价初始化组件）基于 `Runtime` 查询 `Backend`，完成 kernel resolve 与 packed params 绑定。
5. `Executor` 只消费已经冻结的 `ExecutionPlan`，不在热路径重新解析 kernel。
6. `Backend` 通过受控服务视图使用 allocator / workspace 等运行时服务。

---

## 5. 核心对象与所有权模型

### 5.1 所有权模型

| 对象                    | 持有者                             | 生命周期                 | 说明                                                      |
| ----------------------- | ---------------------------------- | ------------------------ | --------------------------------------------------------- |
| `BackendFactory`        | `BackendRegistry`                  | 随 `Runtime` 销毁 | 按 `DeviceType` 注册                                      |
| `Backend`               | `BackendRegistry`                  | 随 `Runtime` 销毁 | 延迟实例化并缓存，表示设备族执行能力                      |
| `ExecutionPlan`                 | 调用方/模型管理侧       | 模型级、只读       | 保存解析后的 `ExecutionStep` 与静态执行元数据；其 resolve Runtime/backend 必须仍存活 |
| `PackedWeights`                 | `PackedWeightStore`     | 与模型级 artifact 一致 | 由 Backend 定义格式并构建，但不由 Backend 持有 |
| `PreparedExecutionBindings`     | `ExecutionContext`      | plan specialization | 拥有 activation、metadata 与 prepared params；借用 external tensor backing |
| `ExecutionContext`              | `Session` / `Request`  | 会话级             | 按值拥有 prepared bindings，借用 workspace arena，保存 KVCacheView；不保存临时输出 buffer 或 sequence state |
| `KernelContext`                 | 执行栈帧                | 短生命周期          | 单次调用的窄执行上下文 |

#### 5.1.1 所有权约束

- `Backend` 表示设备族执行能力，不拥有模型权重数据本身。
- `PackedWeights` 必须由 `PackedWeightStore` 持有，禁止写成 “`Backend` 或 store 二选一”。
- `ExecutionPlan` 是只读的静态执行计划，不承载 request/session 相关的动态地址绑定。
- `PrepareExecutionBindings` 在冷路径将 external TensorViews specialize 为 `PreparedExecutionBindings`；其 external data backing 必须比 bindings 活得久。
- `ExecutionContext::Create` 按值接收 prepared bindings、借用 workspace arena 并保存 KV view。`Clear()` 只清除自身 owned/borrowed handles，不 reset arena 或 release KV reservation。
- Phase 1 默认 `ExecutionPlan` 为模型级不可变计划；若未来出现 request-specific specialization，也只能在其上派生轻量 binding 或 session 级附加对象，不得回写主计划。

---

## 6. 三阶段执行模型：注册、构建与执行

为了确保热路径的高性能，Backend 参与的执行逻辑必须划分为三个阶段：

### 6.1 注册期 (Registration Time)
- Backend 在初始化时通过内部 `KernelRegistry` 安装本设备族支持的 kernels：各内核 TU 通过 `AM_REGISTER_KERNEL` 静态注册 `KernelDescriptor`，首次 `CpuBackend` 实例化时冻结注册表（`Freeze()`）并构建按 `OpType` 分桶的索引。
- `BackendRegistry` 由 `Runtime` 持有（无全局单例）；`KernelRegistry::Global()` 是 backend 模块内部、freeze 后只读的 process-wide descriptor catalog。

### 6.2 计划构建期 (Plan-Build Time)

- `ExecutionPlanBuilder` 基于模型结构、设备信息、数据类型和 layout trait，通过 backend 完成 Kernel 解析。
- `ExecutionPlanBuilder` 通过 `Backend::PrepareKernel(...)` 获得 `ResolvedKernel`，而不是直接操作 `KernelRegistry`。
- CPU backend 使用自身持有的 immutable `CpuCapabilities` snapshot 完成 feature eligibility 检查；通用 `Backend` 接口不暴露无消费者的统一 capability view。
- 将解析结果冻结为 `ExecutionStep`：内置 `ResolvedKernel`（函数指针、attrs、params builder、workspace 要求）以及指向 `PackedWeights` 的指针、预期 packing recipe。
- 此阶段完成所有 fallback 决策、能力适配与静态 workspace 规划。

### 6.3 执行期 (Execution Time)

- `Executor` 仅遍历 `ExecutionPlan` 中的 `ExecutionStep` 序列。
- 执行期不做 registry 查表，不做 capability 判断，不做 fallback 决策。
- 执行期基于 `ExecutionContext` 将 `WorkspaceRequirement` 绑定为本次运行可用的 `WorkspaceBinding`。
- 每个算子通过 `KernelContext` + 已绑定的 workspace 直接调用 `KernelFunc`。

---

## 7. ExecutionPlan 与 ResolvedKernel 设计

### 7.1 WorkspaceRequirement 与 WorkspaceBinding

```cpp
struct WorkspaceRequirement {
    size_t bytes = 0;
    size_t alignment = 64;
    size_t offset = 0;   // 若已完成静态规划，则表示相对于 workspace base 的偏移
};

struct WorkspaceBinding {
    void* data = nullptr;
    size_t size = 0;
};
```

### 7.2 ResolvedKernel 与 ExecutionStep

```cpp
// 类型擦除的内核入口；一次调用携带 KernelContext，从 kernel_params / attrs 读取输入与参数
using KernelFunc = Status (*)(const KernelContext&) noexcept;

struct ResolvedKernel {
    OpType op_type = OpType::kUnknown;
    KernelFunc fn = nullptr;
    std::vector<std::byte> attrs{};               // 规划期由语义参数构建的不变元数据
    const char* debug_name = nullptr;
    KernelParamsBuilder params_builder = nullptr; // 执行期由 tensor 绑定构建 kernel 专属 params
    size_t params_size = 0;
    WorkspaceRequirement workspace_requirement{}; // 规划期冻结的 scratch 要求
    PackingRecipe expected_packing_recipe{};      // packed 权重所需的打包布局；非 packed 为空
};
```

### 7.3 ExecutionPlan

```cpp
class ExecutionPlan {
public:
    const std::vector<ExecutionStep>& steps() const noexcept;  // 已冻结的执行步序列
    const std::vector<ExecutionValueDesc>& values() const noexcept;
    size_t total_workspace_bytes() const noexcept;             // 规划期 workspace 总量
    size_t workspace_alignment() const noexcept;
    // ... 其他只读视图接口
};
```

### 7.4 ExecutionPlan 推荐归属口径

Phase 1 推荐将 `ExecutionPlan` 视为**模型级不可变计划**。
其与 session/request 相关的动态绑定信息不放入 `ExecutionPlan`：external tensors、activation storage 和 prepared params 属于 `PreparedExecutionBindings`；workspace arena 与 KV view 由 `ExecutionContext` 聚合。模型 I/O 仍必须经 `ExternalTensorBindings` 显式绑定，而非另设临时输出缓冲通道。


---

## 8. Backend 核心接口设计

### 8.1 Backend

```cpp
class Backend {
public:
    virtual ~Backend() = default;

    virtual DeviceType device_type() const noexcept = 0;

    // 唯一正式的 Kernel 解析入口，供 ExecutionPlanBuilder 使用
    virtual StatusOr<ResolvedKernel> PrepareKernel(
        OpType op_type,
        const KernelSelector& selector,
        const OpParams& params) const = 0;

    // 仅用于调试/自省，不作为执行构建主入口
    virtual const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept = 0;
};
```

#### 8.1.1 说明

- `PrepareKernel(...)` 是唯一正式的 kernel 解析入口。
- `ExecutionPlanBuilder` 不得绕过 `Backend` 直接访问 `KernelRegistry`。
- `KernelRegistry` 作为 backend 内部设施存在，`TryGetKernelRegistryForDebug()` 仅用于调试、自省或测试。

### 8.2 KernelContext

```cpp
struct KernelContext {
    DeviceType device_type = DeviceType::kUndefined;
    Stream* stream = nullptr;                    // Phase 1 为 CpuInlineStream 占位
    WorkspaceArena* workspace = nullptr;        // 提供切片借用与绑定支持
    WorkspaceBinding workspace_binding{};        // 本次调用绑定的 workspace 切片
    const void* packed_weights = nullptr;        // packed 权重数据起始指针
    const void* kernel_params = nullptr;         // kernel 专属 params（执行期构建）
    std::span<const std::byte> attrs{};          // 规划期冻结的不变元数据
};
```

#### 8.2.1 说明

- `KernelContext` 只携带窄执行资源；stream、workspace、params/attrs、packed weights 均为执行所需的窄句柄。
- 后端专属执行资源（线程池、NUMA/ISA 辅助等）为后续扩展预留接缝；当前 `CpuBackend` 仅持有能力快照。
- 后续 CUDA / CANN backend 可复用同一通用接口，而无需向 `KernelContext` 追加后端专属字段（必要时再引入细分的执行资源传递通道）。

---

## 9. CPU Backend 内部组成

Phase 1 CPU Backend 需实现以下关键组件以支持高性能推理：


- **CpuCapabilities / CpuFeaturePolicy**：负责检测 AVX2、AVX512、AMX 等指令集支持，并生成三层 capability 快照（hardware/usable/effective）。模型详见 `docs/designs/cpu_capability_design.md`。
- **执行资源接缝（预留）**：线程池 / NUMA / ISA 辅助信息等后端专属执行资源为后续扩展预留；当前 CPU kernels 直接消费 `KernelContext` 中的窄资源。
- **CpuWeightPrepacker**：负责将逻辑权重转换为符合 CPU 指令集与缓存友好布局的 packed 格式。
- **PackedWeights**：预打包权重的存储实体，**由 `PackedWeightStore` 持有**；CPU backend 只定义 packed 格式与构建逻辑。
- **CpuWorkspaceArena**：实现基于预分配 buffer 的切片借用与按 offset 绑定逻辑。

---

## 10. Workspace、Stream 与 KV 契约

### 10.1 Workspace (Arena 语义)

为了支持 steady-state zero allocation，Workspace 采用“规划信息 + 运行时绑定”两段式语义：

```cpp
struct WorkspaceRequirement {
    size_t bytes = 0;
    size_t alignment = 64;
    size_t offset = 0;
};

struct WorkspaceBinding {
    void* data = nullptr;
    size_t size = 0;
};

class WorkspaceArena {
public:
    virtual ~WorkspaceArena() = default;

    // 根据 requirement 或 offset 返回本次运行可用的 binding
    virtual WorkspaceBinding Bind(const WorkspaceRequirement& req) noexcept = 0;

    // 允许在 request / layer 边界重置运行期状态
    virtual void Reset() noexcept = 0;
};
```

#### 10.1.1 设计约束

- `ExecutionPlan` 中只冻结 `WorkspaceRequirement`，不冻结具体地址。
- 具体地址绑定在 session/runtime 准备阶段通过 `WorkspaceArena::Bind(...)` 获得。
- `WorkspaceArena::Bind(...)` 不得触发新的底层堆分配，只能在预分配 arena 内完成地址映射或切片绑定。
- 任何 kernel 不得在热路径自行向 allocator 申请临时内存。

### 10.2 Stream
Phase 1 中 `Stream` 为最小占位接口。CPU 提供 `CpuInlineStream` 实现同步串行语义，仅为 Phase 2 兼容点。

### 10.3 KV Cache 物理契约
`KVCacheManager` 负责逻辑索引与视图组织，但 **KV cache 的物理布局、对齐规则与 kernel 访问契约由 Backend 定义并冻结**。例如，CPU Backend 可能要求 K/V 按照 head-size 对齐或采用特定的转置存储以便于向量化。

---

## 11. Phase 1 落地策略与演进边界

### 11.1 Phase 1 推荐落地顺序

#### 阶段 A：Backend 骨架与 Runtime 装配
- `BackendFactory` / `BackendRegistry`
- `RuntimeBuilder` 扩展注册入口
- `Runtime::GetBackend()`

#### 阶段 B：CpuBackend 执行核心
- `CpuCapabilities` 与指令集探测（`cpu_info.cpp`）
- 执行资源接缝（线程池 / NUMA / ISA 辅助，预留）
- `CpuWorkspaceArena`
- `CpuWeightPrepacker`

#### 阶段 C：ExecutionPlan 冻结
- `ExecutionPlan` 与 `ResolvedKernel`/`ExecutionStep` 定义
- 计划构建阶段的 Resolve 逻辑
- `WorkspaceRequirement` 静态规划与 `WorkspaceBinding` 运行期绑定

#### 阶段 D：Execution 层接入
- `LayerRunner` / `Executor` 遍历 Plan 执行
- 热路径 Direct Call 验证

### 11.2 Phase 1 边界
- **CPU-first**: 同步阻塞执行为主，Stream 仅占位。
- **Static Resolution**: 不支持执行期的动态算子替换。
- **No Hot-path Global Resolution**: `KernelRegistry::Global()` 仅服务 registration/resolve；Runtime 仍拥有 backend instance registry，执行期不查询任一 registry。

### 11.3 Phase 1 冻结合同

- `ExecutionPlanBuilder` 是唯一的 kernel resolve 发起方。
- `Executor` 不做 kernel resolve，不直接访问 `KernelRegistry`。
- `PackedWeights` 由 `PackedWeightStore` 持有。
- `ExecutionPlan` 只保存静态执行信息；`PreparedExecutionBindings` 保存 tensor specialization，`ExecutionContext` 保存 workspace 与 KV view。
- `Workspace` 采用 arena 借用语义；计划期冻结 requirement/offset，执行期完成地址绑定。
- `KernelContext` 只携带窄执行资源（stream、workspace 切片、packed weights、params/attrs），不暴露后端专属类型。
- `KVCacheManager` 仅负责逻辑索引与视图组织，物理契约由 backend 定义。

---

## 12. 验证与测试建议

### 12.1 验收点
- [ ] `Runtime` 持有 `BackendRegistry`
- [ ] 执行热路径不存在 `std::unordered_map::find` 或虚函数查表
- [ ] Workspace 内存地址在稳态推理（Decode）阶段保持不变
- [ ] CPU 指令集特征在 `CpuBackend::PrepareKernel()` 中完成 eligibility 检查，执行期 kernel 不再读取 capability

### 12.2 回归关注点
- 避免 Runtime / Backend 循环引用。
- 确保 `PackedWeights` 在多模型并行时的所有权清晰。
- 验证 KV cache 对齐规则是否被 Backend 正确强制。

---

## 结语

Backend 层的目标不是引入一套泛化过度的设备框架，而是在现有基础上建立面向设备族的计算执行层。对 Phase 1 而言，重点是 CPU 路径的可落地性：能力探测、权重预打包、执行计划冻结，以及热路径上的直接函数调用。
