# AetherMind Backend 层设计文档

**版本**: v1.1  
**日期**: 2026-04-13  
**作者**: AetherMind Team

---

## 目录

1. [概述](#1-概述)
2. [职责与范围](#2-职责与范围)
3. [设计目标与核心原则](#3-设计目标与核心原则)
4. [分层定位与核心关系](#4-分层定位与核心关系)
5. [核心对象与所有权模型](#5-核心对象与所有权模型)
6. [三阶段执行模型：注册、构建与执行](#6-三阶段执行模型注册构建与执行)
7. [ExecutionPlan 与 OpExec 设计](#7-executionplan-与-opexec-设计)
8. [Backend 核心接口设计](#8-backend-核心接口设计)
9. [CPU Backend 内部组成](#9-cpu-backend-内部组成)
10. [Workspace、Stream 与 KV 契约](#10-workspace-stream-与-kv-契约)
11. [Phase 1 落地策略与演进边界](#11-phase-1-落地策略与演进边界)
12. [验证与测试建议](#12-验证与测试建议)

---

## 1. 概述

### 1.1 文档目的

本文档定义 AetherMind 在 `RuntimeContext`、`AllocatorRegistry`、`Buffer`、`Tensor` 之后的 Backend 层设计，用于冻结以下内容：

- Backend 层在整体架构中的职责边界
- Runtime 与 Backend 的所有权关系
- **三阶段执行模型**（注册、计划构建、执行）
- **ExecutionPlan / OpExec** 核心抽象
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
Backend 层必须由 `RuntimeBuilder` 显式装配，并由 `RuntimeContext` 持有，禁止退回到全局注册表模式。

#### 3.1.2 CPU First，稳态零分配
Phase 1 优先支持 CPU Backend。Workspace 接口必须支持预分配切片借用语义，以兼容 decode 阶段稳态零分配约束。

#### 3.1.3 Init-time Resolve, Hot-path Direct Call
Kernel 的最终执行形态必须是 plan-build time resolve 后的函数指针，执行阶段严禁查表。

#### 3.1.4 窄执行上下文
不把整个 `RuntimeServices` 下放到 Kernel。单次调用只携带执行所需的窄资源（Stream, Workspace Slice, Tracing Sink）。

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
2. `RuntimeContext` 持有 `BackendRegistry`。
3. `BackendRegistry` 延迟创建并缓存各 `DeviceType` 的 `Backend` 实例。
4. `ExecutionPlanBuilder`（或等价初始化组件）基于 `RuntimeContext` 查询 `Backend`，完成 kernel resolve 与 packed params 绑定。
5. `Executor` 只消费已经冻结的 `ExecutionPlan`，不在热路径重新解析 kernel。
6. `Backend` 通过受控服务视图使用 allocator / workspace / tracing 等运行时服务。

---

## 5. 核心对象与所有权模型

### 5.1 所有权模型

| 对象                    | 持有者                             | 生命周期                 | 说明                                                      |
| ----------------------- | ---------------------------------- | ------------------------ | --------------------------------------------------------- |
| `BackendFactory`        | `BackendRegistry`                  | 随 `RuntimeContext` 销毁 | 按 `DeviceType` 注册                                      |
| `Backend`               | `BackendRegistry`                  | 随 `RuntimeContext` 销毁 | 延迟实例化并缓存，表示设备族执行能力                      |
| `ExecutionPlan`         | `ModelInstance`                    | 模型实例级、只读         | 保存解析后的 `OpExec` 与静态执行元数据                    |
| `PackedWeights`         | `ModelInstance` 的 backend sidecar | 与模型实例一致           | 由 Backend 定义格式并构建，但不由 Backend 持有            |
| `RuntimeBindingContext` | `Session` / `Request`              | 会话级                   | 保存 workspace base、KV views、临时输出缓冲等动态绑定信息 |
| `OpKernelContext`       | 执行栈帧                           | 短生命周期               | 单次调用的窄执行上下文                                    |

#### 5.1.1 所有权约束

- `Backend` 表示设备族执行能力，不拥有模型权重数据本身。
- `PackedWeights` 必须由 `ModelInstance` 的 backend sidecar 持有，禁止写成 “`Backend` 或 `ModelInstance` 二选一”。
- `ExecutionPlan` 是只读的静态执行计划，不承载 request/session 相关的动态地址绑定。
- 与某次运行相关的动态地址、KV 视图、workspace base 等信息，统一放入 `RuntimeBindingContext`。
- Phase 1 默认 `ExecutionPlan` 为 `ModelInstance` 级不可变计划；若未来出现 request-specific specialization，也只能在其上派生轻量 binding 或 session 级附加对象，不得回写主计划。

---

## 6. 三阶段执行模型：注册、构建与执行

为了确保热路径的高性能，Backend 参与的执行逻辑必须划分为三个阶段：

### 6.1 注册期 (Registration Time)
- Backend 在初始化时通过内部 `KernelRegistry` 安装本设备族支持的 kernels。
- `Dispatcher` 记录 operator 到 dispatch metadata 的静态注册关系，而不是直接承担最终设备族内解析。

### 6.2 计划构建期 (Plan-Build Time)

- `ExecutionPlanBuilder` 基于模型结构、设备信息、数据类型、layout trait 及 `BackendCapabilities` 完成 Kernel 解析。
- `ExecutionPlanBuilder` 通过 `Backend::ResolveKernel(...)` 获得 `KernelFn`，而不是直接操作 `KernelRegistry`。
- `Backend::ResolveKernel(...)` 基于 `Dispatcher` 提供的静态 dispatch metadata 与 backend capability 完成最终设备族内解析。
- 将解析结果冻结为 `OpExec`：包含 `KernelFn`、指向 `PackedWeights` 的指针、`WorkspaceRequirement`、以及用于调试/Tracing 的 `OpKind`。
- 此阶段完成所有 fallback 决策、能力适配与静态 workspace 规划。

### 6.3 执行期 (Execution Time)

- `Executor` 仅遍历 `ExecutionPlan` 中的 `OpExec` 序列。
- 执行期不做 registry 查表，不做 capability 判断，不做 fallback 决策。
- 执行期基于 `RuntimeBindingContext` 将 `WorkspaceRequirement` 绑定为本次运行可用的 `WorkspaceBinding`。
- 每个算子通过 `OpKernelContext` + 已绑定的 workspace 直接调用 `KernelFn`。

---

## 7. ExecutionPlan 与 OpExec 设计

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

### 7.2 OpExec

```cpp
using KernelFn = Status (*)(const KernelInvocation&,
                            const OpKernelContext&,
                            const WorkspaceBinding&) noexcept;

struct OpExec {
    KernelFn fn = nullptr;
    const void* packed_params = nullptr;          // 指向 ModelInstance sidecar 中的 packed 数据
    WorkspaceRequirement workspace_req{};         // 规划期冻结的信息，而非具体地址
    OpKind op_kind{};                             // 调试与 tracing 用
};
```

### 7.3 ExecutionPlan

```cpp
class ExecutionPlan {
public:
    std::span<const OpExec> ops_for_layer(size_t layer_idx) const noexcept;
    // ... 其他只读视图接口
};
```

### 7.4 ExecutionPlan 推荐归属口径

Phase 1 推荐将 `ExecutionPlan` 视为 **`ModelInstance` 级不可变计划**。
其与 session/request 相关的动态绑定信息，例如 workspace base、KV views、临时输出缓冲等，不放入 `ExecutionPlan`，而由 `RuntimeBindingContext` 单独持有。


---

## 8. Backend 核心接口设计

### 8.1 Backend

```cpp
class Backend {
public:
    virtual ~Backend() = default;

    virtual DeviceType device_type() const noexcept = 0;
    virtual const BackendCapabilities& capabilities() const noexcept = 0;

    // 唯一正式的 Kernel 解析入口，供 ExecutionPlanBuilder 使用
    virtual KernelFn ResolveKernel(const KernelKey& key) const noexcept = 0;

    // 仅用于调试/自省，不作为执行构建主入口
    virtual const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept = 0;
};
```

#### 8.1.1 说明

- `ResolveKernel(...)` 是唯一正式的 kernel 解析入口。
- `ExecutionPlanBuilder` 不得绕过 `Backend` 直接访问 `KernelRegistry`。
- `KernelRegistry` 作为 backend 内部设施存在，`TryGetKernelRegistryForDebug()` 仅用于调试、自省或测试。

### 8.2 BackendExecutionResources 与 OpKernelContext

```cpp
struct BackendExecutionResources {
    void* opaque_backend_resources = nullptr;
};

struct OpKernelContext {
    Device device{};
    Stream* stream = nullptr;                    // Phase 1 可为 CpuInlineStream
    WorkspaceArena* workspace = nullptr;        // 提供切片借用与绑定支持
    TracingSink* tracing = nullptr;
    const BackendCapabilities* caps = nullptr;  // 只读能力视图
    BackendExecutionResources backend_resources{};
};
```

#### 8.2.1 说明

- `OpKernelContext` 不直接暴露 `CpuThreadPool*` 等后端专属类型。
- CPU backend 可约定 `opaque_backend_resources` 指向 `CpuExecutionResources`，其中再包含 `CpuThreadPool*`、NUMA/ISA 辅助信息等。
- 后续 CUDA / CANN backend 可复用同一通用接口，而无需继续向 `OpKernelContext` 追加后端专属字段。
- `opaque_backend_resources` 只用于传递 backend-private execution resources，不得借此回传整个 runtime、model、registry 等宽对象指针，也不得成为绕过窄接口约束的逃逸口。

---

## 9. CPU Backend 内部组成

Phase 1 CPU Backend 需实现以下关键组件以支持高性能推理：


- **CpuCapabilities / CpuIsaDetector**：负责检测 AVX2、AVX512、AMX 等指令集支持，并生成 capability 视图。
- **CpuThreadPool**：专为推理优化的线程池，由 `CpuExecutionResources` 持有并通过 `opaque_backend_resources` 暴露给 CPU kernels。
- **CpuWeightPrepacker**：负责将逻辑权重转换为符合 CPU 指令集与缓存友好布局的 packed 格式。
- **PackedWeights**：预打包权重的存储实体，**由 `ModelInstance` 的 backend sidecar 持有**；CPU backend 只定义 packed 格式与构建逻辑。
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
- `RuntimeContext::GetBackend()`

#### 阶段 B：CpuBackend 执行核心
- `CpuCapabilities` 与指令集探测
- `CpuThreadPool`
- `CpuWorkspaceArena`
- `CpuWeightPrepacker`

#### 阶段 C：ExecutionPlan 冻结
- `ExecutionPlan` 与 `OpExec` 定义
- 计划构建阶段的 Resolve 逻辑
- `WorkspaceRequirement` 静态规划与 `WorkspaceBinding` 运行期绑定

#### 阶段 D：Execution 层接入
- `LayerRunner` / `Executor` 遍历 Plan 执行
- 热路径 Direct Call 验证

### 11.2 Phase 1 边界
- **CPU-first**: 同步阻塞执行为主，Stream 仅占位。
- **Static Resolution**: 不支持执行期的动态算子替换。
- **No Global Registry**: 严格遵循 Runtime 所有权。

### 11.3 Phase 1 冻结合同

- `ExecutionPlanBuilder` 是唯一的 kernel resolve 发起方。
- `Executor` 不做 kernel resolve，不直接访问 `KernelRegistry`。
- `PackedWeights` 由 `ModelInstance` 的 backend sidecar 持有。
- `ExecutionPlan` 只保存静态执行信息；workspace、KV view、临时输出缓冲等动态绑定信息由 `RuntimeBindingContext` 持有。
- `Workspace` 采用 arena 借用语义；计划期冻结 requirement/offset，执行期完成地址绑定。
- `OpKernelContext` 不直接暴露后端专属类型；后端专属执行资源通过 `opaque_backend_resources` 下传。
- `KVCacheManager` 仅负责逻辑索引与视图组织，物理契约由 backend 定义。

---

## 12. 验证与测试建议

### 12.1 验收点
- [ ] `RuntimeContext` 持有 `BackendRegistry`
- [ ] 执行热路径不存在 `std::unordered_map::find` 或虚函数查表
- [ ] Workspace 内存地址在稳态推理（Decode）阶段保持不变
- [ ] CPU 指令集特征能正确通过 `capabilities()` 传递给 Kernel

### 12.2 回归关注点
- 避免 Runtime / Backend 循环引用。
- 确保 `PackedWeights` 在多模型并行时的所有权清晰。
- 验证 KV cache 对齐规则是否被 Backend 正确强制。

---

## 结语

Backend 层的目标不是引入一套泛化过度的设备框架，而是在现有基础上建立面向设备族的计算执行层。对 Phase 1 而言，重点是 CPU 路径的可落地性：能力探测、权重预打包、执行计划冻结，以及热路径上的直接函数调用。
