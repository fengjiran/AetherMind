下面是一版适合当前推理引擎、并且**对齐现有 Backend 架构约束**的 **轻量化 Dispatch 模块设计草案**。

# 0. 变更记录

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-04-15 | 初始设计冻结：backend-owned KernelRegistry |
| v1.1 | 2026-06-04 | **设计偏离**：实际实现采用全局 singleton KernelRegistry + `AM_REGISTER_KERNEL` 静态注册宏。原因见第 7、9、16.11 节。本文档其余部分保持原始设计论证，但在冲突处已标注实际实现。 |

---

# 1. 设计目标

当前引擎不需要 PyTorch 那种完整动态 dispatcher，而是需要一个：

> **由全局 KernelRegistry 持有、kernel 通过 `AM_REGISTER_KERNEL` 静态自注册、在执行计划构建期完成 kernel 选择、在执行期只做函数指针 direct call 的推理专用 dispatch 机制。**

核心目标：

1. 算子语义与具体实现解耦；
2. 支持 CPU reference / optimized kernel；
3. 未来可扩展 CUDA、量化、不同 ISA；
4. Executor 热路径不做字符串查找、不做哈希查找、不做内存分配；
5. 保持与现有 Backend / Runtime / ExecutionPlan 架构一致；
6. Phase 1 保持简单、稳定、可测试。

# 2. 模块定位与所有权

Dispatch 的核心类型位于 `backend` 目录，但 `KernelRegistry` **实际实现为全局 singleton**（而非 per-backend 实例），kernel 通过 `AM_REGISTER_KERNEL` 宏静态自注册到其中。

当前所有权链：

```text
AM_REGISTER_KERNEL 宏（kernel .cpp 文件中展开）
   ↓ 编译期/加载时调用 RegisterGlobal()
KernelRegistry::Global() ← 全局 function-local static singleton
   ↓ Freeze() 后只读
CpuBackend 通过 KernelRegistry::Global() 查询
   ↓
ExecutionPlanBuilder
   ↓ asks Backend to resolve
ResolvedKernel / OpExec
   ↓
Executor direct call
```

关键约束：

* **KernelRegistry::Global()** 是唯一的 kernel 注册中心，不是 per-backend 实例；
* **Kernel** 通过 `AM_REGISTER_KERNEL` 宏自注册，不依赖 Backend 初始化顺序；
* **ExecutionPlanBuilder** 是唯一正式 resolve 发起方；
* **Executor** 只执行已经选好的 kernel；
* **Runtime** 负责资源、线程、内存、上下文，不负责算子选择；
* 全局 singleton 通过 `std::mutex` 保证并发安全（`Register`、`Freeze`、`Resolve` 均加锁）；
* `CpuBackend::CpuBackend()` 构造函数调用 `KernelRegistry::Global().Freeze()`，之后 registry 进入只读态。

## 2.1 明确不采用的方向

当前推理引擎 **不采用** 以下模式：

* PyTorch 式全局 `Dispatcher`
* `DispatchKeySet` / backend bitmask 多重分发
* 运行期字符串查找
* 运行期 hash dispatch
* 运行期 boxed fallback
* 把整个 `RuntimeContext` 宽对象直接下放给 kernel

# 3. 核心抽象

## 3.1 OpType

`OpType` 用于表达算子语义。

```cpp
enum class OpType : uint16_t {
    kEmbedding,
    kLinear,
    kMatMul,
    kRMSNorm,
    kRoPE,
    kAttentionPrefill,
    kAttentionDecode,
    kSiluMul,
    kSoftmax,
    kArgMax,
};
```

Phase 1 不建议引入复杂 `OpSchema`。因为当前项目不是通用动态图框架，不需要完整的 schema 校验系统。

## 3.2 与现有 OperatorName 的关系

当前代码库已经存在 `OperatorName`。为了降低迁移成本，Phase 1 不要求立刻在全仓库彻底移除 `OperatorName`。

推荐做法是：

* **新 dispatch 主线内部以 `OpType` 为核心语义；**
* 在过渡期允许通过一层映射把 `OperatorName` 转成 `OpType`；
* 不再继续扩展基于 `OperatorName + overload_name` 的通用 dispatcher 体系。

例如：

```cpp
StatusOr<OpType> ToOpType(const OperatorName& op_name) noexcept;
```

这样可以逐步迁移，而不是一次性硬切。

# 4. KernelSelector

`KernelSelector` 描述选择 kernel 所需的条件。

```cpp
enum class IsaLevel : uint8_t {
    kScalar,
    kAVX2,
    kAVX512,
    kAMX,
};

enum class ExecPhase : uint8_t {
    kPrefill,
    kDecode,
    kBoth,
};

enum class WeightFormat : uint8_t {
    kPlain,
    kPacked,
    kQuantizedInt8,
    kQuantizedInt4,
};

struct KernelSelector {
    DeviceType device;
    DataType activation_dtype;
    DataType weight_dtype;
    WeightFormat weight_format;
    IsaLevel isa;
    ExecPhase phase;
};
```

## 4.1 为什么不做 DispatchKeySet

这里不要搞成 PyTorch 那种 `DispatchKeySet`，否则：

* 维度会快速膨胀；
* 很容易把 inference engine 变成“通用框架式 dispatcher”；
* 最终又回到运行期多重分发与热路径复杂化。

## 4.2 Phase 1 收敛说明

`KernelSelector` 在未来可以扩展更多维度，但 Phase 1 建议先只保留：

* `device`
* `activation_dtype`
* `weight_dtype`
* `weight_format`
* `isa`
* `phase`

像 `layout` 这样的维度，在当前 tensor/layout 约束尚未完全稳定前，不建议过早放进主选择器主线。

但这并不意味着后续优化 kernel 不能表达内部布局约束。对于 Linear / MatMul 等算子，后续很快会遇到：

```text
Plain weight
Packed weight
Blocked weight
Quantized packed weight
```

因此建议明确：

> **Phase 1 不以 `layout` 作为通用 selector 维度，但允许特定 op 通过 `weight_format` 或 attrs 表达内部布局约束。**

# 5. Kernel 函数签名

Phase 1 建议统一成一个 C 风格函数指针，避免模板和虚函数进入热路径。

但是：

> **不要把整个 `RuntimeContext*` 直接下放给 kernel。**

当前 Backend 设计已经要求 kernel 使用窄执行上下文，因此这里建议使用一个最小 `KernelContext`，只携带单次调用所需的只读能力与窄资源。

```cpp
struct KernelContext {
    Device device{};
    const BackendCapabilities* caps = nullptr;
    WorkspaceBinding workspace{};
    BackendExecutionResources backend_resources{};
};

using KernelFn = Status (*)(KernelContext& ctx,
                            const Tensor* inputs,
                            size_t input_count,
                            Tensor* outputs,
                            size_t output_count,
                            const void* attrs) noexcept;
```

说明：

* `inputs/outputs` 使用数组，避免动态容器；
* `attrs` 指向算子属性，例如 RMSNorm epsilon、RoPE 参数；
* `noexcept` 保持推理热路径稳定；
* 返回 `Status`，不抛异常；
* `KernelContext` 只保留窄资源，不允许把 `RuntimeContext`、registry、builder 等宽对象直接传给 kernel。

另外建议进一步收紧约束：

> **KernelContext 不允许持有 registry、backend、execution plan、model、tokenizer 等宽对象。**

`WorkspaceBinding` 应表示预分配 workspace 的视图，而不是 allocator 本身，这样才能满足 steady-state zero allocation 的目标。

# 6. KernelDescriptor

注册时使用。

```cpp
struct KernelDescriptor {
    OpType op_type;
    KernelSelector selector;
    KernelFn fn;
    const char* name;
    int priority;
};
```

`priority` 用来解决多个 kernel 都匹配时的优先级问题。例如：

```text
AVX512 kernel > AVX2 kernel > Scalar reference kernel
Packed weight kernel > Plain weight kernel
```

但 `priority` 不应承担“万能排序”语义。推荐将规则明确为：

```text
先硬匹配 device / dtype / weight_format / phase
再按 ISA 兼容性过滤
最后才按 priority 选择最优实现
```

也就是说，`priority` 只负责“多个可用实现谁更优”，不负责表达能力匹配。

# 7. KernelRegistry（实际实现）

注册中心只在初始化阶段和计划构建阶段使用，不进入 token 级热路径。

## 7.1 实际实现（v1.1 设计偏离）

实际实现采用 **全局 singleton** + `AM_REGISTER_KERNEL` 静态宏注册，而非原始设计中的 per-backend 实例。

### 全局 singleton 模式

```cpp
class KernelRegistry {
public:
    static KernelRegistry& Global() noexcept;   // function-local static
    static Status RegisterGlobal(const KernelDescriptor& desc) noexcept;

    Status Register(const KernelDescriptor& desc) noexcept;
    Status Resolve(OpType op_type,
                   const KernelSelector& selector,
                   const KernelDescriptor** out) const noexcept;
    void Freeze() noexcept;

private:
    std::vector<KernelDescriptor> kernels_;
    bool frozen_ = false;
    mutable std::mutex mutex_;
};
```

### 静态注册宏

```cpp
// include/aethermind/backend/kernel_registration.h
#define AM_REGISTER_KERNEL(unique_name, ...)                              \
    static bool unique_name = []() noexcept {                              \
        ::aethermind::RegisterKernel(__VA_ARGS__);                         \
        return true;                                                       \
    }()
```

Kernel .cpp 文件中通过该宏自注册：

```cpp
AM_REGISTER_KERNEL(g_rmsnorm_avx2_registration, {
    .op_type = OpType::kRmsNorm,
    .selector = { DeviceType::kCPU, DataType::Float32(), DataType::Float32(),
                  WeightFormat::kPlain, IsaLevel::kAVX2, ExecPhase::kBoth },
    .fn = &CpuRmsNormKernelEntry_FP32_AVX2,
    .name = "cpu::rmsnorm_f32_avx2",
    .priority = 20,
});
```

### 为何偏离原始设计

1. **消除重复注册**：不同 Backend（CPU/CUDA）通过同一设备选择器 `KernelSelector` 区分，per-backend registry 会导致 descriptor 重复注册，且跨 backend resolve 需要额外桥接。
2. **简化 kernel 添加流程**：kernel 作者只需在 .cpp 文件中添加 `AM_REGISTER_KERNEL` 一行，无需修改任何 Backend 代码。
3. **与现有 `AM_REGISTER_OPERATOR` 模式一致**：引擎已有静态注册的先例。
4. **线程安全**：通过 `std::mutex` 保护，`Freeze()` 后 registry 进入只读态，后续 `Resolve` 为 lock-free 读。

### 代价与风险

* 失去了 per-backend registry 的模块隔离（CUDA kernel 不会注册到 CPU 路径；实际通过 `KernelSelector.device` 硬匹配过滤）；
* 依赖静态初始化顺序（function-local static 无顺序问题，但 `AM_REGISTER_KERNEL` 与 `CpuBackend` 构造之间的时序需要保证 `Freeze` 不提前执行）；
* 当前未使用 `__attribute__((used))`，存在被链接器 GC 的风险（与 `AM_REGISTER_OPERATOR` 同一风险等级）。

# 8. 匹配规则

不要一开始做太复杂。推荐规则：

```cpp
bool PhaseMatch(ExecPhase candidate, ExecPhase request) noexcept {
    return candidate == request || candidate == ExecPhase::kBoth;
}

bool Match(const KernelSelector& candidate,
           const KernelSelector& request) noexcept {
    return candidate.device == request.device
        && candidate.activation_dtype == request.activation_dtype
        && candidate.weight_dtype == request.weight_dtype
        && candidate.weight_format == request.weight_format
        && PhaseMatch(candidate.phase, request.phase)
        && candidate.isa <= request.isa;
}
```

其中 `candidate.isa <= request.isa` 表示：

* 当前机器支持 AVX512 时，可以选择 AVX2；
* 当前机器支持 AVX2 时，不能选择 AVX512。

而 `candidate.phase == ExecPhase::kBoth` 时，应能同时匹配 `kPrefill` 和 `kDecode` 请求，否则 `kBoth` 没有实际意义。

最后再按 `priority` 选择最高的实现。

# 9. 注册方式（实际实现）

原始设计推荐由 backend 在构造阶段显式注册。**实际实现完全偏离了该方案**。

## 9.1 实际注册方式：AM_REGISTER_KERNEL 宏

参见 7.1 节。每个 kernel 在其 .cpp 文件中通过 `AM_REGISTER_KERNEL` 宏自注册到 `KernelRegistry::Global()`，无需修改任何 Backend 代码。

### 示例：RMSNorm scalar 注册

```cpp
// src/backend/cpu/kernels/rmsnorm/rmsnorm_fp32_scalar.cpp
AM_REGISTER_KERNEL(g_rmsnorm_scalar_registration, {
    .op_type = OpType::kRmsNorm,
    .selector = {
        .device = DeviceType::kCPU,
        .activation_dtype = DataType::Float32(),
        .weight_dtype = DataType::Float32(),
        .weight_format = WeightFormat::kPlain,
        .isa = IsaLevel::kScalar,
        .phase = ExecPhase::kBoth,
    },
    .fn = &CpuRmsNormKernelEntry_FP32_Scalar,
    .name = "cpu::rmsnorm_f32_scalar",
    .priority = 10,
});
```

### Backend 端变化

`CpuBackend` 不再持有 `KernelRegistry` 成员，也不再调用 `RegisterBuiltinKernels()`：

```cpp
// src/backend/cpu/cpu_backend.cpp
CpuBackend::CpuBackend() {
    // 关键：冻结全局 registry（kernel 此时已被 AM_REGISTER_KERNEL 注册）
    KernelRegistry::Global().Freeze();
}

StatusOr<const KernelDescriptor*> CpuBackend::ResolveKernel(
    OpType op_type, const KernelSelector& selector) const noexcept {
    if (selector.device_type != DeviceType::kCPU) {
        return Status::InvalidArgument("CpuBackend can only resolve CPU kernels");
    }
    return KernelRegistry::Global().Resolve(op_type, selector);
}
```

### 关键差异总结

| 维度 | 原始设计 | 实际实现 |
|------|---------|---------|
| Registry 所有权 | Backend 持有 | 全局 singleton |
| 注册时机 | Backend 初始化 | 静态初始化（加载时） |
| 注册方式 | 显式调用 `Register()` | `AM_REGISTER_KERNEL` 宏 |
| 添加新 kernel 需修改 | Backend 代码 | 仅 kernel 自身 .cpp |
| 线程安全 | 未考虑 | `std::mutex` 保护 |
| 冻结入口 | Backend 内部管理 | `KernelRegistry::Global().Freeze()` |

# 10. ResolvedKernel

Executor 不应该每次执行时去 registry 查找。计划构建期应解析成：

```cpp
struct ResolvedKernel {
    OpType op_type;
    KernelFn fn;
    const void* attrs;
    size_t attrs_size;
    const char* debug_name;
};
```

这里必须明确 `attrs` 的生命周期约束：

```text
attrs 由 ExecutionPlan 或 ModelGraph 持有；
attrs 生命周期必须覆盖整个 ExecutionPlan；
kernel 不拥有 attrs；
kernel 不允许修改 attrs；
执行期不允许分配 attrs。
```

其中 `attrs_size` 主要用于 debug、校验和 plan dump，而不是让 kernel 在执行期做动态内存解释。

然后 `ExecutionPlan` / `OpExec` 中保存的是**已冻结**的结果，而不是执行期再 resolve。

例如：

```cpp
struct ExecutionNode {
    ResolvedKernel kernel;
    Tensor* inputs;
    size_t input_count;
    Tensor* outputs;
    size_t output_count;
};
```

Executor 热路径就是：

```cpp
for (auto& node : plan.nodes()) {
    Status st = node.kernel.fn(ctx,
                               node.inputs,
                               node.input_count,
                               node.outputs,
                               node.output_count,
                               node.kernel.attrs);
    if (!st.ok()) {
        return st;
    }
}
```

这就是最终目标：**dispatch 发生在计划构建期，执行期只有函数指针调用。**

# 11. KernelResolver

`KernelResolver` 是一个可接受的命名，但它的职责必须与当前 Backend 架构对齐。

关键不是名字，而是下面这条约束：

> **ExecutionPlanBuilder 是唯一正式 resolve 发起方。**

因此可以有两种等价落点：

## 11.1 方案 A：显式 KernelResolver

```cpp
class KernelResolver {
public:
    explicit KernelResolver(const Backend& backend) noexcept
        : backend_(backend) {}

    Status ResolveNode(const ModelNode& node,
                       ResolvedKernel* out) const noexcept;

private:
    const Backend& backend_;
};
```

## 11.2 方案 B：ExecutionPlanBuilder 内部私有 resolve 逻辑

```cpp
class ExecutionPlanBuilder {
private:
    Status ResolveKernelForNode(const ModelNode& node,
                                const Backend& backend,
                                ResolvedKernel* out) const noexcept;
};
```

无论采用哪种形式，resolve 都应基于：

* 模型 dtype；
* 权重格式；
* 当前 CPU ISA；
* prefill/decode 阶段；
* backend 能力；

构造 `KernelSelector`，然后通过 backend 内部 registry resolve。

# 12. 与现有旧 dispatch 体系的关系

当前代码库中已经存在：

* `dispatcher.h`
* `dispatch_key.h`
* `dispatch_key_set.h`
* 基于 `OperatorName` 的早期分发设施

这套草案的建议是：

* **不要继续扩展这套旧体系作为未来主线；**
* Phase 1/Phase 2 的新 dispatch 主线应基于 `OpType + KernelSelector + KernelDescriptor + ResolvedKernel`；
* 旧的 `Dispatcher / DispatchKeySet` 可以在迁移期保留，但应视为待冻结/待退场模块，而不是新设计基础。

# 13. 对 Phase 1 的建议收敛

Phase 1 已落地：

```text
OpType
KernelSelector
KernelDescriptor
全局 KernelRegistry singleton（设计偏离，见 7.1 节）
ResolvedKernel
KernelResolver（或等价的 plan-build resolve 逻辑）
```

先支持：

```text
Device: CPU
DType: FP32
WeightFormat: Plain
ISA: Scalar / AVX2 可选
Phase: Prefill / Decode / Both
```

不要做（当前已遵守）：

```text
DispatchKeySet
Autograd key
TLS include/exclude
Boxed fallback
动态 shape generic schema
运行期字符串查找
运行期 hash dispatch
把 Runtime 宽对象传给 kernel
```

# 14. 推荐演进路线

## Phase 1

目标：能跑通 Llama-family dense greedy decode。

```text
CPU FP32 reference kernel
计划构建期 resolve
Executor 直接函数指针调用
```

## Phase 2

加入优化内核：

```text
AVX2 / AVX512
Packed Linear
RMSNorm optimized
RoPE optimized
Attention decode optimized
```

## Phase 3

加入量化：

```text
INT8
INT4
weight-only quantization
per-channel scale
```

## Phase 4

加入多 Backend：

```text
CUDA
Metal
其他加速器
```

# 15. 最终建议

当前推理引擎引入的 dispatch 定位：

> **它不是 PyTorch 式动态多重分发系统，而是一个全局 registry + plan-build-time resolve 的 kernel 选择器。**（注：与原始设计文档的 "backend-owned" 不同，实际采用全局 singleton，见 7.1 节。）

核心类命名：

```text
KernelRegistry（全局 singleton）
KernelSelector
KernelResolver
ResolvedKernel
```

其真正语义：

* registry 为全局 singleton，kernel 通过 `AM_REGISTER_KERNEL` 自注册；
* resolve 发生在 `ExecutionPlanBuilder`；
* executor 只消费冻结后的 kernel 函数指针；
* 不回到全局 dispatcher / DispatchKeySet 路线。

这套设计足够支撑 Phase 1，也不会堵死后续 CPU 优化、量化、多 backend 扩展。

# 16. 执行方案

下面给出这份 dispatch 草案的推荐落地顺序。核心原则是：

> **先立新主线，再迁移 resolve，最后冻结旧 dispatcher。**

## 16.1 总体目标

把当前代码从：

```text
KernelKey + OperatorName + old Dispatcher/DispatchKeySet
```

迁移到：

```text
OpType + KernelSelector + KernelDescriptor + 全局 KernelRegistry
+ AM_REGISTER_KERNEL 静态注册 + plan-build-time resolve + ResolvedKernel
```

并且保证：

* 执行期不做 registry/hash/string lookup；
* 旧 `Dispatcher / DispatchKeySet` 冻结，不再扩展；
* `ExecutionPlanBuilder` 是唯一正式 resolve 发起方；
* `Executor` 只做函数指针 direct call；
* Kernel 通过 `AM_REGISTER_KERNEL` 自注册到 `KernelRegistry::Global()`（注：与原始设计的 "backend-owned" 不同）。

## 16.2 Phase A：冻结边界

目标：先定规则，不急着大改代码。

### 主要工作

1. 冻结本文档作为 dispatch 主线基线；
2. 明确以下约束为唯一主线：
   - ~~Backend owns kernel registry~~ **（设计偏离：实际为全局 singleton + AM_REGISTER_KERNEL，见 7.1 节）**
   - `ExecutionPlanBuilder` 是唯一 resolve 发起方
   - `Executor` 只消费 `ResolvedKernel`
   - 旧 `Dispatcher / DispatchKeySet` 不再扩展
3. 明确过渡策略：短期保留 `OperatorName`、`KernelKey`、`dispatcher_bridge`，但仅作为迁移辅助，不再作为未来主线。

### 完成标准

* 设计文档全部对齐；
* 团队对“旧 dispatch 冻结、新 dispatch 主线”无歧义。

## 16.3 Phase B：落最小新类型

目标：先把新 dispatch 抽象站起来，但不立即改执行链路。

### 主要工作

新增核心类型：

* `OpType`
* `KernelSelector`
* `KernelDescriptor`
* `ResolvedKernel`

短期保留并兼容：

* `kernel_key.h`
* `dispatcher_bridge.h`
* `operator_name.h`

同时增加一个过渡入口，例如：

```cpp
StatusOr<OpType> ToOpType(const OperatorName& op_name) noexcept;
```

### 完成标准

* 新类型定义完成；
* 不破坏现有编译；
* 旧代码仍可工作。

## 16.4 Phase C：重构 KernelRegistry

目标：把 registry 从“按 key 查表”演进为“按 selector resolve”。

### 主要工作

将接口从当前的：

```cpp
Register(KernelKey, KernelFn)
Find(KernelKey)
```

逐步演进到：

```cpp
Register(const KernelDescriptor&)
Resolve(OpType, KernelSelector, const KernelDescriptor**)
```

实现最小匹配规则：

* `device` 硬匹配
* `activation_dtype` 硬匹配
* `weight_dtype` 硬匹配
* `weight_format` 硬匹配
* `PhaseMatch(candidate, request)`
* `candidate.isa <= request.isa`
* 最后按 `priority` 选择最优实现

过渡期可以保留旧接口，内部桥接到新结构，避免一次性重写全部测试。

**设计偏离说明**：实际实现中 `KernelRegistry` 为全局 singleton（非 backend-owned），kernel 通过 `AM_REGISTER_KERNEL` 宏自注册（见 7.1 节）。匹配规则和 resolve 逻辑与原始设计一致。

### 完成标准

* ~~backend-owned~~ `KernelRegistry` 支持 descriptor 注册（实际为全局 singleton，见 7.1 节）；
* selector-based resolve 可用；
* `kBoth` 匹配正确；
* `priority` 仅用于“可用实现中的最优选择”。

## 16.5 Phase D：收敛 CpuBackend（实际实现）

目标：让 `CpuBackend` 正确接入全局 kernel resolve 路径。

### 主要工作（已落地，与原始设计不同）

实际实现中 `CpuBackend` **不持有** `KernelRegistry` 成员——kernel 已通过 `AM_REGISTER_KERNEL` 注册到 `KernelRegistry::Global()`。`CpuBackend` 的变化：

1. 移除了 `RegisterBuiltinKernels()` 方法；
2. 移除了 `KernelRegistry kernel_registry_` 成员；
3. 构造函数调用 `KernelRegistry::Global().Freeze()` 冻结全局 registry；
4. `ResolveKernel(...)` 和 `ResolveKernelInfo(...)` 先检查 `device_type == kCPU`，再委托给 `KernelRegistry::Global().Resolve(...)`。

```cpp
// 简化示意
CpuBackend::CpuBackend() {
    KernelRegistry::Global().Freeze();
}

StatusOr<const KernelDescriptor*> CpuBackend::ResolveKernel(
    OpType op_type, const KernelSelector& selector) const noexcept {
    if (selector.device_type != DeviceType::kCPU) {
        return Error("CpuBackend: not a CPU selector");
    }
    return KernelRegistry::Global().Resolve(op_type, selector);
}
```

### 完成标准（已满足）

* CPU kernel 通过 `AM_REGISTER_KERNEL` 自注册到全局 registry；
* `CpuBackend` 通过 `KernelRegistry::Global()` 查询，不影响 `KernelSelector.device` 硬匹配过滤；
* 不再新增旧 `KernelKey` 主线逻辑。

### 与原始设计的关键差异

| 维度 | 原始设计 | 实际实现 |
|------|---------|---------|
| Registry 所有权 | `CpuBackend` 持有 | `KernelRegistry::Global()` |
| 注册方式 | `RegisterBuiltinKernels()` 显式注册 | `AM_REGISTER_KERNEL` 静态宏 |
| 冻结时机 | `CpuBackend` 内部 | `KernelRegistry::Global().Freeze()` |
| Resolve 入口 | `backed_registry_.Resolve(...)` | `KernelRegistry::Global().Resolve(...)` + CPU guard |

## 16.6 Phase E：把 resolve 拉进计划构建期

目标：真正实现“plan-build-time resolve”。

### 主要工作

1. 引入 `ResolvedKernel`；
2. 在 `ExecutionPlanBuilder` 中实现 `ResolveKernelForNode(...)`，或提供等价的 `KernelResolver`；
3. 明确 attrs 生命周期：
   - attrs 由 plan/model graph 持有
   - 生命周期覆盖整个 execution plan
   - 执行期不分配 attrs

### 完成标准

* 计划构建期可为节点生成 `ResolvedKernel`；
* 执行节点不再依赖 runtime lookup；
* attrs 生命周期规则落地。

## 16.7 Phase F：接 Executor direct call

目标：执行期只做函数指针调用。

### 主要工作

1. `ExecutionNode` / `OpExec` 接入 `ResolvedKernel`；
2. `Executor` 改为 direct call；
3. 明确禁止热路径中的：
   - registry lookup
   - dispatcher lookup
   - hash dispatch
   - string dispatch

### 完成标准

* executor 热路径 direct call；
* 无 runtime resolve；
* 无 hot-path registry access。

## 16.8 Phase G：冻结旧体系

目标：把旧 dispatch 路线明确降级。

### 主要工作

冻结以下模块：

* `include/dispatcher.h`
* `src/dispatcher.cpp`
* `include/dispatch_key.h`
* `include/dispatch_key_set.h`

要求：

* 不再新增新算子接入到旧 dispatcher；
* 不再继续扩 `DispatchKeySet`；
* 不把新 kernel resolve 逻辑塞回旧路径。

短期先冻结，后面视迁移完成度再决定是否物理删除。

### 完成标准

* 旧模块不再增长；
* 新功能全部走新主线。

## 16.9 推荐批次

### Batch 1

```text
Phase A + Phase B
文档冻结 + 新类型落地
```

### Batch 2

```text
Phase C + Phase D
新 registry + CpuBackend 注册/解析
```

### Batch 3

```text
Phase E
ResolvedKernel + ExecutionPlanBuilder resolve
```

### Batch 4

```text
Phase F + Phase G
Executor direct call + 旧体系冻结
```

## 16.10 测试策略

### Phase B / C

建议新增：

* `test_kernel_selector.cpp`
* `test_kernel_registry_resolve.cpp`

覆盖：

* 硬匹配
* `kBoth` 匹配
* ISA 兼容
* `priority` 选择

### Phase D

建议扩展：

* `test_cpu_resolve_kernel.cpp`

覆盖：

* builtin kernel 注册
* CPU selector resolve
* capability-aware 路径

### Phase E / F

建议新增：

* `test_execution_plan_builder.cpp`
* `test_executor_backend_path.cpp`

覆盖：

* plan-build resolve
* `ResolvedKernel` 冻结
* executor direct call
* no hot-path lookup

### Phase G

建议新增约束测试：

* `test_no_hotpath_resolve.cpp`

## 16.11 风险控制

### 不要做的事（已遵守）

* 不要一次性删除 `OperatorName` — `ToOpType` 仍保留过渡映射
* 不要先删旧 dispatcher 再补新主线 — Batch 1-3 先立新主线，Batch 4 再冻结删除
* ~~不要把全局 singleton registry 引回来~~ **（已偏离：实际采用了全局 singleton，详见 7.1 节；但通过 device 硬匹配保证隔离，未出现预期风险）**
* 不要把 `RuntimeContext*` 直接传给 kernel — `KernelFunc` 保持窄签名
* 不要让 executor 在迁移期继续做 hash/string lookup — Executor 只消费冻结 kernel

### 要优先守住的事

* ~~backend-owned ownership~~ **（已偏离：实际为全局 singleton；通过 `KernelSelector.device` 硬匹配保证跨 backend 隔离）**
* narrow `KernelContext`
* attrs 生命周期明确
* resolve 只发生在 plan-build time

## 16.12 一句话落地策略

这份 dispatch 草案的执行方案可以概括为：

> **先立新主线，再迁移 resolve，最后冻结旧 dispatcher。**

## 16.13 配套任务清单

按文件拆解的 implementation task list 见：

- `docs/designs/dispatch_implementation_task_list.md`
