# 基础库模块（base / dtypes / memory）

本文是 `include/aethermind/base/`、`include/aethermind/dtypes/`、`include/aethermind/memory/` 及其 `src/` 实现的模块设计权威。系统分层见 [架构文档](../architecture.md)，产品边界见 [产品文档](../product.md)。

## 1. 职责与边界

| 子域 | 负责 | 当前主要入口 |
|---|---|---|
| `base` | 统一错误模型；张量元数据与借用视图；设备与对齐值类型；断言与异常工具；侵入式引用计数；workspace 纯数据契约；只读 mmap | `Status` / `StatusOr<T>`、`Tensor`、`ShapeAndStride`、`TensorView` / `MutableTensorView`、`Device`、`KernelSelector`、`WorkspaceArena` |
| `dtypes` | 元素类型元数据、宿主标量表示、reduced-precision 数值类型与转换；不做张量存储或算子计算 | `DataType` / `DLDataType`、`Scalar`、`Half`、`BFloat16`、`Float8_e4m3fn`、`Float8_e5m2`、`complex<T>`、`cast` / `check_and_cast` |
| `memory` | 原始存储的申请、释放合同与生命周期抽象；统一分配器注册中心；CPU 存储策略 | `Allocator`、`AllocatorProvider`、`AllocatorRegistry`、`MemoryHandle`、`Buffer`、`CPUAllocator` |

**不得负责什么**：

- 不负责模型加载、图构建、图优化、lowering、执行计划、kernel resolve 或算子调度；这些属于 model / graph / compiler / execution / backend。
- 不负责 KV cache 物理布局、reservation、commit watermark；`kv_cache` 只依赖本模块的 `Allocator`、`Buffer`、`TensorView` 等基础契约。
- 不负责 workspace 规划算法、activation liveness 复用或 workspace budget；`base` 只提供 `WorkspaceRequirement` / `WorkspaceBinding` / `WorkspaceArena` 契约，规划在 `runtime`。
- 不负责权重 packing、layout 选择、量化算法或 kernel-friendly repack；`memory` 只分配原始字节，`dtypes` 只描述元素类型。
- 不提供自动微分、通用 eager 张量框架、字符串/Tokenizer、HTTP 服务或请求调度。
- 不提供全局分配器单例。`AllocatorRegistry` 必须由 `Runtime` 等显式 owner 持有，`Buffer` 不持有 `Allocator` 指针或引用计数。
- 当前仓库的 `include/`、`src/`、`CMakeLists.txt` 中不存在 `ammalloc`、`ThreadCache`、`CentralCache`、`PageCache`、`SizeClass` 的定义或引用；不要把这些类型当作已实现能力。

## 2. 关键数据结构与所有权

### 2.1 错误与结果

| 类型 | 表示与不变量 | 所有权 / 生命周期 | 线程安全 |
|---|---|---|---|
| `StatusCode` | `enum class StatusCode : uint8_t`，`kOk = 0`，共 18 个值；数值与 `am_status_code` 逐项 `static_assert` 对齐 | 值类型 | 只读安全 |
| `Status` | `StatusCode code_` + `std::string message_`；不变量：`code_ == kOk` 时 `message_.empty()`，反向不成立；非 OK 构造私有，只能经命名工厂创建 | 值类型；消息随对象复制/移动 | `const` 方法可并发读；并发修改需外部同步 |
| `StatusOr<T>` | `std::variant<T, Status> storage_`；`T` 不得为引用或 `Status`，且必须 nothrow move-constructible；不变量：永不持有 `Status::Ok()`；无默认构造 | 值类型；成功值或错误 Status 二选一；`status() const&` 在成功时返回函数内静态 OK（每个 `StatusOr<T>` 实例化各一份） | 不支持并发修改；`const` 读取可并发 |

`Status` 的访问器：`ok()`、`explicit operator bool()`、`code()`、`message() &` / `message() const&`、`WithMessage(std::string)`、`ToString()`、`operator==(const Status&)`、`operator==(StatusCode)`。`message() &&` 与 `message() const&&` 被删除，防止返回指向临时 `Status` 内部的悬垂引用；`WithMessage()` 返回新对象且只替换消息（不追加），对 OK 调用时仍返回无消息 OK。错误工厂为 `Status::{Cancelled, Unknown, DeadlineExceeded, Aborted, Unavailable, DataLoss, Unauthenticated, InvalidArgument, NotFound, Internal, PermissionDenied, AlreadyExists, OutOfRange, FailedPrecondition, Unimplemented, ResourceExhausted, Overflow}(std::string_view)`，成功入口为 `Status::Ok()` / 默认构造。

`StatusOr<T>` 的访问器：`ok()`、`get_if_ok()`（左值返回指针，右值重载删除）、`status() const&` / `status() &&`、`value()`（失败抛 `std::logic_error`）、`operator*` / `operator->`（失败由 `std::get` 抛 `std::bad_variant_access`）、`value_or()`、`operator==(StatusCode)`。`StatusOr<T>` 没有 `operator bool`，调用方必须显式检查 `ok()`。

### 2.2 张量与视图

| 类型 | 表示与不变量 | 所有权 / 生命周期 |
|---|---|---|
| `ShapeAndStride` | `int32_t size_`、`bool initialized_`、`std::array<int64_t, kMaxRank>` 的 shape 与 stride；`kMaxRank == 8`，全部 inline，零堆分配；shape/stride 同时更新；stride 以元素为单位 | 按值拥有；`Tensor` 内部保存一份；默认构造未初始化（`numel()==0`、`is_contiguous()==false`），显式空 shape 为 rank-0（`numel()==1`、`is_contiguous()==true`） |
| `Tensor` | `Buffer buffer_` + `size_t byte_offset_` + `DataType dtype_` + `ShapeAndStride shape_and_strides_`；构造时校验 buffer 已初始化、`byte_offset_ <= buffer.nbytes()`、shape/stride 非负、dtype itemsize 为正、存储范围有效、拒绝 scalable vector dtype | 拥有 `Buffer` 强引用；拷贝共享同一 `Buffer`，移动转移该强引用，均不复制存储；元数据按值复制；`alignment()` 由 `Buffer` 对齐与 `byte_offset_` 的 `std::gcd` 得出 |
| `TensorView` | `const void* data_` + `DataType dtype_` + `IntArrayView shape_` + `IntArrayView strides_` + `size_t alignment_`；`alignment_ == 0` 表示未知；`rank()` 由 `shape().size()` 推导 | 严格借用：不拥有 data，也不拥有 shape/stride 元数据；调用方必须保证两者长于 view；设计用于栈上热路径，不应长期存储 |
| `MutableTensorView` | 与 `TensorView` 相同，但 `data_` 为 `void*`，写入直通底层存储 | 同上；可变性只由类型区分 |

`TensorView(const Tensor&)` / `MutableTensorView(Tensor&)` 是显式构造，不提供从 `Tensor` 的隐式转换；视图构造后调用方仍需保证 `Tensor` 与其元数据存活。

`TensorView` / `MutableTensorView` 的有效性由 `is_valid()` 判定，不存在 `defined()`。有效条件为：dtype/shape/strides 元数据一致（rank 相同、不超过 `ShapeAndStride::kMaxRank`、非负、无溢出、拒绝 scalable vector 与零字节 dtype）；`alignment_` 为 0 或 2 的幂且非空指针满足该对齐；只要 shape 含至少一个逻辑元素，data 必须非空。rank-0（shape `[]`）含 1 个元素并要求非空 data；`[0]` 等零元素 shape 允许 null data。

`TensorView` 没有 `slice()` / `narrow()`；切片由 `Tensor::slice()` 返回共享同一 `Buffer` 的新 `Tensor`，再由调用方派生视图。

### 2.3 元素类型与标量

| 类型 | 表示与不变量 | 所有权 |
|---|---|---|
| `DLDataType` | DLPack 风格 `{code, bits, lanes}`；`bits` 为单 lane 位宽；`lanes` 按 `int16_t` 解释，`> 1` 为定长向量，`< -1` 为 scalable vector 的 vscale 因子，`-1` 保留 | 纯值 |
| `DataType` | 包装 `DLDataType` 并提供校验；`code/bits` 配对受约束（BFloat 必须 16 位，Float8 必须 8 位等）；`lanes == 0` 只允许 `Void` / `Undefined`；`nbytes()` 返回整个元素的字节数（含 lanes，向上取整）；`Undefined` 彼此相等；`hash` 与 `operator==` 保持一致 | 纯值；`Make<T>()` 给出 C++ 标量映射 |
| `Scalar` | 宿主侧 0 维值，`union val { int64_t, uint64_t, double, complex<double> }` + `DataType dtype_`；不是 tensor 存储，不拥有 `Buffer` | 值类型；`Tensor::FromScalar` 仅支持 CPU 并显式传 allocator；`Tensor::item` / `set_item` 要求 CPU、`numel() == 1` |

`dtypes::cast` 提供 `is_cast_overflow`、`cast`、`check_and_cast`。`check_and_cast` 先检查溢出，溢出时抛 `RuntimeError`；`cast` 只执行转换，不做检查；`strict_unsigned` 控制负数到无符号目标是否视为溢出。`Half`、`BFloat16`、`Float8_e4m3fn`、`Float8_e5m2` 均以 `from_bits()` 保留原始位模式；`Half` / `BFloat16` 转换使用固定 round-to-nearest-even、gradual underflow，并对 NaN 做 quiet 化（保留符号与可表示 payload）。`Float8_e4m3fn` 是 finite-and-NaN-only，溢出映射为 NaN；`Float8_e5m2` 保留指数并支持 overflow 到 infinity 的格式语义。

### 2.4 原始存储、分配器与共享契约

| 类型 | 表示与不变量 | 所有权 / 生命周期 |
|---|---|---|
| `MemoryHandle` | move-only；`void* data_`、`void* ctx_`、`memory_deleter_fn deleter_`、`size_t alignment_`、`Device device_`；空 handle 为 null data / null deleter / `kUndefined` / 0 对齐；`reset()` 先清空自身再调用 deleter | 非空 deleter 表示拥有，deleter 至多调用一次；null 或 `NoOpMemoryDeleter` 表示借用；deleter 不得借用可能在 Buffer 之前销毁的 allocator 或其他资源 |
| `BufferImpl` | 不可变；保存 `nbytes_` 与 `MemoryHandle`；校验非零字节需非空指针、device 为 CPU/CUDA/CANN、alignment 为 0 或 2 的幂、CPU 指针满足声明对齐 | 继承 `Object`，由侵入式引用计数管理；不可复制/移动 |
| `Buffer` | 持有 `ObjectPtr<BufferImpl>`；所有拷贝共享同一 impl 与同一块内存；`nbytes()==0` 仍可为 initialized；空/移动后 Buffer 的 observer 返回空值 | 最后一个强引用销毁时调用 `MemoryHandle` deleter；Buffer 不持有 allocator；不同 Buffer 副本可并发复制/赋值/销毁，同一对象或共享内存的并发写需外部同步 |
| `Allocator` | 抽象基类，不可复制；`TryAllocate(const AllocationRequest&) noexcept` 校验请求与 hook 返回值，成功返回 `StatusOr<Buffer>`；`Allocate(size_t)` 是抛异常便捷入口；`AllocateStorage` 由设备实现 | 通常由 `AllocatorRegistry` / `Runtime` 拥有，也可由调用方直接持有；`Buffer` 可长于 allocator |
| `AllocatorProvider` / `AllocatorRegistry` | Provider 按 `DeviceType` 创建 `Allocator`；Registry 按 `Device`（type + index）缓存实例；`RegisterProvider` 重复注册 abort，`SetProvider` 替换并丢弃该类型的缓存实例；`GetAllocator` 缺 provider abort，`TryGetAllocator` 返回 `FailedPrecondition` | Registry 按值拥有 providers 与 instances；`Runtime` 按值拥有 Registry；Registry 非线程安全 |
| `Object` / `ObjectPtr<T>` / `WeakObjectPtr<T>` | 侵入式强/弱引用计数；空 handle 用 `EmptyObjectSentinel<T>` 表示，不存裸 `nullptr`；`ObjectPtr` 拥有强引用，`WeakObjectPtr::lock()` 仅在强计数非零时提升 | `BufferImpl` 继承 `Object`；`ObjectPtr` 的最后强引用触发 deleter；弱引用只延长控制块 |
| `WorkspaceRequirement` / `WorkspaceBinding` / `WorkspaceArena` | Requirement 含 `bytes`、`alignment`（默认 64，非零 2 的幂）、`WorkspaceLifetime`、`reusable`、规划输出 `offset`；Binding 借用一个已对齐 slice；Arena 只有 `Bind()` / `Reset()` | Binding 从 Arena 借用，只在 step 执行期间有效；不得存放到 kernel params 或跨 `Execute()` |
| `MemoryMappedFile` | 只读 `mmap(2)` 的 move-only RAII 包装；`Map()` 返回 `StatusOr`；`Advise()` 只是访问模式提示 | 析构或移动赋值释放映射；非线程安全 |

## 3. 接口与代码位置

| 能力 | 公共接口 | 代码位置 |
|---|---|---|
| 错误码、`Status`、`StatusOr<T>`、传播宏 | `StatusCode`、`StatusCodeName`、`ToAMStatusCode`、`FromAMStatusCode`、`Status`、`StatusOr<T>`、`AM_RETURN_IF_ERROR`、`AM_RETURN_IF_ERROR_WITH_MSG`、`AM_ASSIGN_OR_RETURN` | [include/aethermind/base/status.h](../../include/aethermind/base/status.h)、[src/base/status.cpp](../../src/base/status.cpp) |
| 张量元数据 | `ShapeAndStride::set` / `set_contiguous` / `numel` / `is_contiguous` / `max_element_offset` | [include/aethermind/base/shape_and_stride.h](../../include/aethermind/base/shape_and_stride.h)、[src/base/shape_and_stride.cpp](../../src/base/shape_and_stride.cpp) |
| 拥有型张量 | `Tensor` 构造、`data` / `mutable_data`、`view` / `mutable_view`、`slice` / `narrow`、`storage_range_is_valid`、`FromScalar` / `item` / `set_item` | [include/aethermind/base/tensor.h](../../include/aethermind/base/tensor.h)、[src/base/tensor.cpp](../../src/base/tensor.cpp) |
| 借用视图 | `TensorView` / `MutableTensorView` 构造、`is_valid`、`data` / `dtype` / `shape` / `strides` / `alignment` / `rank` / `numel` / `itemsize` / `logical_nbytes` / `is_contiguous` / `is_rank_zero` | [include/aethermind/base/tensor_view.h](../../include/aethermind/base/tensor_view.h)、[src/base/tensor_view.cpp](../../src/base/tensor_view.cpp) |
| 元素类型 | `DLDataType`、`DataType`、`DataType::Make<T>`、`Match<T>`、`Is*` 判断、`ToString`、`std::hash<DataType>` | [include/aethermind/dtypes/data_type.h](../../include/aethermind/dtypes/data_type.h)、[src/dtypes/data_type.cpp](../../src/dtypes/data_type.cpp) |
| 标量与转换 | `Scalar`、`is_cast_overflow`、`cast`、`check_and_cast`、`Half`、`BFloat16`、`Float8_e4m3fn`、`Float8_e5m2`、`complex<T>` | [include/aethermind/base/scalar.h](../../include/aethermind/base/scalar.h)、[include/aethermind/dtypes/cast.h](../../include/aethermind/dtypes/cast.h)、[include/aethermind/dtypes/half.h](../../include/aethermind/dtypes/half.h)、[include/aethermind/dtypes/bfloat16.h](../../include/aethermind/dtypes/bfloat16.h) |
| 批量转换 | `ConvertFp16ToFp32` / `ConvertFp32ToFp16`、`ConvertBf16ToFp32` / `ConvertFp32ToBf16`、`Fp16ConversionFuncs`、`Bf16ConversionFuncs` | [include/aethermind/dtypes/fp16_conversion.h](../../include/aethermind/dtypes/fp16_conversion.h)、[include/aethermind/dtypes/bf16_conversion.h](../../include/aethermind/dtypes/bf16_conversion.h)、[src/dtypes/fp16_conversion.cpp](../../src/dtypes/fp16_conversion.cpp)、[src/dtypes/bf16_conversion.cpp](../../src/dtypes/bf16_conversion.cpp) |
| 设备、对齐、断言 | `DeviceType`、`Device`、`IsValidAlignment`、`AlignOffsetUp`、`AM_CHECK`、`AM_DCHECK` | [include/aethermind/base/device.h](../../include/aethermind/base/device.h)、[include/aethermind/base/alignment.h](../../include/aethermind/base/alignment.h)、[include/aethermind/base/check.h](../../include/aethermind/base/check.h) |
| 引用计数对象 | `Object`、`ObjectPtr<T>`、`WeakObjectPtr<T>`、`make_object`、`make_array_object` | [include/aethermind/base/object.h](../../include/aethermind/base/object.h)、[include/aethermind/base/object_allocator.h](../../include/aethermind/base/object_allocator.h)、[src/base/object.cpp](../../src/base/object.cpp) |
| 原始内存合同 | `AllocationRequest`、`Allocator`、`AllocatorProvider`、`AllocatorRegistry`、`MemoryHandle`、`Buffer` | [include/aethermind/memory/allocator.h](../../include/aethermind/memory/allocator.h)、[include/aethermind/memory/memory_handle.h](../../include/aethermind/memory/memory_handle.h)、[include/aethermind/memory/buffer.h](../../include/aethermind/memory/buffer.h)、[src/memory/allocator.cpp](../../src/memory/allocator.cpp) |
| CPU 分配 | `CPUAllocator`、`CPUAllocatorPolicy`、`CPUAllocatorProvider` | [include/aethermind/memory/cpu_allocator.h](../../include/aethermind/memory/cpu_allocator.h)、[src/memory/cpu_allocator.cpp](../../src/memory/cpu_allocator.cpp) |
| workspace 契约 | `WorkspaceLifetime`、`WorkspaceRequirement`、`WorkspaceBinding`、`WorkspaceArena` | [include/aethermind/base/workspace_types.h](../../include/aethermind/base/workspace_types.h)、[include/aethermind/base/workspace_arena.h](../../include/aethermind/base/workspace_arena.h) |
| 跨层纯数据契约 | `KernelSelector`、`SelectorMatches`、`ExecPhase`、`WeightFormat`、`KVCacheLayerStorageBinding`、`KVCacheAppendBinding`、`KVCacheReadBinding` | [include/aethermind/base/kernel_selector.h](../../include/aethermind/base/kernel_selector.h)、[include/aethermind/base/kernel_attrs.h](../../include/aethermind/base/kernel_attrs.h)、[include/aethermind/base/kv_cache_binding.h](../../include/aethermind/base/kv_cache_binding.h) |
| 只读文件映射 | `MemoryMappedFile::Map`、`Advise`、`Bytes` | [include/aethermind/base/mmap_file.h](../../include/aethermind/base/mmap_file.h)、[src/base/mmap_file.cpp](../../src/base/mmap_file.cpp) |

分配器实现类型名为 `Allocator` / `AllocatorProvider` / `AllocatorRegistry`。仓库中不存在 `IAllocator`、`Storage`、`DataPtr`、`AllocatorTable` 或兼容 alias；不要把旧设计稿中的这些名字当作当前接口。

错误语义要点：

- `Status` / `StatusOr<T>` 带类级 `AM_NODISCARD`；忽略返回值是警告，不是语言级强制。
- `Status::ToString()` 在成功时返回 `"OK"`，错误且消息为空时返回错误码名，否则返回 `"CODE: message"`；流输出复用同一文本。
- `AM_RETURN_IF_ERROR(expr)` 对 `expr` 只求值一次，接受 `Status` 或 `StatusOr<T>`，失败时提取 `Status` 并从当前函数返回；当前函数返回类型必须可由 `Status` 构造。
- `AM_RETURN_IF_ERROR_WITH_MSG(expr, msg)` 只接受 `Status`；保留错误码，把消息替换为 `"<msg>: <原始消息>"`；原始消息为空时仍会生成带 `": "` 的字符串。
- `AM_ASSIGN_OR_RETURN(lhs, expr)` 要求 `expr` 返回 `StatusOr<T>`，成功时把值移动到 `lhs`；它没有 `do { } while (false)` 包装，不能直接作为无大括号 `if/else` 的单条语句。
- `StatusOr<T>` 用 OK `Status` 构造会抛 `std::invalid_argument`；访问错误值会抛 `std::logic_error` 或 `std::bad_variant_access`。这些是 API 误用异常，不是业务错误传播通道，不得跨越 C ABI。
- `AM_CHECK` 在失败时向 stderr 写 `Check failed: ...` 并 `std::abort()`；`AM_DCHECK` 只在非 `NDEBUG` 构建生效。`AM_CHECK` 用于不变量/前置条件，不用于可恢复业务错误。

## 4. 主流程 / 时序

### 4.1 错误传播

1. 底层函数返回 `Status` 或 `StatusOr<T>`；成功路径不抛异常。
2. 调用方用 `AM_RETURN_IF_ERROR` 逐层返回，或用 `AM_ASSIGN_OR_RETURN` 解包成功值；需要补充上下文时用 `AM_RETURN_IF_ERROR_WITH_MSG`。
3. `StatusCode` 负责稳定的机器分类，消息负责携带节点、路径、shape 等可读上下文；调用方不应依赖消息文本做分类。
4. C ABI 边界通过 `ToAMStatusCode` / `FromAMStatusCode` 映射；`FromAMStatusCode` 对越界值返回 `StatusCode::kUnknown`，不做非法枚举转换。
5. 违反 API 不变量时使用 `AM_CHECK` 终止，或由 `StatusOr` 的访问器抛异常；当前实现尚未把 `value()`、解引用与 `AM_CHECK` 的误用策略完全统一。

### 4.2 分配与释放

```text
RuntimeBuilder::TryBuild()
  -> BuildAllocatorRegistry()                 // 默认注册 CPUAllocatorProvider
  -> Runtime 按值拥有 AllocatorRegistry
  -> Runtime::GetAllocator(device) / TryGetAllocator(device)
  -> AllocatorRegistry 按 Device 缓存或经 Provider 创建 Allocator
  -> Allocator::TryAllocate({nbytes, alignment})
  -> 校验 alignment；调用 AllocateStorage(request)
  -> CPUAllocator: posix_memalign -> MemoryHandle(data, ctx, deleter, device, alignment)
  -> Allocator 校验 handle 非空、拥有、device 匹配、alignment 足够、CPU 指针地址对齐
  -> Buffer(nbytes, std::move(handle)) -> BufferImpl
  -> 最后一个 Buffer 强引用销毁时执行 MemoryHandle deleter
```

`TryAllocate` 的失败语义：非法 alignment 返回 `InvalidArgument`；hook 返回借用/空/错误 device/不足对齐返回 `Internal`；存储 OOM 或 `BufferImpl` 元数据分配 OOM 返回 `ResourceExhausted`（空消息）；hook 自身失败按其 `Status` 返回。`Allocate` 把 `ResourceExhausted` 转成 `std::bad_alloc`，其他失败用 `AM_THROW(RuntimeError)` 抛出。

CPU 分配细节：`nbytes == 0` 仍分配 1 字节，使 `Buffer` 有可释放的非空指针；请求 alignment 为 0 时使用 `CPUAllocatorPolicy::default_alignment`（构建默认 64，`KALLOC_ALIGNMENT` 可覆盖）；小于 `sizeof(void*)` 的非零 alignment 会被提升而不是拒绝；非 2 的幂在分配前被拒绝。默认 `junk_fill = true`，用固定 pattern 覆写新存储；`allow_thp = true` 时，达到 2 MiB 且 `THP_MEM_ALLOC_ENABLE` 环境门开启的分配会提升页对齐并 `madvise(MADV_HUGEPAGE)`，`madvise` 失败只告警。`CUDAAllocatorProvider` / `CANNAllocatorProvider` 在当前构建中是 abort stub。

### 4.3 张量创建与视图派生

1. `Tensor` 由 `Buffer`、`byte_offset`、`DataType`、`ShapeAndStride` 构造；构造时验证 buffer、offset、dtype 与存储范围。
2. `Tensor` 的拷贝共享同一 `Buffer`（同一块内存），但复制 `ShapeAndStride` 元数据；`Tensor::slice()` 调整 offset/stride 后返回新的 `Tensor`，仍共享 `Buffer`。
3. `Tensor::view()` / `mutable_view()` 构造 `TensorView` / `MutableTensorView`，只借用 `Tensor` 的 data 指针与内部 shape/stride 数组；`Tensor` 必须比 view 活得久，且不得在 view 使用期间修改其元数据。
4. `TensorView` 构造时执行 `is_valid()` 校验；`numel()`、`logical_nbytes()`、`is_contiguous()` 在 debug 构建要求 `is_valid()`。`logical_nbytes() == numel() * itemsize()`，不等于 strided/padded view 的实际可寻址跨度。
5. 从 raw parts 构造视图时，shape/stride 数组由调用方持有；不要把临时 `std::vector` 或临时 `IntArrayView` 留给 view 长期借用。

### 4.4 类型元数据与转换

1. `DataType` 由 `DLDataType` 或 `DataType::Int/Float/...` 工厂构造，构造即校验 code/bits/lanes。
2. `DataType::Make<T>()` / `Match<T>()` 负责 C++ 标量映射；`nbytes()` 给出完整元素的字节数。
3. 标量转换先经 `is_cast_overflow` 判断，再经 `cast` / `check_and_cast` 执行；溢出抛 `RuntimeError`。
4. 批量 FP16/BF16 转换先校验源/目标 span 长度、不重叠与函数指针，再写出；成功不分配存储，空 span 合法，失败在写入前返回。

## 5. 关键取舍

| 选择 | 原因 | 被否方案 |
|---|---|---|
| 业务失败返回 `Status` / `StatusOr<T>` | 可跨层显式传播、可携带消息、可映射 C ABI；热路径不依赖栈展开 | 业务异常（不能跨 C ABI，`noexcept` 难统一）；原始错误码（缺消息）；`bool` + 输出参数（易半初始化）；`std::optional<T>`（无法区分原因） |
| `StatusOr<T>` 用 `std::variant<T, Status>` | C++20 可用，支持 move-only 值，保持显式成功/错误状态 | `std::expected<T, E>`（C++23，当前不可用）；手写 union；`std::optional` |
| `StatusOr<T>` 不提供 `operator bool` | 避免把“有值”隐式当作布尔条件，强制调用方检查 `ok()` 并显式取 `status()` | 隐式 bool 转换 |
| `Status` 只保存扁平消息 | 实现简单、可复制、C ABI 映射稳定；`StatusCode` 承担稳定分类 | 结构化 cause chain、`std::source_location`、错误 payload（当前未实现） |
| 两个独立类型 `TensorView` / `MutableTensorView` | 接口清晰，避免模板特化与 const 分支；可变性由类型系统区分 | 单一模板 + const 参数；继承层次 |
| `TensorView` 只借用 shape/stride，不内嵌 `ShapeAndStride` | view 保持小对象、零额外元数据复制；元数据 owner 由 `Tensor` 或调用方保证 | 在 view 内部复制/拥有 `ShapeAndStride`（会增加每个 view 的固定开销；与当前头文件实际字段不符） |
| `Tensor` 拥有 `Buffer`，view 不拥有存储 | 持久态与热路径借用态分离；`Buffer` 引用计数负责释放 | 让单一类型同时承担 owner/view 语义（会迫使热路径复制元数据或持有额外引用） |
| `ShapeAndStride` 固定 `kMaxRank = 8` 的 inline 数组 | 常见 rank 2–4 零堆分配，元数据可安全按值复制 | 动态 `vector` 元数据（热路径分配、生命周期复杂） |
| `alignment == 0` 表示未知，不设 16 默认值 | `storage_offset` / slice 后的逻辑指针可能低于底层 storage 对齐；未知比错误默认更安全 | 统一默认 16 或 64，可能把不满足对齐的 view 误判为满足 |
| 分配器注册中心由 `Runtime` 按值拥有，Provider 按 `DeviceType`、实例按 `Device` 分层 | 避免全局单例；允许按设备缓存 allocator；生命周期明确 | 全局静态 `AllocatorTable` / 全局分配器查找；所有设备共用一个实例 |
| `Buffer` 通过 `MemoryHandle.deleter` 自包含释放 | Buffer 可长于 allocator，释放路径不依赖 allocator 存活；删除时无需回溯分配器 | Buffer 持有 `Allocator*` 或引用计数；手动释放 |
| `TryAllocate` 返回 `StatusOr<Buffer>`，`Allocate` 抛异常 | 冷路径可恢复 OOM / 非法请求用 Status；需要 RAII 便捷入口的场景用异常 | 只提供抛异常 API；只提供错误码 |
| CPU 存储直接走 `posix_memalign`，配合 junk fill 与可选 THP | 实现简单、对齐可控、无额外缓存层；便于定位未初始化读 | 当前不采用 ThreadCache / CentralCache / PageCache / SizeClass 缓存分配器；仓库中也没有这些实现 |
| AllocatorRegistry 当前不加锁 | 与单请求同步 Runtime 模型一致；避免未使用并发时的锁开销 | 内部 mutex（`TryGetAllocator` 中留有 TODO，待多线程 Runtime 再启用） |

当前限制（代码事实，不是承诺）：`StatusOr<T>` 没有 `operator bool`、`and_then` / `transform` / `or_else`，不支持 `StatusOr<T&>` 或 `StatusOr<void>`；默认 copy assignment 未处理 `std::variant::valueless_by_exception`；错误上下文只有扁平字符串。`TensorView` 不携带 device 字段，设备信息由来源 `Tensor` / `Buffer` 决定；视图本身不做线程同步。

## 6. 测试与验证入口

先构建测试目标，再按 suite 过滤运行：

```bash
cmake --build build --target aethermind_unit_tests -j
./build/tests/unit/aethermind_unit_tests --gtest_filter='Status.*:StatusOr.*:StatusMacros.*'
```

| 测试文件 | 覆盖内容 | 建议 filter |
|---|---|---|
| [tests/unit/base/test_status.cpp](../../tests/unit/base/test_status.cpp) | `Status` 工厂/比较/`WithMessage`/`ToString`、C ABI 映射、`StatusOr` 值/错误/移动/`value_or`、三个传播宏 | `Status.*:StatusOr.*:StatusMacros.*` |
| [tests/unit/tensor/test_tensor_view.cpp](../../tests/unit/tensor/test_tensor_view.cpp) | 借用元数据与 data、mutable 写通、raw parts 构造、numel/logical_nbytes 溢出、scalable vector 拒绝 | `TensorView.*` |
| [tests/unit/tensor/test_shape_and_stride.cpp](../../tests/unit/tensor/test_shape_and_stride.cpp) | 默认未初始化、rank-0、`[0]`、重复 `set`、rank/负维度/溢出 death tests | `ShapeAndStrideRankZero.*` |
| [tests/unit/base/test_data_type.cpp](../../tests/unit/base/test_data_type.cpp) | `DataType` code/bits/lanes 校验、类型判断、工厂、相等性、`ToString` | `DataTypeTest.*:ToStringDataTypeValueTest.*` |
| [tests/unit/base/test_scalar.cpp](../../tests/unit/base/test_scalar.cpp) | `Scalar` 与 `is_cast_overflow` / 转换边界 | `CastOverflowsTest.*:CastTest.*:Scalar*.*` |
| [tests/unit/base/test_alignment.cpp](../../tests/unit/base/test_alignment.cpp) | `AlignOffsetUp` 对齐、溢出、无分配、诊断 OOM | `OffsetAlignment*.*` |
| [tests/unit/base/test_object.cpp](../../tests/unit/base/test_object.cpp) | `Object` / `ObjectPtr` / `WeakObjectPtr` 引用计数与 sentinel | `object.*` |
| [tests/unit/base/test_mmap_file.cpp](../../tests/unit/base/test_mmap_file.cpp) | `MemoryMappedFile::Map` / `Advise` / move / 非法路径 | `MemoryMappedFile.*` |
| [tests/unit/device/test_device.cpp](../../tests/unit/device/test_device.cpp) | `Device` 校验、字符串解析、CPU index 约束 | `Device.*` |
| [tests/unit/dtypes/test_half.cpp](../../tests/unit/dtypes/test_half.cpp)、[test_half_conversion.cpp](../../tests/unit/dtypes/test_half_conversion.cpp)、[test_fp16_batch_conversion.cpp](../../tests/unit/dtypes/test_fp16_batch_conversion.cpp) | `Half` 位级转换、RNE、subnormal、NaN/符号零、舍入环境、批量转换 | `Half*.*` |
| [tests/unit/dtypes/test_bfloat16.cpp](../../tests/unit/dtypes/test_bfloat16.cpp)、[test_bf16_conversion.cpp](../../tests/unit/dtypes/test_bf16_conversion.cpp)、[test_bf16_batch_conversion.cpp](../../tests/unit/dtypes/test_bf16_batch_conversion.cpp) | BF16 转换、NaN quiet、批量转换 | `BFloat16*.*` |
| [tests/unit/dtypes/test_float8_e4m3fn.cpp](../../tests/unit/dtypes/test_float8_e4m3fn.cpp)、[test_float8_e5m2.cpp](../../tests/unit/dtypes/test_float8_e5m2.cpp)、[test_reduced_float_negation.cpp](../../tests/unit/dtypes/test_reduced_float_negation.cpp) | FP8 格式边界、NaN/溢出、符号翻转 | `Float8*.*:FP8*.*:ReducedFloat*.*` |
| [tests/unit/memory/test_buffer.cpp](../../tests/unit/memory/test_buffer.cpp) | 空/零字节 Buffer、共享 impl、move、deleter 次数、并发副本 | `Buffer.*` |
| [tests/unit/memory/test_cpu_allocator.cpp](../../tests/unit/memory/test_cpu_allocator.cpp) | 对齐/零字节/非法对齐、OOM 返回 Status、元数据 OOM 回收、Registry/Provider、Buffer 长于 Runtime、hook 合同 | `*AlignedCpuBuffer*.*:CPUAllocator*.*:AllocatorOwnership.*:AllocatorHookContract*.*` |
| [tests/unit/backend/cpu/test_cpu_workspace_arena.cpp](../../tests/unit/backend/cpu/test_cpu_workspace_arena.cpp)、[tests/unit/execution/test_workspace_requirement_planning.cpp](../../tests/unit/execution/test_workspace_requirement_planning.cpp) | workspace 契约与规划消费方 | `*Workspace*.*` |

`test_cpu_allocator.cpp` 中依赖 glibc malloc interposer 的用例在不可用时自行 `GTEST_SKIP`；`AlignedCpuBuffer.SharedBufferRemainsWritableAfterRuntimeDestruction` 与 `AllocatorOwnership.BufferOutlivesItsAllocatorAndReleasesStorage` 是 Buffer 生命周期/释放时机的关键回归。内存行为的整体约束（Decode 稳态零分配）见 [架构文档 §11](../architecture.md)。
