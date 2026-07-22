# Status 设计方案

## 1. 设计意图

### 1.1 为什么需要统一错误模型

推理引擎中从模型加载到执行的完整路径包含大量可预期失败：

- 模型配置、权重和算子参数解析失败；
- Graph IR 节点、端口、shape 或 state binding 不合法；
- 图编译、常量折叠和执行计划构建失败；
- backend 或 kernel 尚未注册、无法匹配；
- kernel 输入 dtype、shape、stride 或内存地址不合法；
- KV Cache 越界、容量不足或数值计算溢出。

这些错误需要同时满足：

1. **统一传播**：跨图构建、编译、执行和 backend 层逐级返回。
2. **保留上下文**：不仅知道失败，还需要错误类别和可读消息。
3. **可组合**：返回值与错误必须能组成同一个函数结果。
4. **调用端可检查**：减少忽略错误或使用未初始化输出的可能。
5. **兼容 C ABI**：错误码需要稳定映射到 `am_status_code`。
6. **避免异常成为正常控制流**：尤其是 kernel 和逐 token 执行路径。

`Status` 表达“成功或错误”，`StatusOr<T>` 表达“成功值或错误”，共同形成统一错误契约。

### 1.2 与其他方案的取舍

| 方案 | 优点 | 主要问题 | AetherMind 取舍 |
|---|---|---|---|
| 异常 | 自动栈展开；成功路径代码简洁 | 不能穿越 C ABI；异常策略和 `noexcept` 难统一；错误路径成本与控制流不直观 | 业务失败不用异常；当前实现仅在 API 误用时抛异常 |
| 原始错误码 | 成本低、易映射 C ABI | 缺少消息；容易混用；返回值需要额外通道 | 错误码封装进 `Status` |
| `bool` + 输出参数 | 实现简单 | 错误原因丢失；输出可能未初始化或被部分修改；组合性差 | 不推荐 |
| `std::optional<T>` | 能表达“有值/无值” | 无法说明为什么没有值 | 只适合“缺失也是正常结果”，不能替代 `StatusOr<T>` |
| `std::expected<T, E>` | 与 `StatusOr<T>` 语义接近 | `std::expected` 是 C++23 API，项目当前使用 C++20 | 可作为未来内部实现候选 |
| `Status` / `StatusOr<T>` | 显式、可携带消息、可组合、兼容 C ABI | 需要显式检查；传播宏有宏本身的限制 | 当前方案 |

对于 `StatusOr<std::optional<T>>`，三种状态具有不同含义：

- `StatusOr` 错误：操作失败；
- `StatusOr` 成功但 `optional` 为空：合法地没有值；
- `StatusOr` 成功且 `optional` 有值：成功取得值。

## 2. 总体设计

三者关系如下：

```text
StatusCode
    └── 表示机器可判断的错误类别

Status
    ├── StatusCode
    └── std::string message

StatusOr<T>
    └── std::variant<T, Status>
```

使用规则：

- 只需要报告成功或失败：返回 `Status`。
- 成功时需要返回值：返回 `StatusOr<T>`。
- 业务错误通过 `Status` 传播。
- `StatusOr` API 误用目前通过异常暴露。

## 3. StatusCode 设计

### 3.1 错误码

```cpp
enum class StatusCode : uint8_t {
    kOk = 0,
    kCancelled,
    kUnknown,
    kInvalidArgument,
    kDeadlineExceeded,
    kNotFound,
    kAlreadyExists,
    kPermissionDenied,
    kResourceExhausted,
    kFailedPrecondition,
    kAborted,
    kOutOfRange,
    kUnimplemented,
    kInternal,
    kUnavailable,
    kDataLoss,
    kOverflow,
    kUnauthenticated
};
```

这些错误码与 `c_api.h` 中的 `am_status_code` 数值一一对应。头文件通过逐项 `static_assert` 保证 C++ API 与 C ABI 不会静默漂移。

新增错误码时必须同步修改：

1. `StatusCode`；
2. `am_status_code`；
3. 两者之间的 `static_assert`；
4. `StatusCodeName()`；
5. 对应工厂函数和测试。

### 3.2 转换 API

```cpp
constexpr std::string_view StatusCodeName(StatusCode code) noexcept;
constexpr am_status_code ToAMStatusCode(StatusCode code) noexcept;
constexpr StatusCode FromAMStatusCode(am_status_code code) noexcept;
```

`FromAMStatusCode()` 会将超出合法范围的 C 状态码映射为 `StatusCode::kUnknown`，避免直接把非法整数转换为枚举值。

## 4. Status 设计

### 4.1 内部表示与不变量

```cpp
class AM_NODISCARD Status {
    StatusCode code_;
    std::string message_;
};
```

核心不变量：

```text
code_ == StatusCode::kOk  =>  message_.empty()
```

反向不成立：错误状态允许携带空消息。

非 OK 构造函数是私有的，外部只能通过命名工厂创建错误，从而维持错误码和消息之间的约束。

### 4.2 成功和错误构造

```cpp
Status() noexcept;
static Status Ok() noexcept;
```

错误工厂均接收 `std::string_view`：

```cpp
Status::Cancelled(message)
Status::Unknown(message)
Status::InvalidArgument(message)
Status::DeadlineExceeded(message)
Status::NotFound(message)
Status::AlreadyExists(message)
Status::PermissionDenied(message)
Status::ResourceExhausted(message)
Status::FailedPrecondition(message)
Status::Aborted(message)
Status::OutOfRange(message)
Status::Unimplemented(message)
Status::Internal(message)
Status::Unavailable(message)
Status::DataLoss(message)
Status::Overflow(message)
Status::Unauthenticated(message)
```

当前没有 `Ok()`、`InvalidArgument()` 等同名自由函数。

### 4.3 查询和调试 API

| API | 用途 |
|---|---|
| `bool ok() const noexcept` | 判断是否为 `kOk` |
| `explicit operator bool() const noexcept` | 支持 `if (status)` |
| `StatusCode code() const noexcept` | 获取错误码 |
| `message() &` / `message() const&` | 获取消息引用 |
| `Status WithMessage(std::string) const noexcept` | 保留错误码并替换消息 |
| `std::string ToString() const` | 生成可读调试字符串 |
| `operator==(const Status&)` | 比较错误码和消息 |
| `operator==(StatusCode)` | 仅比较错误码 |

`ToString()` 的结果形式为：

```text
OK
INTERNAL
NOT_FOUND: weight tensor missing
```

`WithMessage()` 是替换而不是追加，并且不会修改原对象。对 OK 状态调用时仍返回无消息的 OK 状态。

### 4.4 生命周期保护

`message()` 只允许在左值上调用：

```cpp
const std::string& message() & noexcept;
const std::string& message() const& noexcept;

const std::string& message() && noexcept = delete;
const std::string& message() const&& noexcept = delete;
```

这样可以阻止以下悬垂引用：

```cpp
// 编译失败：临时 Status 销毁后引用会悬空
const std::string& message = Status::Internal("failure").message();
```

## 5. StatusOr\<T\> 设计

### 5.1 内部表示

```cpp
std::variant<T, Status> storage_;
```

`std::variant` 在对象内部保存 `T` 或 `Status`，不使用手写 `union` 或 `std::optional`。variant 本身通常不额外分配内存，但 `Status` 内部的 `std::string` 可能分配。

具体对象大小、对齐和 discriminant 布局属于标准库实现细节，不能作为稳定 ABI。此结论基于 `std::variant` 的标准语义推断。

核心不变量：

```text
storage_ 持有 T            => ok() == true
storage_ 持有非 OK Status => ok() == false
storage_ 不应持有 Status::Ok()
```

### 5.2 类型约束

```cpp
static_assert(!std::is_reference_v<T>);
static_assert(!std::is_same_v<std::remove_cv_t<T>, Status>);
static_assert(std::is_nothrow_move_constructible_v<T>);
static_assert(std::is_nothrow_move_assignable_v<T>);
```

因此：

- 不支持 `StatusOr<T&>`；
- 不支持 `StatusOr<Status>`；
- 支持原始指针，但不表达所有权；
- 支持 `std::unique_ptr<T>` 等 nothrow-movable 类型；
- 一般不应使用 `StatusOr<const T>`，因为 const 类型通常不可移动赋值；
- 如需借用对象，可返回指针，但必须另行约定非空性和生命周期。

例如 `StatusOr<const KernelDescriptor*>` 只表示“得到一个指针或发生错误”，它不会自动禁止成功状态中的空指针。

### 5.3 构造与特殊成员

```cpp
template<typename U = T>
StatusOr(U&& value);

template<typename... Args>
StatusOr(std::in_place_t, Args&&... args);

StatusOr(const Status& status);
StatusOr(Status&& status);

StatusOr(const StatusOr&) = default;
StatusOr(StatusOr&&) noexcept = default;
StatusOr& operator=(const StatusOr&) = default;
StatusOr& operator=(StatusOr&&) noexcept = default;
~StatusOr() = default;
```

特点：

- 没有默认构造函数，不会产生“既无值又无错误”的普通状态；
- 值构造函数允许从可转换类型隐式形成成功结果；
- `std::in_place` 用于直接构造 `T`；
- 使用 OK `Status` 构造 `StatusOr` 会抛出 `std::invalid_argument`；
- 如果 `T` 不可复制，默认 copy 操作会被编译器隐式删除；
- move 构造和赋值通过类型约束保证 `noexcept`。

### 5.4 状态和值访问

| API | 成功状态 | 错误状态 |
|---|---|---|
| `ok()` | `true` | `false` |
| `get_if_ok()` | 返回 `T*`/`const T*` | 返回 `nullptr` |
| `status() const&` | 返回共享静态 OK 状态 | 返回内部错误引用 |
| `status() &&` | 返回 `Status::Ok()` | 移出内部错误 |
| `value()` | 返回值引用或 `T&&` | 抛出 `std::logic_error` |
| `operator*` | 返回值引用或 `T&&` | `std::get` 抛出 `std::bad_variant_access` |
| `operator->` | 返回值指针 | `std::get` 抛出 `std::bad_variant_access` |
| `value_or(fallback)` | 返回当前值 | 返回 fallback |
| `operator==(StatusCode)` | 与 `kOk` 匹配 | 与内部错误码匹配 |

`StatusOr<T>` 当前**没有 `operator bool`**，调用方必须使用：

```cpp
if (!result.ok()) {
    return result.status();
}
```

### 5.5 const 与右值语义

安全借用只允许来自左值：

```cpp
const T* get_if_ok() const& noexcept;
T* get_if_ok() & noexcept;

const T* operator->() const&;
T* operator->() &;
```

`get_if_ok()` 和 `operator->()` 的右值重载被删除，避免返回指向临时 `StatusOr` 内部的指针。

移动取值则显式支持：

```cpp
T&& value() &&;
T&& operator*() &&;
Status status() && noexcept;
```

正确移动方式：

```cpp
StatusOr<std::unique_ptr<Foo>> result = CreateFoo();
if (!result.ok()) {
    return result.status();
}

std::unique_ptr<Foo> foo = std::move(result).value();
```

不要把 `value() &&` 或 `operator*() &&` 的返回结果绑定到生命周期超过原 `StatusOr` 的引用。

### 5.6 value_or() 的限制

```cpp
template<typename U>
T value_or(U&& fallback) const&;

template<typename U>
T value_or(U&& fallback) &&;
```

左值版本需要能够复制 `T`；对 move-only 类型应使用右值版本。`value_or()` 会丢弃错误状态，因此只适合“失败等价于使用默认值”的业务语义，不应替代正常错误传播。

## 6. AM_NODISCARD 与传播宏

`AM_NODISCARD` 在编译器支持时展开为：

```cpp
[[nodiscard]]
```

`Status` 和 `StatusOr<T>` 都带有类级 `AM_NODISCARD`。忽略返回结果通常会产生编译警告；是否升级为构建失败取决于编译选项，因此它不是绝对的语言级强制处理。

### 6.1 AM_RETURN_IF_ERROR

```cpp
AM_RETURN_IF_ERROR(expr);
```

语义：

1. `expr` 只求值一次；
2. `expr` 可以返回 `Status` 或 `StatusOr<T>`；
3. 成功时继续执行；
4. 失败时提取 `Status` 并从当前函数提前返回；
5. 当前函数必须返回 `Status`，或能从 `Status` 构造的 `StatusOr<U>`。

内部使用 `detail::ExtractStatus()` 和 `__COUNTER__`，避免临时变量名冲突。

### 6.2 AM_RETURN_IF_ERROR_WITH_MSG

```cpp
AM_RETURN_IF_ERROR_WITH_MSG(expr, "load model");
```

该宏只接受返回 `Status` 的表达式。失败时保留错误码，将消息替换为：

```text
load model: 原始消息
```

当前函数可以返回 `Status` 或 `StatusOr<U>`。如果原始消息为空，当前实现仍会生成带 `": "` 后缀的消息。

### 6.3 AM_ASSIGN_OR_RETURN

```cpp
AM_ASSIGN_OR_RETURN(const GraphValueId value, AddLinear(...));
```

语义：

1. `expr` 只求值一次，且必须返回 `StatusOr<T>`；
2. 失败时返回其 `Status`；
3. 成功时把值移动到 `lhs`；
4. `lhs` 可以是声明，也可以是已有变量。

该宏没有使用 `do { ... } while (false)` 包装，因为声明的变量需要在宏之后仍然可见。因此不要直接把它作为无大括号 `if/else` 的单条语句：

```cpp
// 不推荐
if (condition)
    AM_ASSIGN_OR_RETURN(auto value, GetValue());

// 推荐
if (condition) {
    AM_ASSIGN_OR_RETURN(auto value, GetValue());
}
```

包含逗号的宏参数也需要注意预处理器参数分隔规则。

## 7. 使用示例

### 7.1 只传播错误

仓库中的模型输入验证采用 `Status`：

```cpp
Status ValidateInputs(const HfModelConfig& config,
                      const ResolvedModelWeights& weights) {
    AM_RETURN_IF_ERROR(HfModelValidator::ValidateConfig(config));
    return HfModelValidator::ValidateResolvedModel(config, weights);
}
```

### 7.2 解包成功值

图构建使用 `StatusOr<GraphValueId>` 和 `AM_ASSIGN_OR_RETURN` 组合多个步骤：

```cpp
AM_ASSIGN_OR_RETURN(const GraphValueId normed, AddRmsNorm(...));
AM_ASSIGN_OR_RETURN(const GraphValueId q, AddLinear(...));
AM_ASSIGN_OR_RETURN(const GraphValueId k, AddLinear(...));
```

任何一步失败都会立即返回错误，不会继续构建依赖无效值的后续节点。

### 7.3 Graph IR 错误上下文

`ModelGraph::AddNode()` 返回 `StatusOr<AddedNode>`。当语义分析失败时，它将节点编号、算子类型和名称写入消息：

```cpp
auto inference_or = AnalyzeOperator(op_type, op_params, all_input_specs);
if (!inference_or.ok()) {
    return Status::InvalidArgument(
            get_msg("AnalyzeOperator: " + inference_or.status().message()));
}
```

这说明 `StatusCode` 用于稳定分类，而消息用于携带图节点上下文。

### 7.4 kernel 解析

`KernelRegistry::Resolve()` 返回：

```cpp
StatusOr<const KernelDescriptor*>
```

可能结果包括：

- registry 未冻结：`Status::FailedPrecondition(...)`；
- 没有匹配 kernel：`Status::NotFound(...)`；
- 成功：返回 `const KernelDescriptor*`。

CPU Add kernel 参数解析返回：

```cpp
StatusOr<cpu::detail::AddKernelArgs>
```

输入 TensorView、dtype、rank、broadcast、stride 或地址不合法时返回 `InvalidArgument`，成功时返回完整且已验证的参数对象。这避免了 `bool + AddKernelArgs*` 产生半初始化输出。

### 7.5 执行阶段

`LayerRunner::Run()` 返回 `Status`，任何 step 失败都会停止后续执行：

```cpp
if (const auto status = RunStep(i, steps[i], bindings, alias_plan);
    !status.ok()) {
    return status;
}
```

该模式适合需要在返回前检查、记录或包装错误的场景；无需额外逻辑时可使用 `AM_RETURN_IF_ERROR`。

## 8. 实施约束与注意事项

### 8.1 避免未检查错误

推荐：

```cpp
auto result = TryBuild();
if (!result.ok()) {
    return result.status();
}
auto value = std::move(result).value();
```

不推荐：

```cpp
auto value = TryBuild().value();
```

后者虽然可以编译，但失败时会抛异常，而不是通过 `Status` 正常传播。

### 8.2 不混用业务异常和 Status

建议边界：

- 配置错误、资源不存在、shape 不匹配、kernel 不支持：返回 `Status`；
- API 不变量被调用方违反：断言、检查失败或明确的编程错误机制；
- 不要在同一业务错误路径上有时返回 `Status`、有时抛异常。

当前实现尚未完全统一这一边界：错误构造和 `value()` 误用会抛异常，而部分项目不变量使用 `AM_CHECK` 终止进程。

### 8.3 noexcept 路径

即使函数返回 `Status`，错误消息中的 `std::string` 构造仍可能因内存分配抛出异常。在声明为 `noexcept` 的 kernel 或执行函数中，这类异常会触发 `std::terminate`。

因此，当前机制保证的是“业务错误显式返回”，并不代表所有底层资源错误都可恢复。此结论基于 C++ `std::string` 和 `noexcept` 语义推断。

### 8.4 指针与生命周期

`StatusOr<T*>` 不拥有对象，也不自动检查空指针。API 必须额外说明：

- 成功状态是否允许空指针；
- 指针由谁拥有；
- 指针在多长时间内有效；
- 是否允许跨线程保存。

### 8.5 线程安全

头文件明确说明：

- const 方法可并发读取；
- `StatusOr` 不支持并发修改；
- `status() const&` 返回的共享 OK 状态是 `static const`，只读访问安全。

对象赋值、移动或内部值修改仍需要调用方同步。

## 9. 当前完成度与差距

### 9.1 已完成

- 统一的 `StatusCode`、`Status`、`StatusOr<T>`；
- C++ 与 C ABI 错误码静态校验；
- 错误码和消息封装；
- class-level `[[nodiscard]]`；
- 值、错误和 in-place 构造；
- move-only 类型支持；
- ref-qualified 访问器和临时对象生命周期保护；
- 三种错误传播宏；
- 工厂、宏、移动语义和访问器的单元测试；
- 图构建、执行、kernel 查找和参数解析等模块集成。

### 9.2 当前差距

1. **`std::variant::valueless_by_exception`**

   `StatusOr` 使用默认 copy assignment。如果跨 alternative 拷贝一个可能抛异常的 `T`，`std::variant` 在标准允许的情况下可能进入 `valueless_by_exception`。当前 `ok()`、`status()` 没有处理该状态。

2. **错误上下文是扁平字符串**

   `WithMessage()` 只能替换字符串；没有结构化 cause chain、错误 payload 或 `std::source_location`。

3. **误用策略不统一**

   `value()` 抛 `std::logic_error`，`operator*`/`operator->` 可能抛 `std::bad_variant_access`，其他不变量可能通过 `AM_CHECK` 终止。

4. **传播宏存在语法限制**

   `AM_ASSIGN_OR_RETURN` 不是单语句宏；`AM_RETURN_IF_ERROR_WITH_MSG` 不能直接接收 `StatusOr<T>`。

5. **缺少 monadic API**

   当前没有 `and_then`、`transform`、`or_else` 等组合操作，复杂链路主要依赖传播宏。

6. **不支持引用和 void**

   `StatusOr<T&>` 被明确禁止；基于当前 `std::variant<T, Status>` 的实现也不能实例化 `StatusOr<void>`。

## 10. 建议演进路线

### 10.1 阶段一：强化当前不变量

- 为 `valueless_by_exception` 风险补充测试和明确策略；
- 约束可复制 `T` 的异常保证，或实现自定义 copy assignment；
- 统一 `value()`、解引用和 `AM_CHECK` 的编程错误处理原则；
- 明确所有 `StatusOr<T*>` API 的空指针和生命周期契约。

### 10.2 阶段二：增强上下文

- 提供不会丢失原始错误含义的上下文追加机制；
- 评估结构化 cause、模块信息和 `std::source_location`；
- 保持 `StatusCode` 稳定，避免调用端依赖消息文本分类。

### 10.3 阶段三：评估 C++23

项目升级到 C++23 后，可评估以 `std::expected<T, Status>` 作为内部实现，但应优先保持现有公共接口和 C ABI 映射稳定。是否迁移需要结合编译器支持、对象布局、性能和现有宏兼容性验证。

## 11. 总结

AetherMind 当前的 `Status` / `StatusOr<T>` 方案以 C++20 为基础，通过“错误码 + 消息 + 显式值状态”统一模型加载、图编译、执行和 kernel 层的错误处理。

其核心原则是：

- 业务失败显式返回；
- 无返回值使用 `Status`；
- 有返回值使用 `StatusOr<T>`；
- 调用端检查 `ok()` 后再访问值；
- 使用传播宏减少样板代码；
- 用 `StatusCode` 做稳定分类，用消息携带上下文；
- 不让异常跨越运行时或 C ABI 边界。

当前实现已具备完整的基础能力，但仍需继续统一误用策略、强化 `std::variant` 异常状态不变量，并改善结构化错误上下文。
