## 1. 总体方案

AetherMind 在 C++ 层同时维护两套错误处理机制，分别用于不同边界：

- **异常路径**：基于 `aethermind::Error`（继承自 `std::exception`）+ `AM_THROW` / `AM_THROW_SILENT` 宏 + `ErrorBuilder` RAII 构建器 + `TRACEBACK_HERE`，提供带 `kind`、`message`、traceback 的强类型异常，主要用于内部不可恢复的编程错误和断言失败。
- **返回值路径**：基于 `StatusCode` 枚举 + `Status` + `StatusOr<T>` + `AM_RETURN_IF_ERROR`，模仿 absl::Status/gRPC 风格，作为推理运行时主错误模型，避免异常开销。

两条路径通过 `c_api.h` 中的 `am_status_code` 对齐，C ABI 边界统一用 `am_status_code` 暴露给外部。设计文档 `docs/designs/status设计方案.md` 明确声明 Status 是 "Phase 1 runtime" 的主要错误处理机制，用来替换 exception-heavy 路径。

## 2. 关键文件

- `include/aethermind/base/error.h` — `Error` 异常基类、`ErrorBuilder`、`AM_THROW` / `AM_THROW_SILENT` / `TRACEBACK_HERE` 宏定义。
- `include/aethermind/base/traceback.h` + `src/base/traceback.cpp` — 基于 libbacktrace 的可配置回溯收集（受 `USE_LIBBACKTRACE`、`BACKTRACE_ON_SEGFAULT`、`TRACEBACK_LIMIT` 环境变量控制），支持 Python/C++ 语言边界过滤、gtest 帧排除、符号消解。
- `include/aethermind/base/status.h` + `src/base/status.cpp` — `StatusCode` 枚举（18 个值）、`Status`、`StatusOr<T>`、`AM_RETURN_IF_ERROR`、`StatusCodeName`、`ToAMStatusCode` / `FromAMStatusCode`。
- `include/c_api.h` — C ABI 侧的 `am_status_code` 枚举及 `AM_STATUS_*` 常量，与 `StatusCode` 通过 `static_assert` 一对一绑定；另提供 `am_error_create` / `am_error_destroy` / `am_error_code` / `am_error_message` 诊断对象 API。
- `src/c_api_error.cpp` — C ABI 诊断对象的实现，将 `am_status_code` + message 封装为独立拥有的 `AMErrorObject`，所有方法不抛出异常。
- `tests/unit/base/test_error.cpp`、`tests/unit/base/test_status.cpp` — 覆盖两类错误模型的测试。

## 3. 架构与约定

### 3.1 异常体系 (`Error`)

- `Error` 继承 `std::exception`，携带三个字段：`kind_`（如 `"value_error"`）、`message_`、`traceback_`。
- `what()` 以 `Traceback (most recent call last):\n<reversed traceback><kind>: <message>` 格式输出，使用 thread-local 缓存；若格式化过程中抛出异常（例如 `bad_alloc`），降级为仅 `kind: message`。
- `UpdateTraceback` 允许在捕获后追加或替换 traceback，配合 `BacktraceUpdateMode`（replace/append）。
- `ErrorBuilder` 是 RAII 流式构建器：构造时记录 kind 与初始 traceback，析构时若不在栈展开中则输出到 `stderr` 并抛出 `Error`；若在展开中则静默吞掉以避免二次 throw 触发 `std::terminate`。
- `AM_THROW(kind) << ...` 与 `AM_THROW_SILENT(kind) << ...` 是统一的抛错入口；`TRACEBACK_HERE` 捕获当前源位置并通过 `AetherMindTraceback` 生成 traceback。
- 代码中实际使用的 kind 包括 `runtime_error`、`LogicError`、`value_error` 等，由调用方语义决定（未见集中枚举）。

### 3.2 Status/StatusOr 体系

- `StatusCode` 是 `uint8_t` 枚举，含 18 个值（OK、Cancelled、Unknown、InvalidArgument、DeadlineExceeded、NotFound、AlreadyExists、PermissionDenied、ResourceExhausted、FailedPrecondition、Aborted、OutOfRange、Unimplemented、Internal、Unavailable、DataLoss、Overflow、Unauthenticated），与 gRPC 状态码对齐，另增加 `kOverflow`、`kUnauthenticated`。
- `static_assert` 强制 `StatusCode` 与 `am_status_code` 数值一一对应；新增 code 时必须同步修改 `c_api.h`，否则编译期失败。
- `Status` 提供静态工厂方法（`Status::NotFound(msg)` 等）、`ok()`、隐式 `bool` 转换、`ToString()`、`WithMessage()`、`operator==`/`!=` 直接比较 `StatusCode`。
- `message()` 对 rvalue 重载被 `= delete`，防止临时对象上取引用导致悬垂。
- `StatusOr<T>` 是 `variant<T, Status>` 风格的联合类型，配合 `detail::ExtractStatus` 与 `AM_RETURN_IF_ERROR(expr)` 宏实现链式错误传播：表达式求值一次，非 OK 直接返回，OK 继续执行。
- `Status` 的不变式：`code_ == kOk` 时 `message_` 为空；只有公共工厂能构造非 OK 状态，保证该不变式。

### 3.3 Traceback 子系统

- 启用 `USE_LIBBACKTRACE` 时，通过 `libbacktrace` + `abi::__cxa_demangle` 采集完整栈帧，按 `TRACEBACK_LIMIT`（默认 512）限制深度。
- `DetectBoundary` 在 Python C ABI 入口（`slot_tp_call`、`object_is_not_callable`、`_Py*`、`PyObject*`）处停止收集，避免与上层 Python 堆栈重复。
- `ExcludeFrame` 排除 `src/base/traceback.cpp`、C++ stdlib、`__libc_`、gtest 相关帧。
- 未启用 libbacktrace 时，`traceback.cpp` 提供 stub 实现，使包含 `error.h` 的代码仍可链接。
- `BACKTRACE_ON_SEGFAULT` 开关下可注册信号处理器打印崩溃回溯。

### 3.4 C ABI 边界

- 所有面向 C 的错误均通过 `am_status_code` 传递；`ToAMStatusCode` / `FromAMStatusCode` 做双向转换，未知值映射为 `kUnknown`。
- `src/c_api_error.cpp` 提供 `am_error_create` / `am_error_destroy` / `am_error_code` / `am_error_message`，封装 `AMErrorObject { am_status_code, std::string }`，所有函数 catch 所有异常并返回空指针或默认值，确保 C 边界不抛出异常。

## 4. 约定与约束

- **运行时主路径使用返回值而非异常**：`status.h` 注释明确要求 "the primary error handling mechanism for the inference runtime, replacing exception-heavy paths with explicit status checks"；大多数算子与模型加载函数返回 `Status` 或 `StatusOr<T>`。
- **内部编程错误使用异常**：`AM_THROW(RuntimeError)` / `AM_THROW(LogicError)` / `AM_THROW(value_error)` 集中在 dispatch 分支的 default 路径、参数校验失败、不支持的数据类型等位置（如 `src/base/device.cpp`、`src/base/tensor.cpp`、`src/base/scalar.cpp`、`src/container/string.cpp`）。
- **StatusCode 与 C ABI 必须同步变更**：`status.h` 中 18 条 `static_assert("Status code mismatch!")` 强制 `StatusCode` 与 `c_api.h` 的 `am_status_code` 保持一一对应，新增 code 不修改 C ABI 头会编译失败。
- **禁止从临时 Status 取 message 引用**：rvalue `message()` 重载被显式删除，阻止 `MakeStatus().message()` 这类悬挂引用。
- **Status 消息仅在非 OK 时有效**：`WithMessage` 对 OK 状态直接返回 `Ok()`，不附加消息；构造函数私有化，仅公开工厂可构造非 OK 状态。
- **Traceback 深度可配置**：通过 `TRACEBACK_LIMIT` 环境变量调整，默认 512；Python 边界自动截断。
- **`AM_RETURN_IF_ERROR` 只评估一次表达式**：宏通过 `__COUNTER__` 生成唯一变量名，确保被展开的表达式恰好求值一次。
- **`ErrorBuilder` 在栈展开期间不抛异常**：通过 `std::uncaught_exceptions() > 0` 检测，避免二次 throw 触发终止，属于安全兜底行为。
- **C 诊断对象永不抛出**：`am_error_create` 包裹 new 操作并在 catch 中返回 NULL；`am_error_code` / `am_error_message` 对 NULL handle 返回安全的默认值。

## 5. 观察到的不一致

部分内部代码仍使用 `AM_THROW(RuntimeError)`（大写 R），而 `error.h` 示例中使用 `value_error`（小写 snake_case），说明 kind 字符串目前由调用方自由指定，没有集中枚举约束——这是约定的弱项，而非编译期强制规则。
