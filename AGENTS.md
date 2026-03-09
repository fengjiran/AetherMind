# 🌌 AetherMind: 项目上下文与架构蓝图

> **[AI 助手指令 / AI Assistant Instructions]**
> 本文件为本仓库的智能编码代理执行指南。如果你是 AI 编程助手（如 Google Gemini、Cursor 等）正在读取此文件，请将此文档作为你的**主要执行指南与架构参考**；若文档与当前代码、CMake 或用户指令冲突，应以已验证的仓库事实和用户指令为准。除非用户给出冲突指令，否则默认遵循本指南。
> 你是一个资深的大模型推理引擎架构师，每当你在 `AetherMind` 项目中生成、重构或审查代码时，**必须严格遵守**本文档中定义的架构约束、C++20 现代编程实践以及性能优化指南。

## 1. 项目事实
- 语言：C++20（`CMAKE_CXX_STANDARD 20`）
- 构建系统：CMake >= 3.28
- 核心库目标：`AetherMind`（shared）
- 分配器目标：`ammalloc`（static）
- 单测目标：`aethermind_unit_tests`（GoogleTest，系统安装）
- 基准目标：`aethermind_benchmark`（google benchmark，FetchContent 拉取）
- FetchContent 依赖：`spdlog v1.17.0`、`google_benchmark v1.9.5`
- 关键 CMake 选项：`BUILD_TESTS`、`BUILD_BENCHMARKS`、`USE_LIBBACKTRACE`、`BACKTRACE_ON_SEGFAULT`、`ENABLE_TSAN`
- 测试侧可选项：`BUILD_WITH_TORCH`（定义于 `tests/unit/CMakeLists.txt`，用于联合 Torch 构建测试目标）

## 2. 仓库结构
- `include/`：核心公共头文件（`Tensor`、`Device`、`DataType`、`Error` 等）
- `src/`：核心实现
- `ammalloc/include/ammalloc/`：分配器公共/内部头文件
- `ammalloc/src/`：分配器实现
- `tests/unit/`：gtest 单元测试
- `tests/benchmark/`：benchmark 源码
- `cmake/`：CMake 辅助模块（如 `AddLibbacktrace.cmake`）
- `docs/`：架构与开发文档
- `tools/`：辅助工具
- `3rdparty/`：第三方代码（非必要不要修改）

## 3. 构建命令
```bash
# 默认配置（测试和 benchmark 默认 ON）
cmake -S . -B build

# 显式配置
cmake -S . -B build -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON \
  -DUSE_LIBBACKTRACE=ON -DBACKTRACE_ON_SEGFAULT=ON

# 全量构建
cmake --build build -j

# 按目标构建（推荐：只构建受影响目标）
cmake --build build --target AetherMind -j
cmake --build build --target ammalloc -j
cmake --build build --target aethermind_unit_tests -j
cmake --build build --target aethermind_benchmark -j

# TSAN 变体
cmake -S . -B build-tsan -DENABLE_TSAN=ON -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF
cmake --build build-tsan --target aethermind_unit_tests -j

# 可选 Torch 联合构建
cmake -S . -B build-torch -DBUILD_TESTS=ON -DBUILD_WITH_TORCH=ON
cmake --build build-torch --target aethermind_unit_tests -j
```

## 4. 测试命令
注意：顶层 CMake 未接入 `ctest`，直接运行可执行文件。
```bash
# 全部单元测试
./build/tests/unit/aethermind_unit_tests

# 单个测试套件
./build/tests/unit/aethermind_unit_tests --gtest_filter=Tensor.*

# 单个测试用例（推荐快速回路）
./build/tests/unit/aethermind_unit_tests --gtest_filter=Tensor.random

# 调试用参数
./build/tests/unit/aethermind_unit_tests --gtest_filter=Device.* --gtest_break_on_failure
./build/tests/unit/aethermind_unit_tests --gtest_filter=Tensor.* --gtest_repeat=10
```

## 5. 基准命令
```bash
./build/tests/benchmark/aethermind_benchmark
./build/tests/benchmark/aethermind_benchmark --benchmark_filter=BM_Malloc_Churn
```

## 6. Lint / 格式化
代码格式以根目录 `.clang-format` 为准（基于 LLVM，缩进 4 空格，ColumnLimit 0 即不限行宽）。
```bash
# 格式检查
clang-format -n --Werror $(git ls-files '*.h' '*.hpp' '*.cpp')
# 自动格式化
clang-format -i $(git ls-files '*.h' '*.hpp' '*.cpp')
```
若本地可用，可基于 `build/compile_commands.json` 运行 `clang-tidy`。

## 7. 代码风格规范

### 7.1 头文件与文件组织
- **Include guard**：使用 `#ifndef AETHERMIND_<NAME>_H` / `#define` / `#endif`，不用 `#pragma once`。
- 文件名使用 `snake_case`。
- **包含顺序**：先项目头文件，后标准库头文件（参见现有代码模式）。
- 公共 API 头文件放在 `include/` 或 `ammalloc/include/ammalloc/`。
- 除非任务明确要求，不修改 `3rdparty/`。

### 7.2 格式与排版
- 严格遵循 `.clang-format`：4 空格缩进，不使用 Tab。
- 指针对齐为左对齐（`Type* ptr`）。
- 大括号不换行（AfterFunction: false, AfterClass: false）。
- 空行保持克制（`MaxEmptyLinesToKeep: 2`）。
- `ColumnLimit: 0`——不限行宽，但保持合理可读性。

### 7.3 项目类型与别名
- `String`：项目自定义字符串类型（`container/string.h`），非 `std::string`。
- `ObjectPtr<T>`：侵入式引用计数智能指针（`object.h`）。
- `DataType`：数据类型封装（`data_type.h`）。
- `Device`：设备抽象，`DeviceType` 枚举含 `kCPU`、`kCUDA`、`kCANN`。
- 位宽敏感场景优先固定宽度整数（`int64_t`、`uint32_t`）；大小/索引用 `size_t`。
- 语义明确时使用 `const`、`noexcept`、`explicit`。
- 不应忽略的返回值使用 `AM_NODISCARD`。

### 7.4 命名约定
- 类型/类：`PascalCase`（`Tensor`、`RuntimeConfig`、`DeviceType`）。
- 函数：核心 API 为 `snake_case`；以所在文件/子系统现有风格为准。
- 枚举成员：`kPrefix`（`kCPU`、`kCUDA`、`kUndefined`）。
- 宏/常量：`AM_` 前缀 + 大写下划线（`AM_ALWAYS_INLINE`、`AM_CHECK`）。
- 命名空间：`aethermind` 与 `aethermind::details`。

### 7.5 错误处理与断言
- **抛异常**：`AM_THROW(ErrorKind) << message`——流式构造 `Error` 对象并抛出。
- **前置条件/不变量**：`AM_CHECK(condition, fmt, args...)`——失败时打印格式化消息并 `abort()`。
  - 使用 `std::format` 语法：`AM_CHECK(idx < size(), "index {} out of range {}", idx, size())`。
  - Debug-only 断言：`AM_DCHECK(...)`（Debug 下等价于 `AM_CHECK`；Release 下不会触发失败处理，但条件表达式仍需可编译）。
- **不可达代码**：`AM_UNREACHABLE()`。
- 除明确防御性清理路径外，不要静默吞错。

### 7.6 性能宏
- 分支预测：`AM_LIKELY` / `AM_UNLIKELY`（映射到 `[[likely]]` / `[[unlikely]]`）。
- 内联控制：`AM_ALWAYS_INLINE` / `AM_NOINLINE`。
- 预取：`AM_BUILTIN_PREFETCH(...)`。
- 在分配器和运行时热路径避免额外分配/日志。
- 缩小锁作用域，保持既有锁顺序。

### 7.7 ammalloc 子系统特殊约束
- **禁止使用会触发堆分配的 STL 容器**（如 `std::vector`、`std::string`、`std::map`）以及常规堆 `new`/`delete`——会触发 malloc 递归。
- 使用定长栈数组、嵌入式链表或自定义 `ObjectPool`。
- 允许使用 placement new 在预留存储、页分配器返回的内存或静态缓冲区上原地构造对象；这不属于常规堆分配。
- 热路径、跨线程共享且容易发生 false sharing 的结构优先 `alignas(CACHE_LINE_SIZE)`；若对象天然按页对齐或已有更合适的布局，可按现有实现处理。
- `std::atomic` 必须显式指定内存序，禁止默认 `seq_cst`。
- 详见 `ammalloc/GEMINI.md`。

## 8. 额外 AI 指南
- `.cursorrules`：不存在
- `.cursor/rules/`：不存在
- `.github/copilot-instructions.md`：不存在
- `GEMINI.md`：项目架构蓝图与 C++20 编码指引（架构/性能敏感改动前必读）。
- `ammalloc/GEMINI.md`：分配器子系统的 AI 上下文指南。

## 9. 推荐代理工作流
1. 修改前先阅读相关 `CMakeLists.txt` 和 `GEMINI.md`（若涉及架构/性能）。
2. 进行最小化、风格一致的改动。
3. 构建最小受影响目标（`--target <name>`）。
4. 先跑单个聚焦测试（`--gtest_filter=Suite.Case`），再扩大范围。
5. 分配器/性能相关改动需跑聚焦 benchmark。
6. 改动构建选项/宏时，至少验证一个非默认配置（如 TSAN）。
7. 按 `.clang-format` 格式化改动文件。
8. 在汇报中明确风险：正确性、并发、内存、性能。
