# AGENTS.md

本文件为本仓库的智能编码代理执行指南。
除非用户给出冲突指令，否则默认遵循本指南。

## 1. 项目事实
- 语言：C++20（`CMAKE_CXX_STANDARD 20`）
- 构建系统：CMake >= 3.28
- 核心库目标：`AetherMind`（shared）
- 分配器目标：`ammalloc`（static）
- 单测目标：`aethermind_unit_tests`（GoogleTest）
- 基准目标：`aethermind_benchmark`（google benchmark）
- CMake 拉取依赖：`spdlog`、`google_benchmark`

## 2. 仓库结构
- `include/`：核心公共头文件
- `src/`：核心实现
- `ammalloc/include/ammalloc/`：分配器公共/内部头文件
- `ammalloc/src/`：分配器实现
- `tests/unit/`：gtest 单元测试
- `tests/benchmark/`：benchmark 源码
- `cmake/`：CMake 辅助模块
- `docs/`：架构与开发文档
- `3rdparty/`：第三方代码（非必要不要修改）

## 3. 构建命令
- 默认配置：
  - `cmake -S . -B build`
- 显式配置：
  - `cmake -S . -B build -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON -DUSE_LIBBACKTRACE=ON -DBACKTRACE_ON_SEGFAULT=ON`
- 全量构建：
  - `cmake --build build -j`
- 按目标构建：
  - `cmake --build build --target AetherMind -j`
  - `cmake --build build --target ammalloc -j`
  - `cmake --build build --target aethermind_unit_tests -j`
  - `cmake --build build --target aethermind_benchmark -j`
- TSAN 变体：
  - `cmake -S . -B build-tsan -DENABLE_TSAN=ON -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF`
  - `cmake --build build-tsan --target aethermind_unit_tests -j`

## 4. 测试命令（含单测单例）
- 注意：顶层 CMake 未接入 `ctest`（没有 `enable_testing()` / `add_test()`）。
- 运行全部单元测试：
  - `./build/tests/unit/aethermind_unit_tests`
- 运行单个测试套件：
  - `./build/tests/unit/aethermind_unit_tests --gtest_filter=Tensor.*`
- 运行单个测试用例（推荐快速回路）：
  - `./build/tests/unit/aethermind_unit_tests --gtest_filter=Tensor.random`
- 常用 gtest 参数：
  - `./build/tests/unit/aethermind_unit_tests --gtest_filter=Device.* --gtest_break_on_failure`
  - `./build/tests/unit/aethermind_unit_tests --gtest_filter=Tensor.* --gtest_repeat=10`
- 可选 Torch 版本单测：
  - `cmake -S . -B build-torch -DBUILD_TESTS=ON -DBUILD_WITH_TORCH=ON`
  - `cmake --build build-torch --target aethermind_unit_tests -j`
  - `./build-torch/tests/unit/aethermind_unit_tests`

## 5. 基准命令
- 运行全部 benchmark：
  - `./build/tests/benchmark/aethermind_benchmark`
- 运行单个 benchmark：
  - `./build/tests/benchmark/aethermind_benchmark --benchmark_filter=BM_Malloc_Churn`

## 6. Lint / 格式化命令
- 顶层暂无专用 lint target。
- 代码格式以根目录 `.clang-format` 为准。
- 格式检查：
  - `clang-format -n --Werror $(git ls-files '*.h' '*.cpp')`
- 自动格式化：
  - `clang-format -i $(git ls-files '*.h' '*.cpp')`
- 若本地可用，可基于 `build/compile_commands.json` 运行 `clang-tidy`。

## 7. 代码风格规范

### 7.1 头文件与文件组织
- 使用 include guard（`AETHERMIND_<NAME>_H`），不要用 `#pragma once`。
- 文件名使用 `snake_case`。
- 包含顺序：先项目头，再标准库头。
- 公共 API 头文件放在 `include/` 或 `ammalloc/include/ammalloc/`。
- 实现细节放在 `src/` 与 `ammalloc/src/`。
- 除非任务明确要求，不修改 `3rdparty/`。

### 7.2 格式与排版
- 严格遵循根目录 `.clang-format`。
- 缩进 4 空格，不使用 Tab。
- 指针对齐为左对齐（`Type* ptr`）。
- 空行保持克制（通常不超过连续 1-2 行）。
- 大括号与控制流风格与所在文件保持一致。

### 7.3 类型与 API 设计
- 位宽敏感场景优先固定宽度整数（`int64_t`、`uint32_t` 等）。
- 大小/计数/索引运算优先 `size_t`。
- 语义明确时使用 `const`、`noexcept`、`explicit`。
- 复用现有项目类型/别名（如 `String`、`ObjectPtr`、`DataType`）。
- 不应忽略的返回值使用 `AM_NODISCARD`。
- 不要引入与现有模式冲突的所有权模型。

### 7.4 命名约定（按现状观察）
- 类型/类：`PascalCase`（如 `Tensor`、`RuntimeConfig`）。
- 函数：核心 API 多为 `snake_case`；分配器内部有混用。
- 规则：以你正在修改的文件/子系统现有风格为准。
- 枚举成员常用 `kPrefix`（如 `kCPU`、`kCUDA`）。
- 宏/常量使用大写下划线（如 `AM_ALWAYS_INLINE`、`MAX_PAGE_NUM`）。
- 命名空间：`aethermind` 与 `aethermind::details`。

### 7.5 错误处理与断言
- 优先使用项目错误路径：`AM_THROW(...) << message`。
- 使用 `AM_CHECK(...)` 进行不变量与前置条件校验。
- 不可能状态应快速失败；必要时使用 `AM_UNREACHABLE()`。
- 除明确防御性清理路径外，不要静默吞错。

### 7.6 性能与并发
- 保留热路径宏：`AM_LIKELY`、`AM_UNLIKELY`、`AM_ALWAYS_INLINE`、`AM_NOINLINE`。
- 在分配器和运行时热路径避免额外分配/日志。
- 缩小锁作用域并保持既有锁顺序假设。
- 尊重对齐、页大小和 cacheline 敏感结构约束。
- 修改原子、TLS、跨线程所有权时务必谨慎。

### 7.7 代码变更后的测试要求
- 先构建最小受影响目标，再扩大范围。
- 至少运行一个带 `--gtest_filter` 的聚焦单测。
- 分配器/性能相关改动需运行聚焦 benchmark。
- 若改动构建选项/宏，至少验证一个非默认配置（如 TSAN）。

## 8. Cursor / Copilot 规则文件
- 已检查用户指定路径：
  - `.cursorrules`：不存在
  - `.cursor/rules/`：不存在
  - `.github/copilot-instructions.md`：不存在
- 额外 AI 指南：
  - 存在 `GEMINI.md`，包含 AI 助手的架构/编码指引。
  - 进行架构或性能敏感改动时应先阅读并遵循。

## 9. 推荐代理工作流
1. 修改前先阅读相关 `CMakeLists.txt`。
2. 进行最小化、风格一致的改动。
3. 构建受影响目标。
4. 先跑单个聚焦测试，再扩大测试范围。
5. 涉及性能路径时补跑聚焦 benchmark。
6. 按 `.clang-format` 格式化改动文件。
7. 在汇报中明确风险：正确性、并发、内存、性能。
