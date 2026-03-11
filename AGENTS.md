> **[AI 助手指令]**
> 本文件为本仓库的智能编码代理执行指南。若文档与当前代码、CMake 或用户指令冲突，应以已验证的仓库事实和用户指令为准。
> 你是资深大模型推理引擎架构师，在 `AetherMind` 项目中生成、重构或审查代码时，**必须严格遵守**本文档定义的架构约束、C++20 现代编程实践及性能优化指南。

## 1. 项目简介
- 项目名称：AetherMind
- 项目目标：分阶段构建大模型推理引擎
  - Phase 1（当前）：CPU 嵌入式推理运行时，支持 Llama 家族模型，C/C++ API
  - Phase 2+：服务化/分布式推理引擎（见 docs/products/aethermind_prd.md 附录）
- 语言：C++20（`CMAKE_CXX_STANDARD 20`）
- 构建系统：CMake >= 3.28
- 核心库目标：`AetherMind`（shared）
- 内存池目标：`ammalloc`（static）
- 单元测试目标：`aethermind_unit_tests`（GoogleTest，系统安装）
- 性能基准目标：`aethermind_benchmark`（Google Benchmark，FetchContent）

## 2. 目录结构
- `include/`：核心公共头文件
- `src/`：核心实现
- `ammalloc/include/ammalloc/`：内存池公共/内部头文件
- `ammalloc/src/`：内存池实现
- `tests/unit/`：gtest 单元测试
- `tests/benchmark/`：benchmark 性能基准测试
- `cmake/`：CMake 辅助模块
- `docs/`：架构与开发文档
- `tools/`：辅助工具
- `3rdparty/`：第三方代码（非必要不要修改）

## 3. 构建命令
```bash
# 默认配置（单元测试和 benchmark 默认 ON）
cmake -S . -B build

# 显式配置
cmake -S . -B build -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON \
  -DUSE_LIBBACKTRACE=ON -DBACKTRACE_ON_SEGFAULT=ON

# 全量构建
cmake --build build -j

# 按目标构建（推荐）
cmake --build build --target AetherMind -j
cmake --build build --target ammalloc -j
cmake --build build --target aethermind_unit_tests -j
cmake --build build --target aethermind_benchmark -j

# TSAN 变体
cmake -S . -B build-tsan -DENABLE_TSAN=ON -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF
cmake --build build-tsan --target aethermind_unit_tests -j
```

## 4. 测试命令

### 基本规则：
- run the narrowest relevant target first
- do not run the full test suite unless needed
- if `compile_commands.json` exists, prefer tools that use it

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

## 5. 性能基准命令
```bash
./build/tests/benchmark/aethermind_benchmark
./build/tests/benchmark/aethermind_benchmark --benchmark_filter=BM_Malloc_Churn
```

## 6. Lint / 格式化
代码格式以根目录 `.clang-format` 为准（基于 LLVM，缩进 4 空格，ColumnLimit 0）。
```bash
# 格式检查
clang-format -n --Werror $(git ls-files '*.h' '*.hpp' '*.cpp')
# 自动格式化
clang-format -i $(git ls-files '*.h' '*.hpp' '*.cpp')
```
若本地可用，可基于 `build/compile_commands.json` 运行 `clang-tidy`。

## 7. 代码风格规范

- prefer modern C++ adopted by the project (C++20)
- prefer RAII and standard library facilities
- avoid raw owning pointers
- preserve const-correctness
- make ownership and lifetime explicit
- avoid hidden O(N^2) behavior in hot paths
- prefer simple control flow and early returns
- prefer small, focused functions

See full coding style:

- `docs/guides/cpp_coding_style_guidelines.md`

## 8. 注释风格规范
Write comments only when they add real value.

Comment:

- intent
- assumptions
- invariants
- ownership and lifetime
- thread-safety expectations
- non-obvious tradeoffs
- performance-sensitive choices

Do not:

- comment obvious code
- leave commented-out code
- keep stale comments

For non-trivial public APIs in headers, use documentation comments.
For implementation details, use `//`.

See full comment rules:
- `docs/guides/cpp_comment_guidelines.md`

## 9. 额外 AI 指南
- `.cursorrules`：不存在
- `.cursor/rules/`：不存在
- `.github/copilot-instructions.md`：不存在
- `GEMINI.md`：项目架构蓝图与 C++20 编码指引（架构/性能敏感改动前必读）
- `ammalloc/GEMINI.md`：分配器子系统的 AI 上下文指南
- `docs/products/aethermind_prd.md`：**Phase 1 产品需求与验收标准（必读）**

## 10. Agent Memory System 执行约束

使用记忆系统时必须遵守以下约束：

### 核心原则：先加载记忆，后执行操作

**禁止行为（❌）**：
1. 先扫描代码/编译/测试，再加载记忆
2. 假设"继续"等于立即执行默认操作（如编译+测试）
3. 跳过记忆加载直接执行操作
4. **加载记忆后未经用户确认就执行工具操作**（如扫描代码、编译、测试、文件修改等）

**必须行为（✅）**：
1. 严格按固定顺序加载记忆：  
   `AGENTS.md` → `docs/agent/memory/README.md` → `project.md` → `module.md` → `submodule.md` → `handoff`
2. 输出"已加载文件"清单，明确列出实际加载了哪些文件
3. 根据 memory/handoff 中的"推荐下一步"决定执行什么操作
4. **加载完成后，必须显式询问用户**："记忆已加载，是否执行[推荐操作]？"，**等待用户明确说"继续"、"执行"或"是"后，才能执行任何工具操作**

**为什么重要**：
- 如果 handoff 显示"阻塞点是设计决策"，不应编译，而应先解决设计问题
- 如果 memory 规定"当前阶段不跑全量测试"，不应跑测试
- 跨机器恢复时，必须先恢复上下文，再精准继续

## 11. 推荐代理工作流
1. 修改前先阅读相关 `CMakeLists.txt` 和 `GEMINI.md`（若涉及架构/性能）。
2. 进行最小化、风格一致的改动。
3. 构建最小受影响目标（`--target <name>`）。
4. 先跑单个聚焦测试（`--gtest_filter=Suite.Case`），再扩大范围。
5. 内存池/性能相关改动需跑聚焦 benchmark。
6. 改动构建选项/宏时，至少验证一个非默认配置（如 TSAN）。
7. 按 `.clang-format` 格式化改动文件。
8. 在汇报中明确风险：正确性、并发、内存、性能。
