> **[AI 助手指令]**
> 本文件为本仓库的智能编码代理执行指南。若文档与当前代码、CMake 或用户指令冲突，应以已验证的仓库事实和用户指令为准。
> 你是资深大模型推理引擎架构师，在 `AetherMind` 项目中生成、重构或审查代码时，**必须严格遵守**本文档定义的架构约束、C++20 现代编程实践及性能优化指南。

## 1. 项目简介
- 项目名称：AetherMind
- 项目目标：分阶段构建大模型推理引擎
  - Phase 1（当前）：桌面/服务器 CPU 本地推理运行时，支持 Llama 家族模型，C/C++ API
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
- `ammalloc/AGENTS.md`：分配器子系统的 AI 上下文指南
- `docs/products/aethermind_prd.md`：**Phase 1 产品需求与验收标准（必读）**

## 10. Agent Memory System 执行约束

使用记忆系统时必须遵守以下约束：

### 启动核心原则：先加载最小必要记忆，再渐进式升级读取其他所需文件

**规范地位**：本节是 Agent Memory System 的**唯一规范性启动契约**。其他文档只可引用本节，不得重新定义默认启动顺序。

#### **禁止行为（❌）**：

1. 先扫描代码/编译/测试，再加载启动所需记忆
2. 把"继续"默认解释成可立即执行工具操作
3. 跳过工作流解析与 Resume Gate，直接修改文件
4. **加载完成后未经用户确认就执行工具操作**（如扫描代码、编译、测试、文件修改等）
5. 把 `docs/agent/prompts/handoff_template.md` 当作真实 handoff 读取
6. **读取多个 handoff 文件**（同一 workstream 只读一个 `status: active`，忽略 `superseded`/`closed`）

#### **必须行为（✅）**：

1. **先解析工作流类型**（按优先级）：

   **明确指令**（用户直接说明）：
   - 用户说"项目级"或 `project__<slug>` → **项目级工作**
   - 用户说"模块"或 `<module>__<submodule-or-none>` → **模块工作**

   **默认推断**（用户未明确说明时）：
   - 如果目标匹配 `project__<slug>` 格式（如 `project__docs-reorg`）→ **项目级工作**
   - 如果目标匹配模块名（在 `docs/agent/memory/modules/` 中存在）→ **模块工作**
   - 如果目标匹配 `<module>/<submodule>` 路径格式 → **模块工作**

   **歧义处理**（无法唯一确定时）：
   - **必须询问用户**："请明确指定工作流类型：1) 项目级 2) 模块级"
   - **禁止猜测**：不得在歧义时自动选择

2. **默认启动路径只加载最小必要集合**：

   **Token预算约束**：默认启动加载量 **≤ 15,000 tokens**（约6,000字符）。超出此预算时，必须降级为"仅AGENTS.md + 用户手动指定文件"。

   **项目级工作默认启动路径**：
   ```
   AGENTS.md → docs/agent/memory/project.md → docs/agent/handoff/workstreams/project__<slug>/
   ```
   ⚠️ **硬性约束**：项目级工作**必须跳过** `docs/agent/memory/modules/<module>/module.md` 和 `docs/agent/memory/modules/<module>/submodules/<submodule>.md`

   **模块工作默认启动路径**：
   
   ```
   AGENTS.md → docs/agent/memory/project.md → docs/agent/handoff/workstreams/<module>__<submodule-or-none>/
   ```
   默认启动只读取 `status: active` 的 handoff；按 `created_at` 降序排序，若相同则按文件名字典序 tie-break。

   **默认启动禁止加载**（违反会导致token爆炸）：
   - ❌ `docs/agent/memory/README.md` — 操作手册，**仅在按需升级时读取**
   - ❌ `docs/agent/memory/modules/**/module.md` — 项目级工作时跳过
   - ❌ `docs/agent/tests/**` — 测试文档，不参与恢复
   - ❌ `docs/guides/**` — 指南文档，按需引用
   - ❌ 历史 handoff（`status: superseded` 或 `status: closed`）— 只读 active

3. **渐进式升级读取**（仅在以下情况触发）：
   
   - 目标 workstream 没有可用的 active handoff
   - handoff 缺少恢复所需信息，或未明确可用于低上下文恢复（例如 `bootstrap_ready` 缺失或为 `false`）
   - 任务触及模块边界、所有权/生命周期、线程安全、性能敏感约束
   - 需要写回 memory、处理 handoff/frontmatter/schema、或排查冲突/兼容问题
   - 需要理解模块内部不变量，且 handoff 无法覆盖
   
   **渐进式升级读取规则**：
   
   - `docs/agent/memory/README.md`：**只在按需升级时读取**，用于操作细则、schema、兼容与写回规则
   - `docs/agent/memory/modules/<module>/module.md`：模块工作且需要模块级约束时读取
   - `docs/agent/memory/modules/<module>/submodules/<submodule>.md`：需要子模块级不变量时读取
   - `docs/agent/memory_system.md`：仅作架构参考，不属于默认启动路径

4. **Handoff 路径规则**：
   
   - 模板文件：`docs/agent/prompts/handoff_template.md`（只读参考）
   - 存储目录：`docs/agent/handoff/workstreams/<workstream_key>/`（实际 handoff 位置）

5. **执行 Resume Gate 检查点**（任何工具操作前必须完成）：
   ```markdown
   ## Resume Gate
   - [x] 工作流类型：[模块级工作 | 项目级工作]
   - [x] resolved_workstream_key：[<module>__<submodule-or-none> | project__<slug>]
   - [x] 默认启动已加载文件：[列出完整路径]
   - [x] 按需升级读取：[无 | 列出额外读取文件]
   - [x] 跳过/缺失项：[项目级时列出跳过的 module/submodule，或列出缺失项]
   - [x] resume_status: complete | partial | blocked
   - [x] handoff 路径：docs/agent/handoff/workstreams/<workstream_key>/（非 docs/agent/prompts/handoff_template.md）
   ```

6. 输出"已加载文件"清单，明确列出实际加载了哪些文件

7. **加载完成后，必须显式询问用户**："记忆已加载，是否执行[推荐操作]？"，**等待用户明确说"继续"、"执行"或"是"后，才能执行任何工具操作**

**为什么重要**：
- 默认启动路径必须尽量小，避免新对话一开始就加载过量上下文
- handoff 负责当前状态，memory/README 负责稳定规则；二者不能互相替代
- 跨机器恢复时，必须先恢复上下文，再精准继续

## 11. 推荐代理工作流
1. 修改前先阅读相关 `CMakeLists.txt` 和 `AGENTS.md`（若涉及架构/性能）。
2. 进行最小化、风格一致的改动。
3. 构建最小受影响目标（`--target <name>`）。
4. 先跑单个聚焦测试（`--gtest_filter=Suite.Case`），再扩大范围。
5. 内存池/性能相关改动需跑聚焦 benchmark。
6. 改动构建选项/宏时，至少验证一个非默认配置（如 TSAN）。
7. 按 `.clang-format` 格式化改动文件。
8. 在汇报中明确风险：正确性、并发、内存、性能。
