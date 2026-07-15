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
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# 显式配置
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON \
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
./build/tests/unit/aethermind_unit_tests --gtest_filter=TensorRandomNew.*

# 单个测试用例（推荐快速回路）
./build/tests/unit/aethermind_unit_tests --gtest_filter=TensorRandomNew.UniformTensorDeterministicWithSameSeed

# 调试用参数
./build/tests/unit/aethermind_unit_tests --gtest_filter=Device.* --gtest_break_on_failure
./build/tests/unit/aethermind_unit_tests --gtest_filter=TensorRandomNew.* --gtest_repeat=10
```

## 5. 性能基准命令
```bash
./build/tests/benchmark/aethermind_benchmark
./build/tests/benchmark/aethermind_benchmark --benchmark_filter=BM_Malloc_Churn
```

## 6. Lint / 格式化
代码格式以根目录 `.clang-format` 为准（基于 LLVM，缩进 4 空格，ColumnLimit 0）。
```bash
# 格式检查（仅收集工作树改动文件，范围：include/ src/ ammalloc/ tests/）
{
  git diff --cached --name-only --diff-filter=d
  git diff --name-only --diff-filter=d
  git ls-files --others --exclude-standard
} | grep -E '\.(c|cc|cpp|cxx|h|hpp|hxx)$' | grep -E '^(include/|src/|ammalloc/|tests/)' | sort -u | xargs -r clang-format -n --Werror
# 自动格式化（同上收集逻辑，写入模式）
{
  git diff --cached --name-only --diff-filter=d
  git diff --name-only --diff-filter=d
  git ls-files --others --exclude-standard
} | grep -E '\.(c|cc|cpp|cxx|h|hpp|hxx)$' | grep -E '^(include/|src/|ammalloc/|tests/)' | sort -u | xargs -r clang-format -i
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

## 9. 测试编写规范

Tests use GoogleTest and live under `tests/unit/` (see §4 for run commands).

- suite and test names use PascalCase; no `DISABLED_`, no commented-out `GTEST_SKIP()`
- one behavior per test; prefer `EXPECT_*`, use `ASSERT_*` only when continuing would crash
- death tests match `"Check failed"` (the `AM_CHECK` abort message) and guard debug-only checks with `#ifndef NDEBUG`
- shared helpers go in `tests/unit/.../test_*_helpers.h` as `inline` functions, not in `include/`
- reset global singletons in `TEST_F` `SetUp`/`TearDown`; seed RNGs explicitly

See full test writing rules:
- `docs/guides/test_writing_guidelines.md`

## 10. 额外 AI 指南
- `.cursorrules`：不存在
- `.cursor/rules/`：不存在
- `.github/copilot-instructions.md`：不存在
- `ammalloc/AGENTS.md`：分配器子系统的 AI 上下文指南
- `docs/products/aethermind_prd.md`：**Phase 1 产品需求与验收标准**（产品范围、公开 API、架构或验收标准相关工作时必读）

## 11. 指令优先级与作用域

多级指令共存时，按以下优先级裁决：

1. **用户当前明确指令**：最高优先级，覆盖一切与之冲突的静态文档。
2. **已验证的仓库事实**（代码行为、CMake 配置、已实现的接口等）：次高优先级。
3. **根目录 AGENTS.md**：仓库级默认规则，对整个仓库生效。
4. **嵌套 AGENTS.md**（如 `ammalloc/AGENTS.md`）：在其所在目录子树内添加局部规则与约束；与根 AGENTS.md 冲突时以根 AGENTS.md 为准，除非用户明确指示使用嵌套规则。

嵌套 AGENTS.md 中的审批门（如生成代码前需先审核）仅在其所在目录子树内生效，不影响同仓库其他模块。

## 12. Agent Memory System 执行约束

使用 Agent Memory 系统恢复/继续已有工作流时，必须遵守以下约束。本节不适用于未请求记忆恢复的普通新任务。

### 启动核心原则：先加载最小必要记忆，再渐进式升级

**规范地位**：本节是 Agent Memory System 的**唯一规范性启动契约**。其他文档只可引用本节，不得重新定义默认启动顺序。

#### 允许的引导操作（Resume Gate 前）

在通过 Resume Gate 之前，只允许以下只读引导操作：

1. 解析工作流类型（按下面定义的优先级）
2. 列出候选 handoff 路径
3. 读取 AGENTS.md 和 `docs/agent/memory/project.md`
4. 扫描候选 handoff 文件的 YAML frontmatter（仅 frontmatter，不读取正文）
5. 根据 frontmatter 信息筛选后读取唯一选中 handoff 的正文
6. 按需读取许可的记忆文档（module.md、submodule.md 等）

**禁止行为**：在通过 Resume Gate 并获得用户确认前，不得执行以下操作：

- 与恢复无关的代码扫描
- 构建、测试、编辑、写入
- 任何外部副作用操作

#### 工作流类型解析（按优先级）

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

#### 默认启动路径

**Token 预算约束**：默认启动加载量 **≤ 15,000 tokens**。超出此预算时，默认启动降级为仅 AGENTS.md 加用户手动指定的文件。

**项目级工作默认启动路径**：
```
AGENTS.md → docs/agent/memory/project.md → docs/agent/handoff/workstreams/project__<slug>/<selected-active-handoff>.md
```
项目级工作**必须跳过** `docs/agent/memory/modules/<module>/module.md` 和 `docs/agent/memory/modules/<module>/submodules/<submodule>.md`，除非因模块边界、所有权/生命周期、线程安全或性能约束需要按需读取这些模块记忆，并在 Resume Gate 中记录。

**模块工作默认启动路径**：
```
AGENTS.md → docs/agent/memory/project.md → docs/agent/handoff/workstreams/<module>__<submodule-or-none>/<selected-active-handoff>.md
```

若 `docs/agent/memory/project.md` 缺失或无法读取，则 `resume_status` 为 `blocked`，且不加载任何 handoff 正文。

**Handoff 选择规则**：
1. 扫描目录下所有候选 handoff 文件的 YAML frontmatter，仅作为元数据收集
2. 筛选 `status: active` 的条目
3. 按 `created_at` 降序排序；若相同则按文件名字典序 tie-break
4. 仅读取排序后第一个 handoff 的完整正文
5. 同一 workstream 只读取一个 handoff 正文（frontmatter 扫描不计为读取多个 handoff 正文）

**默认启动禁止加载**：
- `docs/agent/memory/README.md` — 操作手册，仅按需升级时读取
- `docs/agent/memory/modules/**/module.md` — 项目级工作时默认跳过
- `docs/agent/tests/**` — 测试文档，不参与恢复
- `docs/guides/**` — 指南文档，按需引用
- 历史 handoff（`status: superseded` 或 `status: closed`）— 只读取 active

#### 渐进式升级读取

仅在以下情况触发：
- 目标 workstream 没有可用的 active handoff
- handoff 缺少恢复所需信息，或未明确可用于低上下文恢复（例如 `bootstrap_ready` 缺失或为 `false`）
- 任务触及模块边界、所有权/生命周期、线程安全、性能敏感约束
- 需要写回 memory、处理 handoff/frontmatter/schema、或排查冲突/兼容问题
- 需要理解模块内部不变量，且 handoff 无法覆盖

**渐进式升级读取规则**：
- `docs/agent/memory/README.md`：仅按需升级时读取，用于操作细则、schema、兼容与写回规则
- `docs/agent/memory/modules/<module>/module.md`：模块工作且需要模块级约束时读取；项目工作触及该模块边界/所有权/生命周期/并发/性能时也可按需读取
- `docs/agent/memory/modules/<module>/submodules/<submodule>.md`：需要子模块级不变量时读取；项目工作触及相关边界/所有权/生命周期/并发/性能时也可按需读取
- `docs/agent/memory_system.md`：仅作架构参考，不属于默认启动路径

#### Handoff 路径规则

- 模板文件：`docs/agent/prompts/handoff_template.md`（只读参考）
- 存储目录：`docs/agent/handoff/workstreams/<workstream_key>/`（实际 handoff 位置）

#### Resume Gate 检查点

在引导操作完成后、任何非恢复/业务工具操作前，必须通过以下检查点：

```markdown
## Resume Gate
- [x] 工作流类型：[模块级工作 | 项目级工作]
- [x] resolved_workstream_key：[<module>__<submodule-or-none> | project__<slug>]
- [x] 默认启动已加载文件：[列出完整路径]
- [x] 按需升级读取：[无 | 列出额外读取文件]
- [x] 跳过/缺失项：[项目级时列出跳过的 module/submodule，或列出按需读取的模块]
- [x] resume_status: complete | partial | blocked
- [x] handoff 路径：docs/agent/handoff/workstreams/<workstream_key>/（非 docs/agent/prompts/handoff_template.md）
```

#### 用户确认

加载完成后，输出已加载文件清单，并显式询问用户："记忆已加载，是否执行[推荐操作]？"

**关键区分**：
- 第一轮对话中的"继续"表示选择/恢复 workstream，不等于授权执行业务操作
- 只有在 Resume Gate 中明确询问后，用户明确说"继续"、"执行"或"是"时，才授权执行推荐操作。此确认在第二回合或之后发生，而非第一回合

#### 为什么重要

- 默认启动路径必须尽量小，避免新对话一开始就加载过量上下文
- handoff 负责当前状态，memory/README 负责稳定规则；二者不能互相替代
- 跨机器恢复时，必须先恢复上下文，再精准继续

## 13. 推荐代理工作流
1. 修改前先阅读相关 `CMakeLists.txt` 和 `AGENTS.md`；涉及产品范围、公开 API、架构或验收标准时，还需阅读 `docs/products/aethermind_prd.md`。
2. 进行最小化、风格一致的改动。
3. 构建最小受影响目标（`--target <name>`）。
4. 先跑单个聚焦测试（`--gtest_filter=Suite.Case`），再扩大范围。
5. 仅性能敏感改动需跑聚焦 benchmark。
6. 改动构建选项/宏时，验证受影响的非默认配置；涉及并发/同步或相关构建选项时至少验证 TSAN。
7. 按 `.clang-format` 格式化改动文件。
8. 在汇报中明确风险：正确性、并发、内存、性能。
