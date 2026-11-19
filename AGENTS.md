> **[AI 助手指令]**
> 本文件为本仓库的智能编码代理执行指南。若文档与当前代码、CMake 或用户指令冲突，应以已验证的仓库事实和用户指令为准。
> 你是资深大模型推理引擎架构师，在 `AetherMind` 项目中生成、重构或审查代码时，**必须严格遵守**本文档定义的架构约束、C++20 现代编程实践及性能优化指南。

## 1. 项目简介
- 项目名称：AetherMind
- 项目目标：构建可独立交付的桌面/服务器 CPU 本地大模型推理引擎，支持 Llama 家族模型与 C/C++ API
  - 当前产品边界：CPU-only、单进程、同步、单请求、Token IDs I/O、静态 KV Cache
  - 长期演进方向：服务化、异构计算与分布式推理（见 docs/product.md 附录），不构成当前交付承诺
  - 范围约束：除非用户明确要求，不要预先实现当前产品边界之外的服务化、调度、GPU 或分布式能力。
- 语言：C++20（`CMAKE_CXX_STANDARD 20`）
- 构建系统：CMake >= 3.28
- 核心库目标：`AetherMind`（shared）
- 独立推理工具目标：`aethermind_infer`（`BUILD_INFERENCE_TOOL=ON`，默认 OFF；不依赖 tests/benchmark）
- 单元测试目标：`aethermind_unit_tests`（GoogleTest v1.18.0，`BUILD_TESTS=ON` 时默认通过 FetchContent 获取；复用父工程已提供的 `GTest::gtest` / `GTest::gtest_main`，离线源码可通过 `FETCHCONTENT_SOURCE_DIR_GOOGLETEST` 指定）
- 性能基准目标：`aethermind_benchmark`（Google Benchmark v1.9.5，`BUILD_BENCHMARKS=ON` 时默认通过 FetchContent 获取；复用父工程已提供的 `benchmark::benchmark` / `benchmark::benchmark_main`）
- 共享依赖源码：首次 configure 按版本 tag 获取源码到 `.deps/googletest`、`.deps/benchmark`、`.deps/spdlog`；通过进程锁、同目录 staging 和校验后发布供各 build 复用，已有 checkout 不自动更新，版本不符或有修改时拒绝覆盖。编译产物和生成文件归各自 build 的 `_deps/`。显式 `FETCHCONTENT_SOURCE_DIR_*`、父工程 targets、先前声明与 dependency providers 保留优先级；离线、超时和显式升级规则见 README。

## 2. 目录结构
- `include/`：核心公共头文件
- `src/`：核心实现
- `apps/`：独立应用源码，仅消费库公共接口；`apps/inference/` 提供可选的 `aethermind_infer` CLI、同步 runner 与私有输入/配置 helper；builder 注入入口仅在测试编译中启用
- `tests/unit/`：gtest 单元测试
- `tests/benchmark/`：benchmark 性能基准测试
- `cmake/`：CMake 辅助模块
- `docs/`：架构与开发文档
- `tools/`：开发与维护辅助工具
- `3rdparty/`：第三方代码（非必要不要修改）

### 2.1 源码模块所有权（严格边界）

模块职责、所有权和依赖规则见 [架构文档的源码模块所有权](docs/architecture.md#源码模块所有权)。修改跨模块接口、所有权或依赖时，按该节检查。

**权重准备文件约定**：model/weight 的 `weight_binding_resolver.h` 定义角色到 raw weight 的解析入口，`weight_packing_request.h` 定义请求数据，`packed_weight_collection.h` 定义 `WeightArtifactKey`、`PackedWeightCollection` 与自由函数构建入口 `PrepackWeightRequests`；该头仅前向声明 Allocator、Backend 与 WeightPackingRequest，批量打包实现归 `packed_weight_collection.cpp`。backend 的 `packing_recipe.h` 与 `packed_weight.h` 分别定义布局身份与产物接口；CPU identity/B-panel 共用 `backend/cpu/cpu_packed_weight_layout.h` 布局契约，尺寸公式实现归对应 `.cpp`，独立于 backend 打包实现，kernel 消费布局契约与 CPU 层的 `backend/cpu/packed_weight_validation.h`。请求 recipe 查询与合并定义在 `src/inference/executable_model.cpp`，私有声明与 Decode helper 统一放在 `src/inference/inference_internal.h`，由 `ExecutableModel::Prepare` 编排；调用方按职责包含对应头文件。

**权重命名约定**：`packed_weight_*` / `PackedWeight*` 表示已打包产物及其视图、存储和校验；`weight_packing_*` / `WeightPacking*` 表示打包过程、请求及其构建/解析，文件名应保留完整词组（如 `compiler/weight_packing_request_builder.h`）。`Prepack` / `Prepacker` 限定模型准备期、执行前的打包时机，归入过程组；具体布局仍由 recipe 指定。

**打包布局身份约定**：`PackingRecipe::layout` 与 `PackedWeightView::recipe_layout` 共用 backend 纯数据枚举 `PackingLayout`，`kNone` 表示未指定。生产 packed descriptor/request 必须携带已知非空布局，未知枚举值须在边界校验；布局 ID 对应不可变的物理字节格式，改变 tile、panel 顺序或 padding 须新增 ID。布局名称使用 dtype、blocking 及必要的 panel 顺序或 padding 等具体格式特征区分，不使用 V1、V2 等通用版本序号。`ToString(PackingLayout)` 仅用于诊断，不参与身份匹配，alignment 与 ISA eligibility 仍按各自契约处理。

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
cmake --build build --target aethermind_unit_tests -j
cmake --build build --target aethermind_benchmark -j

# TSAN 变体
cmake -S . -B build-tsan -DAETHERMIND_SANITIZER=tsan -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF
cmake --build build-tsan --target aethermind_unit_tests -j
```

## 4. 测试命令

### 基本规则：
- run the narrowest relevant target first
- do not run the full test suite unless needed
- if `compile_commands.json` exists, prefer tools that use it

```bash
# 全部单元测试（先准备 HF fixtures 并构建独立 CLI，见 README）
AETHERMIND_INFER_BINARY="$PWD/build-infer/apps/inference/aethermind_infer" \
  ./build/tests/unit/aethermind_unit_tests

# 单个测试套件
./build/tests/unit/aethermind_unit_tests --gtest_filter=TensorRandomNew.*

# 单个测试用例（推荐快速回路）
./build/tests/unit/aethermind_unit_tests --gtest_filter=TensorRandomNew.UniformTensorDeterministicWithSameSeed

# 调试用参数
./build/tests/unit/aethermind_unit_tests --gtest_filter=Device.* --gtest_break_on_failure
./build/tests/unit/aethermind_unit_tests --gtest_filter=TensorRandomNew.* --gtest_repeat=10
```

`CliMatrix/InferenceCliIntegration.*` 为 Linux CLI/直接 Session 集成门禁，须提供上述绝对路径；缺少 CLI、fixture 或平台条件会显式失败。该环境变量仅供测试消费，不是工具运行配置。其他聚焦单测不需要 CLI；HF 资产准备与独立工具构建见根 README 和 `tests/fixtures/inference_admission/README.md`。

## 5. 性能基准命令
```bash
./build/tests/benchmark/aethermind_benchmark
./build/tests/benchmark/aethermind_benchmark --benchmark_filter=BM_Malloc_Churn
```

## 6. Lint / 格式化
代码格式以根目录 `.clang-format` 为准（基于 LLVM，缩进 4 空格，ColumnLimit 0）。
```bash
# 格式检查（仅收集工作树改动文件，范围：include/ src/ apps/ tests/）
{
  git diff --cached --name-only --diff-filter=d
  git diff --name-only --diff-filter=d
  git ls-files --others --exclude-standard
} | grep -E '\.(c|cc|cpp|cxx|h|hpp|hxx)$' | grep -E '^(include/|src/|apps/|tests/)' | sort -u | xargs -r clang-format -n --Werror
# 自动格式化（同上收集逻辑，写入模式）
{
  git diff --cached --name-only --diff-filter=d
  git diff --name-only --diff-filter=d
  git ls-files --others --exclude-standard
} | grep -E '\.(c|cc|cpp|cxx|h|hpp|hxx)$' | grep -E '^(include/|src/|apps/|tests/)' | sort -u | xargs -r clang-format -i
```
若本地可用，可基于 `build/compile_commands.json` 运行 `clang-tidy`。

文档链接检查使用 `python3 tools/check_docs_links.py`。默认检查根目录 `README.md`、`AGENTS.md`、`CHANGELOG.md` 和 `docs/` 非归档文档中的相对链接及 Markdown 标题锚点。

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

## 10. 产品文档

- `docs/product.md`：**当前产品需求与验收标准**（产品范围、公开 API、架构或验收标准相关工作时必读）

## 11. 指令优先级与作用域

多级指令共存时，按以下优先级裁决：

1. **用户当前明确指令**：最高优先级，覆盖一切与之冲突的静态文档。
2. **已验证的仓库事实**（代码行为、CMake 配置、已实现的接口等）：次高优先级。
3. **根目录 AGENTS.md**：仓库级默认规则，对整个仓库生效。
4. **嵌套 AGENTS.md**：在其所在目录子树内添加局部规则与约束；与根 AGENTS.md 冲突时以根 AGENTS.md 为准，除非用户明确指示使用嵌套规则。

嵌套 AGENTS.md 中的审批门（如生成代码前需先审核）仅在其所在目录子树内生效，不影响同仓库其他模块。

## 12. Agent 上下文

使用仓库内上下文恢复或继续已有工作时：

1. 先读本文件，按需通过 [文档导航](docs/README.md) 找到任务相关资料。
2. 继续已有任务时，按该任务明确指定的 handoff 恢复。仅在追溯上轮文档整理时阅读 `docs/agent/handoff.md`。
3. 修改模块边界、接口或所有权时，读取 [源码模块所有权](docs/architecture.md#源码模块所有权)，并从 [项目手册](docs/handbook/README.md) 查对应实现、接口与测试。
4. `docs/guides/**` 按需引用。
5. 默认启动加载量控制在 ≤15,000 tokens；超出时只保留本文件和用户明确指定的文件。

约束：

- `docs/agent/` 不维护 workstream、frontmatter、模块记忆或 ADR 子目录。
- 恢复上下文不构成业务执行授权；执行构建、测试或写入前仍需用户明确确认。
- 上下文与代码冲突时，以已验证的代码、CMake 与测试事实为准。

## 13. 推荐代理工作流
算子语义、reference kernel、optimized kernel、packing/layout、fusion 或 threading 工作必须遵循 `docs/guides/operator-development-workflow.md`（该单一文档含 O0–O4 证据门禁、Benchmark 规范、证据记录格式（§4）与优化方法（§7）），并达到对应证据门禁。不得以 microbenchmark 代替 production-path 或端到端证据。

1. 修改前先阅读相关 `CMakeLists.txt` 和 `AGENTS.md`；涉及产品范围、公开 API、架构或验收标准时，还需阅读 `docs/product.md`。
2. 进行最小化、风格一致的改动。
3. 构建最小受影响目标（`--target <name>`）。
4. 先跑单个聚焦测试（`--gtest_filter=Suite.Case`），再扩大范围。
5. 仅性能敏感改动需跑聚焦 benchmark。
6. 改动构建选项/宏时，验证受影响的非默认配置；涉及并发/同步或相关构建选项时至少验证 TSAN。
7. 按 `.clang-format` 格式化改动文件。
8. 在汇报中明确风险：正确性、并发、内存、性能。
