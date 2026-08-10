# ammalloc 独立仓库化迁移清单

> 状态：待执行（决策已确认）
> 目标：ammalloc 作为**独立 git 仓库**演进（丢弃历史、非子模块），AetherMind 通过 FetchContent 引入

基于当前已验证的仓库事实编写，分为 5 个阶段。

---

## 阶段 0：前置决策

| 决策项 | 建议值 | 说明 |
|---|---|---|
| 托管位置 | 自备（GitHub/GitLab SSH） | 与 3rdparty 现有子模块同风格 |
| 首个版本 | `v0.1.0` | 独立演进后必须 tag 化发布 |
| 执行时机 | AetherMind 首次正式使用 ammalloc 时 | 若 ammalloc 先独立发布则提前 |
| 开发期联动 | `FETCHCONTENT_SOURCE_DIR_AMMALLOC` | AetherMind 本地开发可指向 ammalloc 本地克隆，避免每次拉远端 |

## 阶段 1：独立仓库创建（丢弃历史）

1. **确认迁移源快照**：以当前工作树 `ammalloc/` 为准（已完全解耦状态，24 个 git 跟踪文件：`AGENTS.md`、`CMakeLists.txt`、`GEMINI.md`、`include/ammalloc/` 13 头、`src/` 9 源）。
2. **创建新仓库**：在托管平台建空仓库，`git init` 后 `git remote add origin <url>`（**不用** filter-repo/subtree split，历史直接丢弃）。
3. **补 `.gitignore`**（当前 ammalloc 目录下为空，独立仓库必须自备）：
   ```
   build*/
   .cache/
   ```
   否则 `ammalloc/build/` 会进入独立仓库（当前依赖根 .gitignore 的 `build/` 模式，独立后失效）。
4. **初始提交并打 tag**：
   ```bash
   git add -A && git commit -m "chore: import ammalloc as standalone repository (v0.1.0)"
   git tag v0.1.0 && git push origin main --tags
   ```
5. **删除 AetherMind 本地残留**（可选，见阶段 3）：`ammalloc/` 目录本体在 AetherMind 主仓库中 `git rm -r ammalloc/`。

## 阶段 2：测试随库走（ammalloc 自包含测试闭环）

当前 10 个测试文件位于 AetherMind 仓库，独立演进后必须迁入，否则独立仓库无法自证正确：

1. **迁移文件**（从 AetherMind 仓库移入 ammalloc 独立仓库）：
   - `tests/unit/memory/test_page_cache.cpp`、`test_page_allocator.cpp`、`test_size_class.cpp`、`test_central_cache.cpp`、`test_thread_cache.cpp`
   - `tests/unit/base/test_span.cpp`（Span 测试，随库走）
   - `tests/benchmark/ammalloc/` 下 4 个 benchmark 文件
   - 目标布局建议：`ammalloc/tests/unit/`、`ammalloc/tests/benchmark/`
2. **ammalloc/CMakeLists.txt 扩展测试构建**（在现有 `if(BUILD_TESTS)` 逻辑上扩展）：
   - `BUILD_TESTS` ON 时：`find_package(GTest)` 或 FetchContent 获取 GoogleTest，`add_executable(ammalloc_unit_tests ...)`，链接 `ammalloc` + `GTest::gtest_main`；保留 `AMMALLOC_TEST` 宏定义（测试依赖 `g_mock_huge_alloc_fail` 等 AMMALLOC_TEST guard 符号）；
   - `BUILD_BENCHMARKS` ON 时：FetchContent `google_benchmark`（与根一致 `v1.9.5`），benchmark 目标链接 `ammalloc` + `benchmark::benchmark_main`——**直接链接 ammalloc**，避免重蹈"依赖宿主库传递符号"的坑（见已记录 pitfall）。
3. **本地验证**：独立仓库内 `cmake -S . -B build -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON` 全量构建 + 测试（89 个用例应全绿）。
4. **CI（可选但推荐）**：GitHub Actions 配 gcc/clang 两个矩阵跑 `ammalloc_unit_tests`。

## 阶段 3：AetherMind 侧解除绑定（原子切换，一次提交完成）

> ⚠️ 步骤 3.1 和 3.2 必须**同一提交**完成，避免中间态（移除 add_subdirectory 但未声明 FetchContent 时构建断裂）。

1. **根 CMakeLists.txt**（`add_subdirectory(ammalloc)` 处）：
   ```diff
   -add_subdirectory(ammalloc)
   +FetchContent_Declare(
   +        ammalloc
   +        GIT_REPOSITORY <ammalloc-repo-url>
   +        GIT_TAG v0.1.0
   +)
   +FetchContent_MakeAvailable(ammalloc)
   ```
   位置必须保持在 `FetchContent_MakeAvailable(spdlog google_benchmark)` **之后**——ammalloc 的 `if(NOT TARGET spdlog::spdlog)` 依赖宿主已提供 spdlog。`target_include_directories(ammalloc ...)` 的 include 路径在 FetchContent 下自动正确（`CMAKE_CURRENT_SOURCE_DIR` 解析到 `_deps/ammalloc-src`），无需改动。
2. **src/CMakeLists.txt**：删除 PRIVATE 链接 `ammalloc`（空链接，AetherMind.so 未导出其符号——已验证）。
3. **tests/unit/CMakeLists.txt**：删除 `ammalloc` 链接 + 删除已迁走的 6 个测试文件（若测试已迁入独立仓库）。
4. **tests/benchmark/CMakeLists.txt**：删除 `ammalloc/include` 路径与 `ammalloc` 链接 + 删除 `tests/benchmark/ammalloc/` 目录。
5. **文档同步**：根 [AGENTS.md](../../../AGENTS.md)（模块表、目录结构、构建命令、clang-format 范围中 `ammalloc/` 条目）、docs/designs 中引用。
6. **`.gitmodules` 不动**（ammalloc 非子模块，无条目）。
7. **本地开发提示**：AetherMind 开发期间用 `-DFETCHCONTENT_SOURCE_DIR_AMMALLOC=/path/to/ammalloc` 指向本地克隆，避免每次构建拉远端。

## 阶段 4：全量验证

| # | 验证项 | 命令 |
|---|---|---|
| 1 | ammalloc 独立仓库自测 | `cmake -S ammalloc -B build -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON && cmake --build build -j && ./build/tests/unit/ammalloc_unit_tests` |
| 2 | AetherMind 重新配置 | `cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`（确认 FetchContent 拉取 ammalloc tag） |
| 3 | AetherMind 构建 | `cmake --build build --target AetherMind aethermind_unit_tests aethermind_benchmark -j` |
| 4 | 依赖方向断言 | `nm -D build/libAetherMind.so \| grep am_malloc` 应为空（AetherMind 仍不依赖 ammalloc 符号） |
| 5 | 单测回归 | `./build/tests/unit/aethermind_unit_tests`（非 ammalloc 测试全部通过） |

## 阶段 5：常态工作流（迁移完成后）

- **ammalloc 变更**：独立仓库提交 → 打 tag（`v0.1.1`、`v0.2.0`…）→ push；
- **AetherMind 升级依赖**：更新根 CMakeLists 的 `GIT_TAG` → 重新配置构建；
- **AetherMind 核心库首次使用 ammalloc**：直接 `#include "ammalloc/ammalloc.h"` + `ammalloc::am_malloc(...)`（命名空间已独立），链接 ammalloc target（从 PRIVATE 空链接改为真实使用）。

---

## 风险与注意事项汇总

1. **切换原子性**：阶段 3 的 FetchContent 声明与 `add_subdirectory` 移除必须同一提交，否则中间态不可构建；
2. **AMMALLOC_TEST 宏**：测试迁入后，独立测试目标编译时必须定义该宏（依赖 `g_mock_huge_alloc_fail`），且只对测试构建生效；
3. **测试依赖面**：迁移前需确认 10 个测试文件不引用 AetherMind 头文件（此前调查确认仅用 ammalloc + gtest/benchmark，但迁移时建议再 grep 一遍 `aethermind/` include）；
4. **版本锁定**：ammalloc 独立后 AetherMind 必须钉 tag（不可用 `main` 分支追踪，否则构建不可复现）；
5. **build 目录清理**：迁移后 AetherMind 根目录下的 `ammalloc/build`（独立构建产物）应删除。该目录仍被根 .gitignore 的 `build/` 模式覆盖，删除仅为了整洁，不影响 git 行为；
6. **历史信息保全**：丢弃 git 历史前，确认设计知识已存在于 `ammalloc/AGENTS.md`、`GEMINI.md`、docs/designs/ 与记忆系统（已验证：关键设计决策均已沉淀）。
