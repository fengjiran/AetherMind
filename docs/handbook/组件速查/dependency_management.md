## 1. 总体方案

AetherMind 推理引擎采用 **CMake 作为唯一的依赖管理入口**，不使用语言级包管理器（无 go.mod/package.json/requirements.txt）。第三方依赖分为两类：

- **FetchContent + 自定义获取脚本**：spdlog、googletest、google_benchmark 通过 `cmake/AcquireSharedDependency.cmake` 拉取到 `.deps/<name>`，并强制校验 tag、本地修改与目录完整性。
- **Git 子模块**：libbacktrace、cpufp、STREAM 三个底层库以 `.gitmodules` 声明，路径位于 `3rdparty/`。

构建要求 CMake ≥ 3.28、C++20（含 `<format>`），并通过 `find_package(Threads REQUIRED)` 使用系统线程库。

## 2. 关键文件

| 文件 | 作用 |
|---|---|
| `CMakeLists.txt` | 根装配，声明 spdlog / googletest / google_benchmark 的版本 tag 并调用共享依赖函数 |
| `cmake/AcquireSharedDependency.cmake` | 核心获取器：Git clone → detached checkout → tag 校验 → 原子重命名发布 |
| `cmake/AetherMindDependencySources.cmake` | `aethermind_declare_shared_dependency()` / `aethermind_make_shared_dependency_available()` 两个对外 API |
| `.gitmodules` | 声明 libbacktrace、cpufp、STREAM 三个子模块及其 SSH URL |
| `cmake/AetherMindFindPython.cmake` | Python 发现逻辑（测试/工具链） |
| `cmake/AetherMindLibbacktrace.cmake` | libbacktrace 集成（由子模块提供） |

## 3. 架构与约定

### 3.1 共享依赖获取流程 (`AcquireSharedDependency.cmake`)

每个共享依赖的获取被封装为一次独立的 CMake 进程调用（`-P AcquireSharedDependency.cmake`），遵循以下顺序：

1. 校验参数 `SOURCE / REPOSITORY / TAG / MODE` 均为必填。
2. 默认 `LOCK_TIMEOUT=600s`、`DOWNLOAD_TIMEOUT=180s`，可通过 `AETHERMIND_DEPENDENCY_LOCK_TIMEOUT` / `AETHERMIND_DEPENDENCY_DOWNLOAD_TIMEOUT` 覆盖。
3. 对父目录创建 `.${directory}.lock` 进程级 advisory lock（`file(LOCK ... GUARD PROCESS TIMEOUT ...)`），保证多并行配置互斥。
4. 若源目录已存在且非空：检查它是独立 Git 仓库、HEAD commit 等于 tag、且无本地变更；任一失败则拒绝继续构建。
5. 若为空或不存在：在 `parent/.<dir>.tmp-<random>` 下 clone --no-hardlinks --no-checkout，再 detach checkout 到 tag。
6. 用最多 16 次尝试将 staging 目录原子 rename 到目标 `source`，避免 ExternalProject 的空占位符竞争。
7. 所有失败路径均删除临时 staging 目录。

模式变量 `AETHERMIND_DEPENDENCY_MODE` 控制行为：
- `ENSURE`（默认）：缺失时在线克隆。
- `VERIFY`：仅校验已有内容，缺失即报错——用于 `FETCHCONTENT_FULLY_DISCONNECTED` 下的离线验证。

### 3.2 对外 API (`AetherMindDependencySources.cmake`)

```text
aethermind_declare_shared_dependency(<name> <absolute_source_dir> <repository_url> <tag>)
aethermind_make_shared_dependency_available(<name> TARGETS target1 target2 ...)
```

- 声明阶段通过 `FetchContent_Declare` 注册，但把 `SOURCE_SUBDIR` 设为 `__aethermind_population_only__`，阻止未经验证的第三方 `CMakeLists.txt` 被直接执行。
- 可用阶段通过 `FetchContent_MakeAvailable` 触发真正的 populate，此时 `_available` 检查目标是否已由外部 provider 暴露；若未暴露，再按 `ENSURE`/`VERIFY` 重新运行获取器。
- 最后要求目标确实定义，否则 `FATAL_ERROR`。

这种设计允许上层项目通过 `FetchContent_Populate` 或自定义 target 提前注入依赖，同时仍保留 AetherMind 自管版本的回退能力。

### 3.3 版本清单

| 依赖 | Tag | 来源 |
|---|---:|---|
| spdlog | v1.17.0 | github.com/gabime/spdlog.git |
| googletest | v1.18.0 | github.com/google/googletest.git |
| google_benchmark | v1.9.5 | github.com/google/benchmark.git |
| libbacktrace | — (子模块) | git@github.com:tlc-pack/libbacktrace.git |
| cpufp | — (子模块) | git@github.com:pigirons/cpufp.git |
| STREAM | — (子模块) | git@github.com:jeffhammond/STREAM.git |

### 3.4 Provider 优先机制

根 `CMakeLists.txt` 先检查 `spdlog::spdlog` / `GTest::gtest` / `benchmark::benchmark` 等 target 是否已存在；若存在则跳过自管下载。若只提供了其中一个 target（例如只有 `GTest::gtest` 没有 `GTest::gtest_main`），会直接 `FATAL_ERROR`，确保消费方始终获得完整的最小 target 集。

## 4. 约定与约束

- **必须使用 CMake ≥ 3.28**：根 `cmake_minimum_required(VERSION 3.28)`，并在启动时检测 `std::format` 支持，缺失则 `FATAL_ERROR`。
- **共享依赖源码必须位于绝对路径**：`aethermind_declare_shared_dependency` 校验 `IS_ABSOLUTE source_dir`，否则 `FATAL_ERROR`。
- **共享依赖必须有 version tag**：空 tag 直接报错；tag 形如 `v1.17.0`，通过 `refs/tags/<tag>` 精确检出。
- **禁止符号链接**：若 `.deps/<name>` 是 symlink，获取器报 `managed source is a symbolic link; use an explicit source override instead`。
- **禁止本地修改**：checkout 后执行 `git status --porcelain=v1`，若有 untracked 或 modified 文件即 `FATAL_ERROR`，防止误提交第三方源码。
- **必须是独立 Git 仓库**：校验 `git rev-parse --show-toplevel` 与 checkout 目录一致，拒绝嵌套仓库。
- **并行安全**：通过 `file(LOCK ... GUARD PROCESS TIMEOUT ...)` 序列化多个 CMake configure 进程；`GIT_TERMINAL_PROMPT=0` 禁用交互式提示；`GIT_OPTIONAL_LOCKS=0` 强制锁定。
- **离线构建**：设置 `FETCHCONTENT_FULLY_DISCONNECTED` 时进入 VERIFY 模式，只校验本地 .deps 目录，不联网。
- **Provider 契约**：外部 provider 必须暴露 `spdlog::spdlog`（或 `spdlog`）、`GTest::gtest` + `GTest::gtest_main`、`benchmark::benchmark` + `benchmark::benchmark_main`；缺少任一 target 即 `FATAL_ERROR`。
- **GoogleTest 选项**：`BUILD_GMOCK=OFF`、`INSTALL_GTEST=OFF`、`gtest_build_tests=OFF`、`gtest_build_samples=OFF`，最小化构建产物。
- **Google Benchmark 选项**：`BENCHMARK_ENABLE_GTEST_TESTS=OFF`、`BENCHMARK_ENABLE_TESTING=OFF`、`BENCHMARK_ENABLE_INSTALL=OFF`。
- **spdlog 后端选择**：默认启用 `SPDLOG_USE_STD_FORMAT=ON`（因为根工程已要求 C++20 `<format>`），可显式关闭切换至 fmt 后端。
- **子模块 URL 使用 SSH**：`.gitmodules` 中三个子模块均使用 `git@github.com:...` 格式，需要对应 SSH key。
- **Lock 超时可配置**：`AETHERMIND_DEPENDENCY_LOCK_TIMEOUT` 默认 600 秒，`AETHERMIND_DEPENDENCY_DOWNLOAD_TIMEOUT` 默认 180 秒，必须为正整数。
- **更新依赖需显式升级**：由于 tag 校验严格，任何对 `.deps/<name>` 的手动修改都会导致下次 configure 失败，强制走官方 tag 升级流程。
