## 1. 总体方案

AetherMind 推理引擎采用 **CMake 3.28+** 作为唯一构建系统，顶层 `CMakeLists.txt` 声明工程版本 `1.0.0`、启用 C++20、通过编译 `constexpr std::format_string<int>` 断言标准库 `<format>` 可用性，并通过 `include(cmake/*.cmake)` 聚合共享逻辑。项目提供 `tools/build_project.sh` 包装脚本与 `CMakePresets.json` 预置配置，统一驱动 configure / build / test 流程。

## 2. 关键文件

- `CMakeLists.txt`：根装配入口，定义选项、依赖拉取、子目录（`src`、`apps/inference`、`tests/unit`、`tests/benchmark`）、Doxygen 文档目标。
- `cmake/AetherMindDependencySources.cmake` + `cmake/AcquireSharedDependency.cmake`：统一的第三方依赖获取框架，通过 `aethermind_declare_shared_dependency()` 声明、`aethermind_make_shared_dependency_available()` 消费，基于 `FetchContent` 并支持外部 provider 注入目标。
- `cmake/AetherMindSanitizers.cmake`：`AETHERMIND_SANITIZER=none|asan-ubsan|tsan` 三态注入，编译/链接双探针验证，仅作用于 project-owned target。
- `cmake/AetherMindTests.cmake`：测试标签体系 (`unit`/`smoke`/`numerical`/`model`/`cli`/`isa-avx2`/`lifecycle`/`context_boundary`/`interposer`/`validation` 等) 与 CTest label 过滤。
- `cmake/AetherMindCpuIsa.cmake`：CPU ISA 探测 (AVX2/F16C)，对应 `AETHERMIND_ENABLE_AVX2`/`AETHERMIND_ENABLE_F16C` 选项。
- `cmake/AetherMindBuildChecks.cmake`：构建期能力检查（如内存对齐校验）。
- `cmake/AetherMindFindPython.cmake`：Python 发现逻辑。
- `cmake/AetherMindLibbacktrace.cmake`：libbacktrace 集成。
- `tools/build_project.sh`：用户友好的构建入口。
- `CMakePresets.json`：CI 与开发者共用 preset（`validation-*`、`benchmark-*`、`validation-cli-only`、ASAN/TSAN 变体）。

## 3. 架构与约定

### 3.1 依赖管理
所有第三方库通过 `aethermind_declare_shared_dependency(name source_dir repository tag)` 集中声明，例如：
- spdlog v1.17.0 → `.deps/spdlog`
- googletest v1.18.0 → `.deps/googletest`
- google-benchmark v1.9.5 → `.deps/benchmark`

依赖的获取被推迟到 `FetchContent_Declare` 阶段，并使用 `SOURCE_SUBDIR __aethermind_population_only__` 防止未校验的第三方 `CMakeLists.txt` 直接执行；实际源码在 `aethermind_make_shared_dependency_available()` 中经 `add_subdirectory` 引入。该机制允许上游通过 `target` provider 注入已存在的 IMPORTED target，从而绕过 FetchContent。

AcquireSharedDependency.cmake 使用进程级文件锁 (`file(LOCK ... GUARD PROCESS)`) 串行化并发克隆，默认锁超时 600s、下载超时 180s；通过 `GIT_TERMINAL_PROMPT=0` 禁用交互，并在原子重命名前对 staging 目录做 tag SHA 一致性校验，拒绝符号链接与本地修改的 checkout。FETCHCONTENT_FULLY_DISCONNECTED 模式降级为 VERIFY 而非主动获取。

### 3.2 Sanitizer 注入
`AETHERMIND_SANITIZER` 仅在 GNU/Clang/AppleClang Unix 工具链上生效，且通过 C/C++ 双向 compile+link 探针确认工具链可用；`asan-ubsan` 使用 `-fsanitize=address,undefined`，`tsan` 使用 `-fsanitize=thread` 并将目标设为 `-O1`。`aethermind_apply_sanitizer(target)` 仅对显式调用的 project-owned target 生效，第三方 target 保留自身选项。

### 3.3 CPU ISA 策略
通过 `AetherMindCpuIsa.cmake` 探测 AVX2/F16C 硬件能力，并以 `AETHERMIND_ENABLE_AVX2`/`AETHERMIND_ENABLE_F16C` 开关控制是否编译对应内核路径；`CMakePresets.json` 中 `validation-baseline` 关闭可选 ISA，`validation-avx2` 开启以区分 baseline 与加速路径。

### 3.4 测试标签体系
`AetherMindTests.cmake` 根据测试名称正则自动打标签（`unit`、`smoke`、`numerical`、`model`、`cli`、`isa-avx2`、`lifecycle`、`context_boundary`、`interposer`、`validation`、`long-context`、`thread-smoke` 等），CTest 通过 `testPresets` 中的 `filter.include/exclude` 精确选择用例集合并输出 JUnit XML。

### 3.5 构建产物
- 核心库由 `src/CMakeLists.txt` 聚合。
- CLI 可执行 `aethermind_infer` 位于 `apps/inference/`，默认不构建，需 `BUILD_INFERENCE_TOOL=ON`。
- 单元测试在 `tests/unit/`，基准在 `tests/benchmark/`。
- Doxygen 文档通过自定义目标 `aethermind_docs` 生成至 `${CMAKE_BINARY_DIR}/docs`。

## 4. 约定与约束

- **C++ 标准**：根 `CMakeLists.txt` 强制 `CMAKE_CXX_STANDARD=20`，并通过编译 `constexpr std::format_string<int>` 断言编译器必须提供 `<format>`，否则 `FATAL_ERROR`。
- **位置无关代码**：全局启用 `CMAKE_POSITION_INDEPENDENT_CODE ON`。
- **依赖版本锁定**：spdlog `v1.17.0`、googletest `v1.18.0`、google-benchmark `v1.9.5` 硬编码于根 CMakeLists。
- **Provider 契约**：若上游提供了 spdlog/GTest/benchmark 目标，必须同时提供约定好的多个 target（如 `GTest::gtest` 与 `GTest::gtest_main`），否则报 `FATAL_ERROR`；缺失任一必需 target 时构建失败。
- **Sanitizer 白名单**：`AETHERMIND_SANITIZER` 仅接受 `none|asan-ubsan|tsan`，非 Unix 或不受支持的编译器直接 `FATAL_ERROR`。
- **模型夹具路径**：测试要求绝对路径 `AETHERMIND_TEST_MODELS_DIR`（默认 `.models`）和可选 `AETHERMIND_TEST_NUMERICAL_DIR`，均为绝对路径校验。
- **构建类型限制**：`tools/build_project.sh` 仅接受 `Debug|Release|RelWithDebInfo|MinSizeRel`，其余直接退出。
- **Preset 隔离**：`CMakePresets.json` 将 validation、benchmark、CLI-only 三类场景拆分为独立 `configurePreset`/`buildPreset`/`testPreset`，各自指定 `binaryDir`、`targets`、`jobs` 与 JUnit 输出路径。
- **Doxygen 依赖**：仅当 `BUILD_DOCS=ON` 时才 `find_package(Doxygen REQUIRED dot)`，避免非必要依赖阻塞常规构建。
- **Backtrace 选项**：`USE_LIBBACKTRACE` 默认 `ON`（非 MSVC），`BACKTRACE_ON_SEGFAULT` 默认 `ON`，可通过 `--disable-libbacktrace` / `--disable-segfault-hook` 关闭。
- **验证模式**：`AETHERMIND_VALIDATION_REQUIRED_ASSETS` 用于 CI 环境强制要求 HF 模型夹具与 CLI 存在，以便运行端到端数值校验。
- **依赖获取安全**：受管依赖目录不可为符号链接；现有 checkout 必须满足 Git tag SHA 一致、无本地修改且为独立仓库根，否则保持原内容并中止构建。
- **并行克隆保护**：依赖获取通过父目录下的隐式 `.lock` 文件实现进程级互斥，防止并发 `git clone` 冲突。
- **离线构建**：设置 `FETCHCONTENT_FULLY_DISCONNECTED` 后，缺失的依赖源会触发 VERIFY 模式而非网络获取，适合 CI 缓存场景。
