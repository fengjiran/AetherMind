# CTest 与 sanitizer 第二批验收

- **状态**: Current
- **日期**: 2026-10-26
- **关联方案**: [13 号方案 Q2](../improvement-plan/13-deepinfer-quality-and-validation.md)、[02 工程质量](../improvement-plan/02-engineering-quality-system.md)
- **基线**: `d448e36fbaa50984bd5eced49b0f8652d079c62b` 加本批工作树补丁；执行时 dirty，尚未创建提交。
- **环境**: Linux x86_64、GCC 14.2.0、CMake/CTest 3.28.3、系统 GoogleTest 1.16.0。

## 1. 实现范围

[CMakePresets.json](../../CMakePresets.json) 提供 Release、ASAN/UBSAN、独立 TSAN 和关闭测试依赖的 CLI 配置。[sanitizer helper](../../cmake/AetherMindSanitizers.cmake) 只向项目 targets 注入编译/链接 flags，覆盖共享库、CLI、C/C++ 单测和测试 probe；第三方 targets 保持原有选项。旧 `ENABLE_TSAN` 选项已移除且不提供兼容映射；非法 `AETHERMIND_SANITIZER` 值在 configure 时拒绝，ASAN/UBSAN 与 TSAN 不能合并。

[CTest helper](../../cmake/AetherMindTests.cmake) 使用标准 GoogleTest PRE_TEST discovery。labels 区分 smoke、numerical、lifecycle、context_boundary、model、cli、interposer、isa-avx2 和 thread-smoke；validation 是限定生产链集合。model case 依赖 checksum fixture，正式配置缺资产即失败；普通配置对缺模型/CLI 的用例明确标为 disabled，不能计作通过。Python CLI 逐方法注册，两项 preload 注入只在适用普通构建运行。

[malloc interposer](../../tests/unit/inference/test_malloc_interposer.cpp) 在 sanitizer 构建中不定义进程级 malloc-family/madvise 覆盖符号，而非仅在运行时过滤用例。正常 Linux/glibc 构建继续运行故障注入与 prepared Decode 零分配合同。精确适用清单包含 typed ParseNumber 以及无 availability guard 的分配计数用例。

[GTest 原生 guard](../../tests/unit/test_utils/test_required_execution_guard.cpp) 仅在 AETHERMIND_REQUIRED_VALIDATION=1 时，将 OnTestEnd 中的 skip 转为失败，且先于普通/brief printer 输出。listener/environment 均由 GTest 拥有；直接 GTest 调用保持原有 skip 行为。没有复制或修改 CMake 内部 GoogleTest 模块。缺二进制占位和零 discovery 均带 validation/build 失败项。

本批保留了执行前已有的 hf_directory_reader.cpp 格式修改，未暂存、未提交；未修改推理 API 或 kernel 算法。

## 2. 动态结果

| 配置 | CTest 验收项 | 其中 GTest case | 其他验收项 | 结果 |
|---|---:|---:|---|---|
| validation-release | 362 | 319 | 2 CMake 合同、fixture 校验、Python converter、39 Python CLI 方法 | 全部通过，0 skipped/disabled |
| validation-asan-ubsan | 268 | 227 | 2 CMake 合同、fixture 校验、Python converter、37 普通 Python CLI 方法 | 全部通过，0 skipped/disabled；ASAN/UBSAN/LSAN 无报告 |
| validation-tsan | 190 | 186 | 2 CMake 合同、fixture 校验、Python converter | 全部通过，0 skipped/disabled；包含 7 项基础线程测试 |

各配置复用既有测试，不能相加为独立用例数。Python/CMake 验收项不等同于受 sanitizer instrumentation 的 C++ case。实际 discovery 集合与 JUnit 名称、数量逐项一致；正式集合中没有 disabled 项。sanitizer 下的 interposer/preload 用例属于显式排除范围，未被计入通过数。

验证还包括：

- 隔离 configure 探测：none、ASAN/UBSAN、TSAN、未知值拒绝；工具链探测覆盖编译与链接，非法组合在 configure 时失败。
- 真实 compile_commands 核对：三个配置的项目 C/C++ 源文件 flags 符合选择，第三方源码无 sanitizer flags。
- nm 检查：普通 interposer 对象保留覆盖符号，ASAN/TSAN 对象均不导出这些符号。
- 三个完整单测 executable 的 interposer 合同可启动；当前 ASAN/LSAN 运行没有复现旧函数 registry 泄漏基线。
- tests/benchmarks 关闭、GTest/Python 查找禁用时，独立 CLI 配置和实际构建通过。

## 3. 防假通过的负面证据

| 场景 | 实际结果 |
|---|---|
| required GTest skip，普通与 brief printer | 子 CTest 2 failed、exit 8，并包含 guard 明确诊断 |
| optional GTest skip | 明确 Skipped、exit 0，未转换为通过 |
| required 空模型 root | checksum fixture 失败，model case 阻断，exit 8 |
| optional 空模型 root | model case 为 DISABLED + missing_assets |
| unit executable 未构建 | NOT_BUILT 被 validation 选中，exit 8 |
| executable 存在但 discovery 无 case | aethermind.gtest_empty 失败，exit 8 |

[分类合同](../../tests/cmake/test_test_classification.cmake) 覆盖真实参数化/typed 名称、kernel numerical 标签和线程集合；[skip 合同](../../tests/cmake/test_ctest_skip_policy.cmake) 调用真实子 CTest 与最小 GTest probe，检查明确失败原因，不能将启动崩溃算作预期 skip 拒绝。

## 4. 复现

先按 [fixture 合同](../../tests/fixtures/inference_admission/README.md) 准备并校验固定资产，再从仓库根目录运行：

```bash
cmake --preset validation-release
cmake --build --preset validation-release
ctest --preset validation-release

cmake --preset validation-asan-ubsan
cmake --build --preset validation-asan-ubsan
ctest --preset validation-asan-ubsan

cmake --preset validation-tsan
cmake --build --preset validation-tsan
ctest --preset validation-tsan
```

本次配置通过 `-DFETCHCONTENT_SOURCE_DIR_SPDLOG=/home/richard/project/AetherMind/build/_deps/spdlog-src` 复用已有的 1.17.0 源码。ASAN preset 使用 `halt_on_error=1:detect_leaks=1`，UBSAN 使用 halt_on_error；没有关闭 leak detection 或添加 suppression。

本主机裸 TSAN 最小程序在 main 前报 unexpected memory mapping、exit 66。进程级 `setarch x86_64 -R` 后最小程序以及真实集合均运行成功，因此本次实际 TSAN 命令为：

```bash
setarch x86_64 -R ctest --preset validation-tsan
```

这是本主机的运行条件，不是工具链仅能链接就算运行通过，也没有修改系统级 ASLR 配置。不允许该 personality 设置的主机仍须报告启动失败。

独立 CLI 验收复用既有 build-infer 目录：

```bash
cmake --preset validation-cli-only -B build-infer \
  -DFETCHCONTENT_SOURCE_DIR_SPDLOG="$PWD/build/_deps/spdlog-src" \
  -DCMAKE_DISABLE_FIND_PACKAGE_GTest=ON -DCMAKE_DISABLE_FIND_PACKAGE_Python3=ON
cmake --build build-infer --target aethermind_infer -j2
```

## 5. 证据与剩余边界

原始 configure/build/CTest 日志、JUnit、inventory JSON 位于 gitignored `build/validation/<preset>/`；集合校验摘要为 `build/validation/q2-results.json`。辅助负面探测和配置检查归档在 `build/validation/q2-artifacts/`。这些是本地构建产物，尚无 CI retention 承诺。

文档检查保持本批前的 2 problems / 29 warnings，诊断集合逐字一致；两条旧断链仍指向历史 src/function.cpp。本批没有新增文档诊断，也不能将该检查写成全门禁通过。

本批没有运行全仓测试、模型 benchmark、外部 numerical golden 或第三批 ISA 编译隔离。unit target 的全局 AVX2/FMA flags 仍待下一批收敛；当前结果不证明 baseline CPU 可移植性。基础线程用例及同步产品路径的 TSAN 结果不构成并发 Session 支持。

GCC TSAN 构建明确警告既有 Object 引用计数中的 atomic_thread_fence 不被其完整建模；Release/TSAN 还保留 SafetensorsIndexParser/StatusOr 内联路径的 Wfree-nonheap-object 诊断，未在本批消除。warnings baseline/profile 仍需后续验收。本次通过只对应上述配置与集合，不能据此宣布全部 PRD hard gates 或 Q2 完成。
