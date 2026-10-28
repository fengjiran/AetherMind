# 五批验证的集成与最终验收

- **状态**: Current
- **日期**: 2026-10-28
- **范围**: [13 号方案](../improvement-plan/13-deepinfer-quality-and-validation.md) 已实现的第一至第五批；不替代其余平台、模型质量、优化推广或安装交付门禁。

## 1. 入口与前置条件

[统一驱动](../../tools/run_final_validation.py) 编排既有 CMake presets、GTest/CTest、Python 黑盒测试及资产校验，不另建测试框架。当前完整矩阵针对 glibc Linux x86_64、具备 usable AVX2/FMA 的节点；需要 C++20 compiler、CMake/CTest >=3.28、系统 GoogleTest 与 Python3。

从仓库根目录运行：

```bash
python3 tools/prepare_validation_assets.py
python3 tools/run_final_validation.py --jobs 4 --build-concurrency 2
```

第一条默认只离线复核现有资产。第二条配置、构建、执行全部七个 profile。只有全部请求通过、源码与资产前后身份一致，且确实覆盖全部七项时，summary.json 的 milestone_complete 才为 true。任一必需项失败均返回非零。

可显式使用本地依赖源码，避免 CMake FetchContent 获取网络依赖：

```bash
python3 tools/run_final_validation.py \
  --spdlog-source "$PWD/build/_deps/spdlog-src" \
  --google-benchmark-source "$PWD/build/_deps/google_benchmark-src" \
  --jobs 4 --build-concurrency 3
```

驱动不下载模型或安装依赖；CMake 原有依赖获取机制保持原义。该命令可以直接作为 CI job 的入口，但本仓库没有因本轮增加特定云服务的执行或 artifact retention 承诺。

## 2. 外部资产准备

源 checkpoint、admission variants 和 numerical reference bundle 不随 Git 交付。准备工具沿用 [admission](../../tests/fixtures/inference_admission/README.md) 和 [numerical](../../tests/fixtures/inference_numerical/README.md) 的单一 recipe/checksum 权威。

- 已有完整资产：只校验，不覆盖。
- 缺源：必须显式 --download-source，只获取 recipe 固定 revision 的 config/weight，并验证字节与 SHA256。
- 缺 admission：标准库生成到 staging，检查后原子发布；失败不留下可误用的最终目录。
- 缺 numerical：提供已审阅 bundle，或指定具有冻结 HF 依赖的隔离 Python；新 candidate 验证通过后才发布。
- 任何路径都不会自动更新仓库中的 expected checksum。模型与输出路径拒绝 symlink/逃逸；Python interpreter 保留正常 venv 的最终 executable symlink 调用方式。

例如 CI 已取得受校验的 numerical artifact：

```bash
python3 tools/prepare_validation_assets.py \
  --download-source --numerical-bundle "$VALIDATED_NUMERICAL_BUNDLE"
```

或在冻结参考环境内重新生成：

```bash
python3 tools/prepare_validation_assets.py \
  --download-source --reference-python "$REFERENCE_ENV/bin/python3"
```

环境版本、轮子/参考源码或浮点结果变化导致 checksum 不匹配时必须停止，不能用当前 manifest 晋升为期望。绝对 --models-dir 可以选择其他根；位于源码仓库内的生成资产遵循现有 generator 的 .models/ 目录约束。外部临时/缓存目录也可按其既有合同使用。

## 3. 统一矩阵

| profile | 构建 preset | 执行集合 |
|---|---|---|
| baseline | validation-baseline | production/lifecycle + numerical-model |
| avx2 | validation-avx2 | production/lifecycle + isa-avx2 + numerical-model |
| asan | validation-asan-ubsan | 适用 production/lifecycle + numerical-model + isa-avx2 |
| tsan | validation-tsan | 适用 validation/thread-smoke，排除 interposer/ISA/preload/CLI |
| cli | validation-cli-only | 禁用 GTest/Python 的 CMake 查找，独立 CLI 的 39 个黑盒方法 |
| benchmark | benchmark-model | 14 个模型 benchmark 行为合同 |
| benchmark-baseline | benchmark-model-baseline | 相同 14 个合同，排除可选 AVX2/FMA 实现 |

长上下文 numerical-model 是单独集合，不能仅运行 validation 标签就宣称覆盖。CTest 自动加入的 fixture 也计入清单。不同切片存在交集，其数量不能相加为独立功能测试数。

benchmark 合同可以执行短 timing 路径来验证报告，但不按耗时决定通过，不采集新的性能基线或设置回退阈值。实际性能采集仍按 [模型 benchmark 指南](../../tests/benchmark/models/README.md) 显式运行。

ASAN 使用 halt_on_error=1:detect_leaks=1，UBSAN/TSAN halt_on_error=1；不合并 ASAN 与 TSAN。原生 TSAN 若只因已知 unexpected memory mapping 在 main 前失败，auto launcher 会保留失败日志，再尝试进程级 setarch -R；其他错误不适用重试。没有修改系统级 ASLR。fallback 仍失败时矩阵失败，不能记为通过。

## 4. 冻结清单与失败语义

[policy.json](../../tests/fixtures/final_validation/policy.json) 固定每个 CTest 切片及两个 unittest 集合的数量和确切名称 SHA256。名称摘要为排序后的 JSON 数组（UTF8、compact separators、无额外空白）的 SHA256。

驱动要求：

1. CTest inventory 与冻结策略完全相符；disabled 或空发现失败。
2. JUnit testcase 集合与 inventory 精确相同，无重复、失败、错误、跳过或未执行。
3. unittest verbose 方法名、数量、末尾摘要和 exit code 均通过；只打印 Ran/OK 不能通过。
4. 源代码/测试/构建配置 patch、policy、模型资产、实际 binary/library、cache/compile database 身份留档。
5. 执行前后严格复核资产；变化或缺失失败。失败的命令及日志保留。

新增或删除合法测试时，先检查实际清单差异，再人工修改 policy。以下命令只输出 candidate，不会更新策略：

```bash
python3 tools/verify_ctest_report.py \
  --inventory <slice>.inventory.json --inventory-only
```

## 5. 局部运行与产物

```bash
python3 tools/run_final_validation.py --profiles baseline
python3 tools/run_final_validation.py --profiles asan tsan
python3 tools/run_final_validation.py --fresh-build
```

局部请求可返回成功，但 milestone_complete 保持 false。--fresh-build 在本次 run 的 builds/ 内建立独立构建；默认重配置并构建既有 preset 目录，依赖检查仍真实执行。

默认产物为 gitignored build/validation/final/<run-id>/：

- status.json / summary.json：请求范围、阶段、结果、计数、源码及资产一致性。
- source-before/after.patch 与 JSON：实际执行代码/测试/构建配置；文档结果可以在运行后归档。
- assets/、各 profile/、ctest/、probes/：命令、sanitizer 环境、configure/build/执行日志、发现清单和 JUnit。
- policy 和实际 cache/compile_commands/binary/library 的摘要。

CI 应完整保存失败与成功 run，按其自身策略设置 retention。通过本矩阵只证明已实现五批在所列环境的集成结果；install/export consumer、机械模块依赖检查、其他精度/家族与代表性模型性能等仍由各自专题验收。
