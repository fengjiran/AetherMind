# 上层 Runtime 组装 M5 合同与共享策略验收

- **状态**: Current
- **日期**: 2026-10-22
- **关联方案**: [10 号方案](../improvement-plan/10-upper-level-runtime-assembly.md) §4.3
- **冻结合同**: [Runtime C ABI v1](../api/runtime-c-abi.md)、[公共 C 头](../../include/c_api.h)
- **共享接口**: [model_execution_policy.h](../../include/aethermind/inference/model_execution_policy.h)

## 1. 结论与交付边界

M5 完成 C ABI v1 合同冻结和共享策略迁移。**runtime/model/session 与版本查询新函数仅声明，没有定义**；不增加 Unimplemented stub，不宣称 C 端推理可运行。M6 才交付适配实现、C 调用链及 A10/A14 动态验收，方案整体仍为 In Progress。

本轮冻结 typed opaque handles、固定 config/version、状态数值与函数签名，以及 owner/error/null/buffer 合同。Generate 直接返回 am_status_code；短 buffer 在执行前报告 N 这一安全容量上界，成功才发布实际输出。v1 结构大小/offset 固定，不再承诺同时“大小不变且尾部增字段”。

FP32 准入唯一实现迁入 inference；MakeFp32KVCacheOptions 复用已有 KVCacheRuntimeOptions 值类型。CLI 只消费公共头，没有旧声明/实现或兼容转发；M6 adapter 必须直接复用。共享映射不代替 adapter 的 Runtime/model 几何比较和所有层 compiled state dtype 检查。

另修复现有 am_error_create 在 message 构造抛出时泄漏对象的问题。PRD v1.3、API/模块设计、README、AGENTS、CHANGELOG 和方案/索引同步。

## 2. 环境与结果

环境为 Linux/x86_64、GCC `14.2.0-4ubuntu2~24.04.1`。本轮重建 `build` 的 Release 单测、`build-infer` 的独立 Release CLI（tool ON、tests/benchmarks OFF）和 `build-asan` 的 Debug instrumented core（address + undefined、frame pointers）。未以旧二进制计入通过；未重新宣称干净/离线发行验收。

HF 来源沿 [M0 recipe/checksums](../../tests/fixtures/inference_admission/README.md) 锁定的 tiny checkpoint。本轮再次通过 20 个变体的 `--check` 与临时重生成校验，模型文件仍在 gitignored 的 .models 下。F16/BF16 变体仅验证准入拒绝，不代表执行支持或生成 golden。

| 验证 | 结果 | 边界 |
|---|---:|---|
| C11/C++20 合同 | 3 passed | 新函数签名/常量/固定48/24-byte layout；opaque 类型与 C/C++ layout 一致；真实 C 调用已实现的 error ABI |
| 既有 error ABI | 5 passed | 普通/null 行为，以及对象/message 两种确定性 OOM |
| 共享策略与应用 helper | 120 passed | 原109项及新增11项共享 KV 映射合同 |
| 合并 C++ 聚焦回归 | 290 passed | 包含以上子项，另有 M0/M2、Runtime/KV/Prepare/Prefill 故障与恢复、scalar oracle/Argmax；不能与子项重复计数 |
| CLI/独立 Session | 48 passed | 4个F32 fixture × 2 preset × 6行为，完整 IDs 对照 |
| Python CLI | 38 passed | 全部16种精度拒绝、参数/阶段、容量/EOS、真实库日志与 logger OOM |
| ASAN/UBSAN 聚焦 | 126 passed | helper/共享策略120 + C contract3 + 普通 error3；无地址/UB报告 |

普通构建共 **338 项选中 C++ 测试**，无 skipped/failed。独立 Session 对照只证明接线；独立数值和阶段执行继续由 scalar oracle 承担，见 [M4 报告](upper_level_runtime_assembly_m4_validation_2026-10-22.md)。未运行全仓测试、TSAN 或性能基准，本轮不修改并发合同。

error OOM 测试使用现有 Linux/glibc malloc interposer，断言注入实际命中。4096-byte message OOM 发生于对象成功分配后，验证 `malloc_calls == free_calls + 1`；唯一没有 free 的调用是失败 malloc。用旧实现运行同一用例稳定失败（3次malloc，仅1次free），修复后通过，证明测试检出了原泄漏。

源码与符号核对确认：共享函数各只有一个 inference 定义；shared library 导出这两个 C++ 接口，但没有导出新 runtime C 函数。CMake 用 gcc `-std=c11` 编译 C contract；其 undefined symbols 仅含既有 am_error_* 和编译器 stack-check 原语，不依赖新声明的假实现。

## 3. 普通复现

从仓库根目录，先按根 README 与 fixture 说明配置 build/build-infer 并准备资产：

```bash
cmake --build build --target aethermind_unit_tests -j 4
cmake --build build-infer --target aethermind_infer -j 4
cc -std=c11 -Wall -Wextra -Werror -pedantic -Iinclude \
  -c tests/unit/c_api/test_c_api_contract.c -o /tmp/aethermind-m5-c-contract.o
c++ -std=c++20 -Wall -Wextra -Werror -pedantic -Iinclude \
  -fsyntax-only tests/unit/c_api/test_c_api_contract.cpp

TMPDIR=/tmp ./build/tests/unit/aethermind_unit_tests \
  --gtest_filter='CAbiContract.CTranslationUnitCallsExistingErrorAbi:AMError.MessageAllocationFailureReleasesObject:Fp32KVCacheOptions.MapsLogicalGeometryWithoutAllocatingStorage'
TMPDIR=/tmp ./build/tests/unit/aethermind_unit_tests --gtest_filter='CAbiContract.*:AMError.*:Fp32KVCacheOptions.*:InferenceRunOptions.*:InferenceRuntimeOptions.*:InferenceCpuPreset.*:Fp32ExecutionInputs.*:AdmissionFixtures/Fp32HfAdmission.*:AdmissionFixtures/Fp32HfCompile.*:RequiredRoles/Fp32RequiredRoleAdmission.*:AdmissionFixtures/InferenceAdmissionFixture.*:InferenceAdmissionFixtures.*:HfPresets/InferenceRunnerEquivalence.*:InferenceRunnerSuccess.*:InferenceRunnerFailure.*:KvPlanes/RunnerKvAllocationFailure.*:AdmissionFixtures/RunnerFp32Admission.*:PreparationPaths/TinyLlamaExecution.*:TinyWorkspaceSession.*:Prefill.*:RuntimeBuilderTryBuild.*:*/KVCacheManagerInitAllocationFailure.*:KVCacheManager.InitDiagnosticAllocationFailureLeavesEmptyStateAndCanRetry:KVCacheManager.BufferMetadataFailureReturnsResourceExhaustedAndReleasesPayload:PreparationPaths/LoadAndPrepareExecutableModelPath.*:CPUKernelArgmax.*'

AETHERMIND_INFER_BINARY="$PWD/build-infer/apps/inference/aethermind_infer" TMPDIR=/tmp \
  ./build/tests/unit/aethermind_unit_tests --gtest_filter='CliMatrix/InferenceCliIntegration.*'
python3 tests/unit/apps/inference/test_inference_cli.py \
  --infer "$PWD/build-infer/apps/inference/aethermind_infer" -v
python3 tools/generate_fp32_admission_hf_fixtures.py \
  --source .models/tiny-random-LlamaForCausalLM \
  --recipe tests/fixtures/inference_admission/recipe.json \
  --output-dir .models/inference-admission \
  --check tests/fixtures/inference_admission/checksums.json
```

## 4. Sanitizer 与泄漏基线

既有全量 ASAN 单测与 malloc interposer 的启动冲突沿 [M4 记录](upper_level_runtime_assembly_m4_validation_2026-10-22.md) 保留。本轮独立 harness 对现有 interposer 源设置其已有的 `AETHERMIND_TEST_MALLOC_INTERPOSER_DISABLED=1`，只链接 inert stubs，过滤列表不选择两个 OOM 用例；它们已在普通构建取得证据。不把不可注入/skipped 当作 OOM 通过，不改变生产行为或全量测试策略。

以下命令链接本轮重建的 instrumented core：

```bash
M5_ROOT="$PWD"
cmake --build build-asan --target AetherMind -j 4
for m5_source in apps/inference/inference_runner.cpp \
  tests/unit/apps/inference/test_inference_options.cpp \
  tests/unit/apps/inference/test_fp32_execution_inputs.cpp \
  tests/unit/inference/test_model_execution_policy.cpp \
  tests/unit/c_api/test_c_api_error.cpp \
  tests/unit/c_api/test_c_api_contract.cpp \
  tests/unit/inference/test_malloc_interposer.cpp
do
  m5_object="/tmp/aethermind-m5-asan-${m5_source//\//_}.o"
  c++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
    -DSPDLOG_USE_STD_FORMAT -DAETHERMIND_TEST_MALLOC_INTERPOSER_DISABLED=1 \
    -DAETHERMIND_TEST_MODELS_DIR="\"$M5_ROOT/.models\"" \
    -Iinclude -Iapps/inference -Itests/unit -Ibuild-asan/_deps/spdlog-src/include \
    -c "$m5_source" -o "$m5_object" || exit "$?"
done
cc -std=c11 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -Iinclude -c tests/unit/c_api/test_c_api_contract.c \
  -o /tmp/aethermind-m5-asan-c-contract.o
c++ -fsanitize=address,undefined /tmp/aethermind-m5-asan-*.o \
  build-asan/src/libAetherMind.so build-asan/_deps/spdlog-build/libspdlog.a \
  -lgtest_main -lgtest -pthread -Wl,-rpath,"$M5_ROOT/build-asan/src" \
  -o /tmp/aethermind-m5-asan-focused

M5_ASAN_FILTER='CAbiContract.*:AMError.CreateReadDestroy:AMError.NullMessageBecomesEmptyString:AMError.NullHandleFallbacks:Fp32KVCacheOptions.*:InferenceRunOptions.*:InferenceRuntimeOptions.*:InferenceCpuPreset.*:Fp32ExecutionInputs.*:AdmissionFixtures/Fp32HfAdmission.*:AdmissionFixtures/Fp32HfCompile.*:RequiredRoles/Fp32RequiredRoleAdmission.*'
TMPDIR=/tmp ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  /tmp/aethermind-m5-asan-focused --gtest_filter="$M5_ASAN_FILTER"
```

在无 ptrace 的环境分别启用 LSAN 执行零用例与同一126项。两次均 exit 1，分别有0/126 tests passed 后报告 **1384 bytes / 17 allocations**。Direct/Indirect allocation block 去除PC地址后，归一化栈多重集合完全相同，仍定位未修改的 GlobalFunctionTable 静态注册。没有新增报告，整体零泄漏门禁未清零；detect_leaks=0 不作泄漏验证，未加入 suppressions。

```bash
TMPDIR=/tmp ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  /tmp/aethermind-m5-asan-focused --gtest_filter=NoSuchM5Test \
  > /tmp/aethermind-m5-lsan-baseline.log 2>&1
TMPDIR=/tmp ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  /tmp/aethermind-m5-asan-focused --gtest_filter="$M5_ASAN_FILTER" \
  > /tmp/aethermind-m5-lsan-tests.log 2>&1
```

预期非零后仍须独立运行第二条，不在遇错退出的批处理里将它跳过。AM_CHECK/noexcept 导致的终止不由 wrapper catch 恢复，本次不声明全链 recoverable OOM。

## 5. 源码身份、日志与检查

本轮工作树包含 M0–M5 未提交改动，HEAD 不是受测快照；关键文件 SHA-256 如下，不以局部 hashes 代替完整构建归档：

| 文件 | SHA-256 |
|---|---|
| include/c_api.h | `87d56938703b8ccd3c3d19256a9d6f4200da8e3d0a88a996e9e69f918bd94a04` |
| src/c_api_error.cpp | `5f76ba13cf4b3e2698242c182223a88b524de9cbb0ab75b97f837bcfea30091c` |
| include/aethermind/inference/model_execution_policy.h | `daf419a8cfefd7648de639bc40253837ac9485d581a32460d4ebc5653065596f` |
| src/inference/model_execution_policy.cpp | `97d3cbf7f4c93c4a57d66ab882faa4898b84254e1abff86e33d5ab25bd1b83b2` |
| apps/inference/inference_runner.cpp | `6abfa75ab3ee41387b4e522aa7ab9276454d119a2f5e6c8a497df32f55e7b188` |
| tests/unit/c_api/test_c_api_contract.c | `e507615728e97aa7f2caec78e9d8d8778f02315dfd3f059fafce7f5cb7045d64` |
| tests/unit/c_api/test_c_api_contract.cpp | `be05a9cfc5aded2517d8f8c9cc794445fc0dd883a6a2c8da8feb164c4a6cba4a` |
| tests/unit/c_api/test_c_api_error.cpp | `7b8bdb7a8cb53ae1cb4cfc791dc3d04c01d741ce9fdaf9a5c501d00d749d5823` |
| tests/unit/inference/test_model_execution_policy.cpp | `3b32d8e1d7660ee7cac939219d9cba77c7a317cd2415f34c5c13d6b4c620c765` |

本机临时日志为 `/tmp/aethermind-m5-regression.log`、`/tmp/aethermind-m5-cli-equivalence.log`、`/tmp/aethermind-m5-cli-tests.log`、`/tmp/aethermind-m5-error-negative.log`、`/tmp/aethermind-m5-asan-tests.log` 与上述两份 LSAN 日志；不视为长期 artifact 归档，复现以源码、fixture、配置和命令为准。

源码与合同完成只读复核，无阻塞问题。clang-format、C11/C++20 严格编译、git diff --check、文档代码块语法检查通过。文档检查 0 problem、29项既有 warning；不替代动态 ABI 验收。
