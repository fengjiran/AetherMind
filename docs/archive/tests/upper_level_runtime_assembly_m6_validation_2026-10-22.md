# 上层 Runtime 组装 M6 C ABI 验收

- **状态**: Current
- **日期**: 2026-10-22
- **关联方案**: [10 号方案](../improvement-plan/10-upper-level-runtime-assembly.md) §4.3、A10/A13/A14
- **合同/实现**: [Runtime C ABI v1](../api/runtime-c-abi.md)、[c_api_runtime.cpp](../../src/c_api_runtime.cpp)
- **前序证据**: [M4 C++ 验收](upper_level_runtime_assembly_m4_validation_2026-10-22.md)、[M5 合同与策略](upper_level_runtime_assembly_m5_validation_2026-10-22.md)

## 1. 结论与范围

M6 完成冻结 C ABI 的正式实现和纯 C 调用验收，10 号方案在限定范围内为 Implemented。**100 项新增 C ABI、290 项既有聚焦回归、48 项 CLI/Session 对照，共 438 项测试通过**，无 skipped/failed；38 项 Python CLI 通过。87 项普通 C ABI ASAN/UBSAN 用例通过，13 项真实 malloc interposer 故障用例在普通构建中独立验证。

LSAN 零用例与 87 项运行均报告既有的 **1384 bytes / 17 allocations**，归一化栈多重集合相同，未增加报告；整体零泄漏门禁仍未通过。不据此声明全部 PRD hard gates、7B 性能、量化/其他家族/GPU、离线发行、install/export 或其他平台已完成。本轮未运行全仓测试、TSAN 或性能基准；并发合同不变，宿主继续串行访问共享 Runtime。

| 维度 | 本轮证据 |
|---|---|
| 环境 | Linux/x86_64，GCC `14.2.0-4ubuntu2~24.04.1`，C11 / C++20 |
| ABI | v1.0 `0x00010000`；8 个版本/生命周期/生成符号实际导出；原固定 48/24-byte 配置与状态值不变 |
| 可执行范围 | CPU、Llama Dense、scalar FP32、O1/plain 或 O2/packed、同步 greedy、单请求、静态 KV、Token IDs I/O |
| 构建 | Release 的 build 单测与 build-infer 独立工具；Debug address/undefined 的 build-asan core，均重建 |
| Fixture | 锁定 M0 tiny HF 的20个变体再次通过 checksums 与临时重生成；4个F32成功，16个准入拒绝 |
| 期望来源 | 独立配置的公开 C++ Load→Compile→TryBuild→Prepare→Create→Generate 链，不调用 adapter 的 preset/几何 helper，不构造 HF golden |

工作树包含 M0–M6 未提交改动，HEAD 本身不代表受测快照。关键文件身份见 §6；fixture 来源/版本及 base hashes 沿 [M0 说明](../../tests/fixtures/inference_admission/README.md) 固定。

## 2. 实现与门禁映射

| 合同 | 实现及证据 |
|---|---|
| 稳定 Runtime 与 owner 链 | RuntimeState 在 Prepare 前安置于稳定 heap 地址；ModelState 强持有 Runtime，am_session 强持有 ModelState；逆序销毁 Session→ExecutableModel→Runtime，无全局 registry/循环引用 |
| C 运行链 | [纯 C consumer](../../tests/unit/c_api/test_c_api_runtime_consumer.c) 实际调用 version/create/load/create/generate/destroy；48项矩阵逐4个F32 fixture/两preset比较三token、重复请求、精确capacity、N=0、EOS、外部owner提前释放 |
| 配置/版本/null | 先读取8-byte prefix并拒绝错误size/version，再复制完整配置；guard page 测试证明不多读。固定宽度→size_t、N输出字节、KV行字节/64-byte对齐溢出均检查；NULL销毁为no-op |
| 模型准入 | CLI/C adapter 共用唯一 inference FP32策略与KV映射；16变体逐一拒绝并可重试有效模型。合法FP32 payload仅改变家族/RoPE时，Loader先成功、Compile拒绝，C侧保留InvalidArgument且Runtime继续可用 |
| KV跨对象 | Compile后、Prepare前唯一私有helper比较layers/heads/逻辑head_dim/capacity；逐根与alias扫描所有层K/V dtype，拒绝后层/alias异常、缺失/重复根；另测BF16 Runtime layout与F32 compiled state不匹配、缺manager |
| 共享CPU preset | 新公共 CpuInferencePreset/MakeCpuCompileOptions 归inference，CLI/C adapter共用；应用旧类型/实现删除，无兼容转发。编译选项仍由lowering决定物理packed消费 |
| Buffer与N=0 | short buffer返回OutOfRange/N上界、先于token语义校验；实际request分配故障在随后的足容量调用才命中，证明query不执行。成功才发布完整tokens；其他错误长度零、buffer不变；N=0仍验证prompt/EOS/context，无KV预约 |
| 失败/异常 | 13项确定性注入实际命中：Runtime注册、K/V两plane及metadata、Model加载、O2 packed Prepare、Session/request、诊断object/message；检查状态、清理、不发布部分对象与重试。bad_alloc实际触发，catch-all映射另经源码审查；不宣称AM_CHECK/noexcept终止可恢复 |
| 日志与构建 | C adapter不改宿主logger；CLI仍先安装stderr logger。独立tool OFF tests/benchmarks下正式编译无测试宏/include/符号、无GTest/benchmark链接；新C ABI在shared library中，外部C程序不依赖GTest |

以上闭合 A10/A14 与 A13 第二生产消费者的范围；A1–A9/A11/A12 的既有资产按 M4/M5 记录回归。C ABI先分配Runtime KV，模型拒绝不撤销既有storage；不能套用CLI“非法模型拒绝前零KV分配”的结论。

## 3. 普通构建与复现

从仓库根目录按 README 配置 build/build-infer 并准备 M0 资产后：

```bash
cmake --build build --target aethermind_unit_tests -j 4
cmake --build build-infer --target aethermind_infer -j 4
TMPDIR=/tmp ./build/tests/unit/aethermind_unit_tests \
  --gtest_filter='CAbiRuntime.RejectsLayoutOverflowBeforeAllocatingPlanes:HfPresets/CAbiRuntimeIntegration.RetainedOwnersSurviveExternalRuntimeAndModelRelease/F32ExplicitO1Plain'
TMPDIR=/tmp ./build/tests/unit/aethermind_unit_tests \
  --gtest_filter='CAbiRuntime.*:CAbiModel.*:CAbiSession.*:CAbiGenerate.*:CAbiKVValidation.*:KvPlanes/CAbiRuntimeAllocationFailure.*:HfPresets/CAbiRuntimeIntegration.*'
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

新增C ABI100项由52项边界/失败和48项矩阵组成；13项注入包含参数化的首/第二plane与metadata。C11编译合同和AMError既有5项计入290项回归，不重复计数。源文件改动仅影响编排/配置，没有新增kernel、packing布局算法或执行并发能力。

## 4. 独立 C 程序

[smoke main](../../tests/unit/c_api/test_c_api_runtime_smoke.c) 使用 M0 recipe 的2层/4KV heads/逻辑head_dim4、capacity8、prompt `[7]`、N=3。它在模型加载后释放外部Runtime handle，在Session创建后释放外部Model handle，再执行两次Generate并检查一致。几何常量只属于此固定fixture测试，adapter不含tiny模型默认值。

```bash
cc -std=c11 -Wall -Wextra -Werror -pedantic -Iinclude -Itests/unit/c_api \
  tests/unit/c_api/test_c_api_runtime_consumer.c \
  tests/unit/c_api/test_c_api_runtime_smoke.c \
  -L"$PWD/build/src" -Wl,-rpath,"$PWD/build/src" -lAetherMind \
  -o /tmp/aethermind-m6-c-smoke
/tmp/aethermind-m6-c-smoke .models/inference-admission/f32-explicit o1-plain
/tmp/aethermind-m6-c-smoke .models/inference-admission/f32-explicit o2-packed
```

两次成功，当前固定输入均输出 `[9555,23124,30954]`；这是本机观测，不是跨preset/环境的数值golden或性能承诺。ldd显示程序只直接消费AetherMind及系统运行库，没有GTest/benchmark；由cc编译链接，不需要C++测试main。

## 5. ASAN/UBSAN 与 LSAN

现有全量单测与glibc malloc interposer在ASAN下的启动冲突沿M4/M5记录保留。本轮独立harness编译当前C API测试/consumer，链接本轮instrumented shared library；对既有interposer源使用其已有disabled宏，仅链接inert stubs。过滤13项 `AllocationFailure` 用例；剩余39项普通边界与48项矩阵共87项通过，无selected skip或地址/UB报告。注入用例的证据来自上述普通构建，不能把inactive/skip计为通过。

```bash
M6_ROOT="$PWD"
cmake --build build-asan --target AetherMind -j 4
for m6_source in tests/unit/c_api/test_c_api_runtime.cpp \
  tests/unit/c_api/test_c_api_runtime_integration.cpp \
  tests/unit/inference/test_malloc_interposer.cpp
do
  m6_object="/tmp/aethermind-m6-asan-${m6_source//\//_}.o"
  c++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
    -DSPDLOG_USE_STD_FORMAT -DAETHERMIND_TEST_MALLOC_INTERPOSER_DISABLED=1 \
    -DAETHERMIND_TEST_MODELS_DIR="\"$M6_ROOT/.models\"" \
    -Iinclude -Isrc -Itests/unit -Ibuild-asan/_deps/spdlog-src/include \
    -c "$m6_source" -o "$m6_object" || exit "$?"
done
cc -std=c11 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -Iinclude -Itests/unit/c_api -c tests/unit/c_api/test_c_api_runtime_consumer.c \
  -o /tmp/aethermind-m6-asan-consumer.o
c++ -fsanitize=address,undefined /tmp/aethermind-m6-asan-*.o \
  build-asan/src/libAetherMind.so build-asan/_deps/spdlog-build/libspdlog.a \
  -lgtest_main -lgtest -pthread -Wl,-rpath,"$M6_ROOT/build-asan/src" \
  -o /tmp/aethermind-m6-asan-focused
TMPDIR=/tmp ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  /tmp/aethermind-m6-asan-focused --gtest_filter='*-*AllocationFailure*'
```

独立纯C main也以address/undefined编译、链接instrumented library，在detect_leaks=0下分别通过两种preset；方法为§4的cc命令增加上述sanitizer flags，并将library路径换为build-asan/src。

在无ptrace环境分别执行零用例与87项LSAN，两次进程均exit1、tests分别0/87 passed后报告1384 bytes/17 allocations。Direct/Indirect allocation block仅移除PC地址，比较归一化栈多重集合完全一致，仍定位未修改的GlobalFunctionTable静态注册。没有新增报告、没有suppression，整体零泄漏门禁未通过。

```bash
TMPDIR=/tmp ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  /tmp/aethermind-m6-asan-focused --gtest_filter=NoSuchM6Test \
  > /tmp/aethermind-m6-lsan-baseline.log 2>&1
TMPDIR=/tmp ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  /tmp/aethermind-m6-asan-focused --gtest_filter='*-*AllocationFailure*' \
  > /tmp/aethermind-m6-lsan-tests.log 2>&1
```

预期非零后仍独立执行第二条；detect_leaks=0只是地址/UB验证。底层AM_CHECK/noexcept终止、全部产品零分配/并发/性能与发行质量门禁继续独立处理。

## 6. 源码身份与原始日志

关键文件SHA-256如下，不以局部hash代替完整构建归档：

| 文件 | SHA-256 |
|---|---|
| include/c_api.h | `5c1e9367d728cadc30a2b1a845841bf7b64f8a4738e67ab4cc1c71d806a39036` |
| src/c_api_runtime.cpp | `6a03c7e9c63dbe88aee8e5f0e17395cb04619a616d1e383c97114c56a63bc90a` |
| src/c_api_runtime_internal.h | `297d36b87fb7350ee9e08f8141a21a8b23deb20c902da42980facfea44aedc57` |
| include/aethermind/inference/cpu_inference_options.h | `03433239d40854cfdb236ac5ec677925d3b900349e803654363d68b710a888ff` |
| src/inference/cpu_inference_options.cpp | `fa3c290dc1d1f53c4b9ff6f7a7db68fc26a15876be2ae49138281ee4879c34dc` |
| apps/inference/inference_runner.cpp | `4fa47fe5547738633758cf90cf533756da0387724173ddeec951e04484b7a01e` |
| tests/unit/c_api/test_c_api_runtime.cpp | `39251d2743d9ef4f88880cac2546e0387f7999f0e304193367d34d9906f60885` |
| tests/unit/c_api/test_c_api_runtime_integration.cpp | `23a54572d2e57680cf2e733ba3f4d39e7d2eb2755d97b04abe8fbbd0560f0113` |
| tests/unit/c_api/test_c_api_runtime_consumer.c | `1f58e5dc5a34b347fed3f984abe717c80f1c85630725ed56f0f7179ee2552115` |
| tests/unit/c_api/test_c_api_runtime_smoke.c | `c64dabfc837eee24e0f1ba07ca4646ff052fabccfa9ac6e9ef271ae7ffb257c7` |

本机临时原始日志为 `/tmp/aethermind-m6-c-api-tests.log`、`/tmp/aethermind-m6-regression.log`、`/tmp/aethermind-m6-cli-equivalence.log`、`/tmp/aethermind-m6-cli-tests.log`、`/tmp/aethermind-m6-asan-tests.log` 及上述两份LSAN日志。临时文件不视为长期artifact，复现以源码/fixture/构建配置/命令为准。

源码与合同经独立只读审查；修正审查发现的极大head_dim行布局Overflow错误码问题，未扩展修改KVManager。clang-format、C11严格编译、git diff --check、文档代码块语法检查通过；文档检查为0 problem、29项既有warning。
