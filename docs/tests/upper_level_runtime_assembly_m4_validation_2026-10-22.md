# 上层 Runtime 组装 M4 C++ 首版验收

- **状态**: Current
- **日期**: 2026-10-22
- **关联方案**: [10 上层 Runtime 组装与推理入口](../improvement-plan/10-upper-level-runtime-assembly.md) §4.2、§6
- **产品边界**: [PRD](../products/aethermind_prd.md)
- **关联代码**: [CLI](../../apps/inference/inference_main.cpp)、[runner](../../apps/inference/inference_runner.cpp)、[CLI/Session 对照](../../tests/unit/apps/inference/test_inference_cli_integration.cpp)、[Python 黑盒测试](../../tests/unit/apps/inference/test_inference_cli.py)

## 1. 结论与范围

M4 完成 C++ 首版验收收口。48 项 CLI/独立 Session 对照、271 项聚焦 C++ 回归、38 项 Python CLI 测试和 20 项转换器测试通过。ASAN/UBSAN 的 48 项对照与 36 项普通 CLI 测试通过；LSAN 的零用例与 48 项用例均报告既有的 1384 bytes / 17 allocations，归一化栈集合相同，未增加报告，整体零泄漏门禁仍未通过。

10 号方案仍为 In Progress：M0–M4 已完成，M5/M6 的 C ABI 合同、适配实现、共享 FP32 准入迁移与独立 C 调用验收未完成。M4 不代表全部 PRD hard gates 达成。

| 维度 | 本轮证据范围 |
|---|---|
| 环境 | Linux / x86_64；GCC `14.2.0-4ubuntu2~24.04.1`；C++20 |
| 普通构建 | `build`：Release、tests ON、benchmarks ON、tool OFF；`build-infer`：Release、tool ON、tests/benchmarks OFF |
| sanitizer | `build-asan`：Debug、address + undefined、frame pointers；测试 harness 单独以 O1 编译，链接同一 instrumented shared library |
| 生产链 | 一次 Load → 参数/FP32 准入 → Compile → TryBuild → PrepareExecutableModel → Create → Generate |
| 可执行范围 | CPU、同步、单请求、Llama Dense、scalar FP32、greedy Argmax、静态 KV、Token IDs I/O |
| preset | 显式 O1/plain、O2/packed；分别与同 preset 的直接 Session 比较，不要求跨 preset 逐位相同，不承诺性能提升 |
| HF 输入 | 锁定 tiny checkpoint 衍生的 4 个 F32 成功变体及 16 个 loader 接受但应用拒绝的精度变体 |
| 未取得的证据 | HF 独立数值 golden、7B 性能、其他模型家族、FP16/BF16/量化执行、GPU、离线发行、install/export、C ABI 与全部 PRD 验收 |

本轮在含 M0–M4 未提交改动的工作树运行；HEAD 本身不代表受测源码快照。关键源码身份见 §6，fixture 来源见 §2。

## 2. Fixture 与期望值来源

来源与生成规则由 [recipe](../../tests/fixtures/inference_admission/recipe.json)、[checksums](../../tests/fixtures/inference_admission/checksums.json) 和 [fixture 说明](../../tests/fixtures/inference_admission/README.md) 固定。远端仓库为 `peft-internal-testing/tiny-random-LlamaForCausalLM`，revision 为 `d0b81c5e23f2d61bb7a5f430a8cd6f3e711c0cb3`。

| Base 文件 | 字节数 | SHA-256 |
|---|---:|---|
| config.json | 720 | `ea68007f47752b68f0f58e5fc2d51fa380bb730af51faff8c1df9fccbe608bc2` |
| model.safetensors | 4131280 | `49c20f32c6c597480fcaec5df2f86c645eabea765cbea1e67886dbae45e5c992` |

本轮再次执行 `--check`：20 个变体的来源、脚本/recipe 身份、文件、逐 tensor metadata/payload 校验值及临时重生成结果一致。生成模型位于 gitignored 的 `.models/inference-admission/`，不属于仓库内置模型。

成功矩阵为 `f32-missing`、`f32-auto`、`f32-explicit`、`f32-tied` × 两种 preset × 六个行为，共 48 项：

| 行为 | 固定请求与检查 |
|---|---|
| 三个新 token | prompt `[7]`、capacity 8、N=3、无 EOS；完整新 IDs 与直接 Session 一致 |
| 独立进程重复 | 同一请求启动两次 CLI，分别与同一个直接 Session 结果比较 |
| 精确容量 | prompt `[7,8]`、capacity 4、N=3，验证 `2 + 3 - 1 == 4` |
| 零生成 | prompt `[7,8]`、capacity 1、N=0，双方成功返回空数组 |
| EOS 包含及停止 | 将同请求直接 Session 的第一个预测设为显式 EOS，双方仅返回该 token；不从模型配置猜测 EOS |
| 超容量拒绝 | prompt `[7,8]`、capacity 3、N=3；直接 Session 为 OutOfRange，CLI 为参数错误退出 2，stdout 为空 |

期望路径仅使用公开库 API，独立构造 RuntimeOptions、编译选项与 GenerationConfig，不调用 RunInference、应用 preset/helper 或 CLI parser。测试通过无 shell 的 `posix_spawn` 启动正式 executable，独立捕获两个 stream，超时杀死并回收子进程，临时文件按 RAII 清理。测试要求显式提供 `AETHERMIND_INFER_BINARY`；缺模型、缺 executable 或非 Linux 平台均显式失败，不跳过。

这 48 项证明装配、配置传递与输出等价性。独立数值/阶段证据继续由 synthetic TinyLlama scalar oracle 测试承担：O1Plain/O2Packed 分别实际执行 Prefill 与两次 Decode，检查 logits、Token IDs、KV 与 commit watermark，并对照 Generate；不将该 synthetic checkpoint 的结果当作 HF golden。

## 3. 验收门禁映射

| 门禁 | 结果与证据 |
|---|---|
| A1 独立构建 | M3 干净 build-infer 已通过；M4 重建并复核 tool ON / tests OFF / benchmarks OFF。无 GTest、google_benchmark、tests include 或测试 seam 符号，保留正式 spdlog。默认 tool OFF、benchmark ON 可构建的回归沿用 M3，M4 未修改构建规则。 |
| A2/A3 builder 与一次准备 | 复核正式 runner 的 Load/Compile/TryBuild/Prepare 各一次，Compile 先于 KV 分配，配置来自当前模型，kBoth plan 进入既有 Generate；Prepare 后 Runtime 保持地址稳定。 |
| A4/A5 阶段、数值与接线 | 48 项同 HF/preset 的完整 IDs 对照；4 项 TinyLlama scalar-oracle 回归；CPUKernelArgmax 覆盖相等最大值选择最低 index。未新增 logits/debug 公共接口。 |
| A6 确定性与复用 | 8 组重复 CLI 进程对照；底层重复 Generate 验证 reservation 释放与请求状态隔离。 |
| A7 输入、日志与进程边界 | 38 项 Python 测试（部分矩阵以 subTests 展开）：严格数字/CSV/flag、空 prompt、词表/EOS、路径含空格、非法家族/RoPE/activation、16 种精度拒绝、stdout/stderr/退出码；真实库 warning 与 logger OOM 另有确定性注入。 |
| A8 容量与零值 | 直接 Session/CLI 矩阵覆盖精确容量、超限、N=0 和 EOS；Python 与底层回归覆盖溢出、模型 context 上限及零生成仍校验 prompt/EOS。 |
| A9 失败与所有权 | 271 项聚焦回归含 Runtime 注册、两块 KV plane/metadata、Prepare/执行、request buffer/binding/activation/workspace 失败及重试；runner 保留原 StatusCode，释放部分资源。ASAN/UBSAN 无地址/UB 报告；LSAN 仅取得既有基线未增加报告的证据，见 §5。 |
| A11 文档与范围 | README、AGENTS、公共 API 参考、方案与索引同步；区分已交付 C++ FP32 slice 和 C ABI/量化/其他产品目标。 |
| A12 分配前拒绝 | runner 用合法 FP32 payload 的未知家族/RoPE 触发 Compile 失败，计数 allocator 为零 KV 分配；CLI 验证相应阶段与退出 1。activation 拒绝发生于 Load，分别记录。 |
| A13 C++ 准入部分 | 20 个真实 HF 变体先证明 Loader 成功；4 个 F32 进入两种 preset，其他 16 个由模型准入返回 Unimplemented，Compile/TryBuild 前拒绝且零 KV 分配；逐角色 synthetic 用例继续检查完整遍历。 |
| A10/A14、A13 P2 部分 | 尚未实施。C 调用、ABI version、opaque handle/输出/error 所有权、Runtime/model KV 跨对象校验及共享准入迁移，由 M5/M6 独立验收。 |

日志故障测试全部留在测试侧：

- [THP shim](../../tests/unit/apps/inference/fixtures/fail_hugepage_advice.c) 仅拒绝 MADV_HUGEPAGE，17-token 请求触发真实 shared-library CPUAllocator warning，stdout 仍是单行 JSON。
- [logger OOM shim](../../tests/unit/apps/inference/fixtures/fail_logger_allocation.c) 在 static initialization 完成、进入 main 前只拒绝下一次 malloc；使用不存在的模型目录，先确认正常路径报 Load 失败，再三次独立验证 logger 初始化先失败、退出 1、stdout 为空且诊断为 RESOURCE_EXHAUSTED。后续分配不受限，未给工具新增故障开关。

## 4. 普通构建与复现

从仓库根目录运行，先按 fixture 说明准备 base/生成变体。工具构建不需要 GTest；C++ 测试仍使用已安装的 GTest。

```bash
python3 tools/generate_fp32_admission_hf_fixtures.py \
  --source .models/tiny-random-LlamaForCausalLM \
  --recipe tests/fixtures/inference_admission/recipe.json \
  --output-dir .models/inference-admission \
  --check tests/fixtures/inference_admission/checksums.json

cmake -S . -B build-infer -DBUILD_INFERENCE_TOOL=ON \
  -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-infer --target aethermind_infer -j
cmake --build build --target aethermind_unit_tests -j

AETHERMIND_INFER_BINARY="$PWD/build-infer/apps/inference/aethermind_infer" TMPDIR=/tmp \
  ./build/tests/unit/aethermind_unit_tests \
  --gtest_filter='CliMatrix/InferenceCliIntegration.MatchesDirectSessionForThreeNewTokens/F32MissingO1Plain'
AETHERMIND_INFER_BINARY="$PWD/build-infer/apps/inference/aethermind_infer" TMPDIR=/tmp \
  ./build/tests/unit/aethermind_unit_tests --gtest_filter='CliMatrix/InferenceCliIntegration.*'

python3 tests/unit/apps/inference/test_inference_cli.py \
  --infer "$PWD/build-infer/apps/inference/aethermind_infer" -v
python3 -m unittest discover -s tests/unit/apps/inference \
  -p 'test_generate_fp32_admission_hf_fixtures.py' -v

TMPDIR=/tmp ./build/tests/unit/aethermind_unit_tests --gtest_filter='InferenceRunOptions.*:InferenceRuntimeOptions.*:InferenceCpuPreset.*:Fp32ExecutionInputs.*:AdmissionFixtures/Fp32HfAdmission.*:AdmissionFixtures/Fp32HfCompile.*:RequiredRoles/Fp32RequiredRoleAdmission.*:AdmissionFixtures/InferenceAdmissionFixture.*:InferenceAdmissionFixtures.*:HfPresets/InferenceRunnerEquivalence.*:InferenceRunnerSuccess.*:InferenceRunnerFailure.*:KvPlanes/RunnerKvAllocationFailure.*:AdmissionFixtures/RunnerFp32Admission.*:PreparationPaths/TinyLlamaExecution.*:TinyWorkspaceSession.*:Prefill.*:RuntimeBuilderTryBuild.*:*/KVCacheManagerInitAllocationFailure.*:KVCacheManager.InitDiagnosticAllocationFailureLeavesEmptyStateAndCanRetry:KVCacheManager.BufferMetadataFailureReturnsResourceExhaustedAndReleasesPayload:PreparationPaths/LoadAndPrepareExecutableModelPath.*:CPUKernelArgmax.*'
```

实际 build-infer 首次配置复用了 `-DFETCHCONTENT_SOURCE_DIR_SPDLOG="$PWD/build/_deps/spdlog-src"` 的本地固定版本 source；未验证离线依赖交付。M4 重建沿用该配置。

| 本轮结果 | 数量 | 原始日志（本机临时文件） |
|---|---:|---|
| CLI/Session 矩阵 | 48 passed | `/tmp/aethermind-m4-cli-equivalence.log` |
| C++ 聚焦回归 | 271 passed | `/tmp/aethermind-m4-regression.log` |
| Python CLI | 38 passed | `/tmp/aethermind-m4-cli-tests.log` |
| Python fixture converter | 20 passed | `/tmp/aethermind-m4-generator-tests.log` |
| ASAN/UBSAN CLI/Session | 48 passed | `/tmp/aethermind-m4-asan-integration.log` |
| ASAN/UBSAN 普通 Python CLI | 36 passed | `/tmp/aethermind-m4-cli-asan-tests.log` |

以上选中的用例无 skipped 或 failed；不是全仓测试结果。临时日志不作为长期存档，复现以源码身份、fixture 校验值和命令为准。

## 5. Sanitizer 复现与既有基线

现有全量单测程序链接全局 malloc interposer，ASAN 下在列举测试前退出 139。M2 的相同 object/link 参数 A/B 已确认增删该 interposer 决定启动是否失败，详见 [fixture sanitizer 说明](../../tests/fixtures/inference_admission/README.md)。M4 独立 harness 仅编译 CLI 对照源，排除 interposer，仍使用当前 instrumented AetherMind shared library；正式 sanitizer CLI 也没有该测试 interposer。

```bash
M4_ROOT="$PWD"
cmake -S . -B build-asan -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF \
  -DBUILD_INFERENCE_TOOL=ON -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer'
cmake --build build-asan --target aethermind_infer -j
c++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -DSPDLOG_USE_STD_FORMAT -DAETHERMIND_TEST_MODELS_DIR="\"$M4_ROOT/.models\"" \
  -Iinclude -Itests/unit -Ibuild-asan/_deps/spdlog-src/include \
  tests/unit/apps/inference/test_inference_cli_integration.cpp \
  build-asan/src/libAetherMind.so build-asan/_deps/spdlog-build/libspdlog.a \
  -lgtest_main -lgtest -pthread -Wl,-rpath,"$M4_ROOT/build-asan/src" \
  -o /tmp/aethermind-m4-asan-integration

AETHERMIND_INFER_BINARY="$PWD/build-asan/apps/inference/aethermind_infer" TMPDIR=/tmp \
  ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  /tmp/aethermind-m4-asan-integration --gtest_filter='CliMatrix/InferenceCliIntegration.*'
```

普通 Python CLI 的 sanitizer 运行排除两个外部 preload 故障用例；它们已在普通构建取得独立证据。其余 36 项使用 sanitizer executable：

```bash
AETHERMIND_INFER_BINARY="$PWD/build-asan/apps/inference/aethermind_infer" \
  ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  python3 - <<'PY'
import importlib.util
import sys
import unittest
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location(
    "m4_cli", "tests/unit/apps/inference/test_inference_cli.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
excluded = {
    "test_logger_initialization_allocation_failure_precedes_model_loading",
    "test_real_library_warning_uses_stderr_without_polluting_json",
}
names = unittest.defaultTestLoader.getTestCaseNames(module.InferenceCliTests)
suite = unittest.TestSuite(module.InferenceCliTests(name) for name in names if name not in excluded)
assert suite.countTestCases() == 36
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
PY
```

`detect_leaks=0` 仅验证地址/UB，不证明无泄漏。LSAN 在 ptrace 沙箱内无法完成，故在沙箱外分别运行零用例与同一 harness 的 48 项；子进程使用 Release CLI，以便独立观察 instrumented 父进程的泄漏基线：

```bash
AETHERMIND_INFER_BINARY="$PWD/build-infer/apps/inference/aethermind_infer" TMPDIR=/tmp \
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  /tmp/aethermind-m4-asan-integration --gtest_filter=NoSuchM4Test \
  > /tmp/aethermind-m4-lsan-baseline.log 2>&1
AETHERMIND_INFER_BINARY="$PWD/build-infer/apps/inference/aethermind_infer" TMPDIR=/tmp \
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  /tmp/aethermind-m4-asan-integration --gtest_filter='CliMatrix/InferenceCliIntegration.*' \
  > /tmp/aethermind-m4-lsan-tests.log 2>&1
```

两次均 exit 1，分别显示 0 和 48 tests passed，随后均报告 1384 bytes / 17 allocations。按各 Direct/Indirect allocation block 去除地址、按栈排序后集合完全相同；报告定位于未修改的 [GlobalFunctionTable 静态注册](../../src/function.cpp:38)。未添加 suppressions，不将相同基线标为 LSAN pass，未扩展修复该既有问题。上述第二条命令需在第一条非零后仍独立执行，不使用遇错退出的 shell 批处理。

本轮不改变并发/同步合同，未运行 TSAN、全量测试或性能基准。`AM_CHECK` 或 `noexcept` 导致的底层终止不能由 runner 的异常捕获恢复；已覆盖的可恢复分配错误不意味着全面 recoverable OOM。

## 6. 受测文件身份与文档检查

下列 SHA-256 对应本轮普通与 sanitizer 运行的关键文件；不以这些局部 hashes 代替完整构建归档。

| 路径 | SHA-256 |
|---|---|
| apps/inference/inference_main.cpp | `1bea54135b8e0c44d2093c079fa2f4ee96408319eacf95b5bde17c08126c8043` |
| apps/inference/inference_runner.cpp | `4433ae41725c4d5758541f8c4d8f346638a640eaf2c8f450a351ba8e4a6f58ca` |
| tests/unit/apps/inference/test_inference_cli_integration.cpp | `275ced08dc128c544c753d26bdd852f9ed9faa60589202ddb6c41d1af18b5ff4` |
| tests/unit/apps/inference/test_inference_cli_process_helpers.h | `8c0641a9035f49e6d0aba7d015bcb83e7fe699740b03db9d5b426babe28f753c` |
| tests/unit/apps/inference/test_inference_cli.py | `f562f7d91395ff184554e987fb7b10a856a1f3c530634d0b0ecdea1444bce196` |
| tests/unit/apps/inference/fixtures/fail_logger_allocation.c | `eefb0857d83b328dc4c9491bdeaee1836f812c7782602e4076a0806f18b895b1` |
| tests/fixtures/inference_admission/recipe.json | `cc566b7b14c4685d3e22fa76801d79843001d90f963d898ae08279e890d7b0d0` |
| tests/fixtures/inference_admission/checksums.json | `46df9def37d7bc609172de7f94540d69865782937402252e2eac8ee0906c3cbd` |

改动 C/C++ 源与头通过根 `.clang-format` 检查，Python 与文档内代码块完成语法检查，`git diff --check` 通过。`tools/verify_docs.py` 为 0 problem、29 项既有 warning；文档检查不替代执行或 sanitizer 证据。
