# sliding_window 模型准入第一批验收

- **状态**: Current
- **日期**: 2026-10-26
- **关联方案**: [13 号方案 Q1](../improvement-plan/13-deepinfer-quality-and-validation.md#41-q1模型准入与公开入口证据p0)
- **执行基线**: `ddbeb92dbe081b8e60c728f1840e9f74d90e70f6` 加本批未提交补丁；执行时 dirty。本记录随实现提交，不将基线 commit 冒充实现 commit。
- **环境**: Linux、GCC 14.2.0、CMake 3.28.3；普通构建为 Release，ASAN/UBSAN 库为 Debug。

## 1. 合同与范围

[HfModelConfig](../../include/aethermind/model/formats/hf/hf_model_config.h) 以 optional 保留 sliding_window 是否声明整数值。[reader](../../src/model/formats/hf/hf_directory_reader.cpp) 接受缺失/null，整数值保留给 [validator](../../src/model/formats/hf/hf_model_validator.cpp) 判定：正整数表示当前未支持的语义，0/负数为非法值，均返回 InvalidArgument。错误类型、小数/指数、非法后缀和 int64 溢出在解析阶段返回包含字段名的 InvalidArgument。

ModelLoader 在读取权重前拒绝；BuildModelGraph 和直接 BuildLlamaDense 复用同一 ValidateConfig。runner 在 load 阶段失败，recording allocator/backend 尚未创建、分配次数为零。没有实现 sliding-window attention，也没有修改 Attention/KV 执行合同。

新增字段改变 C++ HfModelConfig 布局，C++ 消费者需要重新编译；opaque C ABI 合同未改变。

## 2. 结果

| 验证 | 结果 | 边界 |
|---|---|---|
| 单个正数拒绝回归 | 1/1 通过 | 先运行最窄用例 |
| 新增 sliding_window 回归 | 37/37 通过 | parser、validator、loader、两条构图入口、runner；无 skipped |
| 相关普通回归 | 203/203 通过 | 包含上述 37 项；不与其相加。含既有 full-attention 构图、多步 Generate/scalar oracle、C API 三 token 对照 |
| 聚焦 ASAN/UBSAN | 36/36 通过 | 四个 model 测试源；不含 runner case 或 malloc interposer；无 address/UB 报告 |
| 独立 CLI | 4/4 通过 | null/128 × O1/plain/O2/packed |
| 格式与空白 | 通过 | 本批 C++ 文件 clang-format 检查与 git diff --check |

CLI 的 null 配置在两个 preset 下均生成 `[9555,23124,30954]`；128 均在 load 阶段返回 exit 1，stderr 包含 sliding_window，stdout 为空。这是产品入口行为回归，不是独立外部 numerical golden。

## 3. 复现命令

从仓库根目录运行：

```bash
cmake --build build --target aethermind_unit_tests -j2
TMPDIR=/tmp ./build/tests/unit/aethermind_unit_tests \
  --gtest_filter='SlidingWindow/HfSlidingWindowValidatorTest.RejectsDeclaredSlidingWindow/3' \
  --gtest_brief=1
TMPDIR=/tmp ./build/tests/unit/aethermind_unit_tests \
  --gtest_filter='*SlidingWindow*' --gtest_brief=1
TMPDIR=/tmp ./build/tests/unit/aethermind_unit_tests \
  --gtest_filter='ModelLoader_HfConfigTest.*:RequiredFields/HfConfigMissingRequiredFieldTest.*:SlidingWindow/*:ModelLoader_HfModelValidatorTest.*:ModelLoader_PipelineTest.*:ModelArchitecture.*:BuildModelGraph.*:WindowValues/*:LlamaDenseGraphBuilder.*:InferenceRunnerFailure.*:*LoadAndPrepareExecutableModel*:*TinyLlamaExecution.TinyGenerateMatchesScalarOracleAndStartsFreshEachCall/*:*TinyLlamaExecution.TinyTiedGqaLlamaMatchesScalarOracleAndReusesDecodeBindings/*:*CAbiRuntimeIntegration.MatchesDirectSessionForThreeNewTokens/*' \
  --gtest_brief=1
```

ASAN/UBSAN 复用既有 instrumented 库配置；不引入 Q2 的新 build profile。完整单测中的全局 malloc interposer 与 ASAN 存在已记录的启动冲突，因此单独链接这四个测试源：

```bash
cmake -S . -B build-asan -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer'
cmake --build build-asan --target AetherMind -j2
c++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -DSPDLOG_USE_STD_FORMAT \
  -Iinclude -Isrc -Itests/unit -Ibuild-asan/_deps/spdlog-src/include \
  tests/unit/model/test_hf_config_parser.cpp \
  tests/unit/model/test_hf_model_validator.cpp \
  tests/unit/model/test_model_loader.cpp \
  tests/unit/model/test_model_architecture.cpp \
  build-asan/src/libAetherMind.so build-asan/_deps/spdlog-build/libspdlog.a \
  -lgtest_main -lgtest -pthread -Wl,-rpath,"$PWD/build-asan/src" \
  -o /tmp/aethermind-sliding-window-asan
TMPDIR=/tmp ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1 /tmp/aethermind-sliding-window-asan \
  --gtest_filter='*SlidingWindow*' --gtest_brief=1
```

CLI 先 `cmake --build build-infer --target aethermind_infer -j2`，然后将固定 [HF base](../../tests/fixtures/inference_admission/README.md) 复制到临时目录，仅在 config.json 加入 sliding_window=null 或 128，各运行：

```bash
./build-infer/apps/inference/aethermind_infer --model /absolute/path/to/temporary-model \
  --tokens 7 --context-tokens 8 --max-new-tokens 3 --preset o1-plain
# 对同一模型另运行 --preset o2-packed；临时模型不是仓库资产。
```

## 4. 原始结果与限制

本地原始结果保存在 gitignored `build/validation/sliding-window/`：新增用例、相关回归、ASAN/UBSAN 的 XML，以及四项 CLI 检查的 stdout/stderr/exit code。该目录是本地构建产物，没有 CI artifact retention 承诺。

本批未运行 LSAN、TSAN、全仓测试或 benchmark。ASAN 的 detect_leaks=0 只用于 address/UB 检查，不能关闭既有零泄漏门禁；[既有 LSAN 基线](upper_level_runtime_assembly_m6_validation_2026-10-22.md) 仍需后续独立处理。没有新增性能或其他模型/精度支持声明。

`python3 tools/verify_docs.py` 在文档改动前已有 2 problems / 29 warnings，本批同步后须保持相同诊断集合；这不是文档全门禁通过。Q1 的其余证据工作与 Q2–Q5 继续保留各自退出条件。
