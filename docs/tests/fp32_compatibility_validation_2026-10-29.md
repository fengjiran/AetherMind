# FP32 兼容推理验证

- 日期：2026-10-29
- 范围：[11 号方案](../improvement-plan/11-fp16-bf16-execution-support.md) P1：F16/BF16 checkpoint 权重显式拓宽为 FP32，再走 CPU O1/plain 的 Load → Normalize → Compile → Prepare → Prefill/Decode/Generate 链。
- 基线：`4f5c0cef` 加本轮未提交变更；不暂存、不提交。
- 使用和资源边界：[FP32 兼容推理指南](../guides/fp32-compatible-inference.md)。

## 实现合同

P1 使用 model 层 `NormalizeLoadedModelWeightsToFp32`。它保留 config 的 source hint，F32 view 零拷贝，F16/BF16 view 以拥有的 CPU `Buffer` 转换为实际 F32 bytes，并对 tied source view 去重。构图通过 `ModelGraphBuildOptions` 明确将 activation/KV 设为 F32；普通 compiler entry 不自动转换。C++ 的 `LoadAndPrepareCpuExecutableModel(Runtime&, path, CpuModelPrepareOptions)` 只在显式 `kFp32Compatible` + O1/plain 下编排该链。

转换使用未对齐安全的 bit 读取；有限 subnormal 和 signed zero 保留，NaN 结果保留 sign/payload 并置 FP32 quiet bit。它不实现 narrowing 或 native reduced-dtype kernel。转换、source 和 normalized storage 的可见 owner 分类通过 `ModelResourceUsage::weight_normalization` 报告；该字段不是 RSS 或全局 live-byte ledger。

## 最终运行证据

先运行单项 `ModelWeightNormalization.Float16EveryBitPatternMatchesIndependentOracle`，再扩大到新增集合和受影响的既有集合。最终结果如下；没有 failed、disabled、skipped 或未运行项：

| 配置 | 新增集合 | 相关回归集合 | 结果 |
|---|---:|---:|---|
| Release，AVX2 编译关闭 | 40 GTest | 608 GTest | 全通过 |
| Debug，AVX2 编译开启 | 40 GTest | 608 GTest | 全通过 |
| ASAN / UBSAN / LSan | 40 CTest | 573 CTest（571 GTest + 2 fixture） | 全通过，无 sanitizer 报告 |
| TSAN，AVX2 编译关闭 | 40 CTest | 525 CTest（523 GTest + 2 fixture） | 全通过，无 race 报告 |

40 个新增用例包括 15 个 normalizer 合同、9 个 source/execution admission、8 个 graph/compiler policy，以及 8 个生产链/准入用例。两个位级用例分别穷举 F16 与 BF16 的全部 65,536 种输入；有限值以独立的 double 数学解码为 oracle，特殊值按声明的 sign/payload/quiet bit 合同逐位检查，不使用 production helper 或 narrowing round trip 生成 expected。

四种动态本地 checkpoint 为 F16/BF16 × MHA untied/GQA tied，均含两层 decoder。每个 checkpoint 独立解码出对应 FP32 reference checkpoint，在真实 CpuBackend 上比较 Prefill、两次 teacher-forced Decode 的全部所选 logits、各层有效 K/V prefix、commit watermark 与 Argmax；随后比较三个新 token 的 Generate 与重复 Generate。此处 oracle 独立的是 checkpoint 解码，后续 FP32 执行复用已建立的执行链；它不是另一套独立 Transformer 实现。既有外部 HF fixed-trajectory 数值回归同时保留在相关集合中。

生命周期和失败用例验证：原输入/allocator 销毁后 converted owner 仍有效、最后 owner 释放存储、tie 去重、相同内容不同 tensor 不合并、F32 零拷贝、hint 不一致/auto、精确 budget 边界、第二次 allocation 失败清理和重试，以及所有 metadata/budget 预检先于 payload allocation。

首轮新增集合发现 scalable dtype 在 `DataType::operator==` 内调用 `lanes()` 导致断言；normalizer 与 source/hint policy 已在比较前检查 `raw_lanes()==1`，并补充 scalable/reserved/undefined 回归。原失败日志 `release-new.log` 保留，最终成功结果使用 `*-final.*`。

TSAN 的默认进程在 discovery 阶段报 `unexpected memory mapping`，未执行测试。仅对测试进程使用 `setarch x86_64 -R` 后，单项 owner 测试及上述完整集合均通过；未修改系统 ASLR 配置或添加 suppression。该前缀也是复现当前环境 TSAN 结果的条件。

## 命令与原始产物

原始产物目录：`build/validation/fp32-compatibility-20261029/`。最终库、测试和 CLI 均在对应配置中构建；`run_checks.py` 保存新增及相关集合的精确 filter，`*.command.json` 保存实际命令和环境变量：

```bash
cmake --build build/validation/validation-baseline --target aethermind_unit_tests aethermind_infer -j 4
cmake --build build-debug --target aethermind_unit_tests aethermind_infer -j 4
cmake --build build/validation/validation-asan-ubsan --target aethermind_unit_tests aethermind_infer -j 4
cmake --build build/validation/validation-tsan --target aethermind_unit_tests aethermind_infer -j 4

python3 build/validation/fp32-compatibility-20261029/run_checks.py release
python3 build/validation/fp32-compatibility-20261029/run_checks.py debug
python3 build/validation/fp32-compatibility-20261029/run_checks.py asan
setarch x86_64 -R python3 build/validation/fp32-compatibility-20261029/run_checks.py tsan
```

ASAN 使用 `halt_on_error=1:detect_leaks=1`，UBSAN 使用 `halt_on_error=1:print_stacktrace=1`，TSAN 使用 `halt_on_error=1`。Sanitizer 集合排除既有 interposer 专项；TSAN 另按现有 profile 排除 preload/CLI。CTest inventory 与 JUnit 经 `verify_ctest_report.verify_run` 核对精确名字、数量和完整执行状态。`summary.json` 记录最终 source/binary/config/policy hashes；`source.patch` 和 `untracked/` 留存 dirty source 快照。

冻结 policy 的 production 集合仅增加本轮 40 个已审核名字：baseline/AVX 466→506，ASAN 368→408，TSAN 290→330。更新前先从 live inventory 减去这 40 项，核对原 policy 的 count 与 SHA256；`policy-delta.json` 保存差量。AVX 注册 inventory 取本轮 Debug AVX-enabled 构建，不代表重新运行完整 Release AVX profile。既有 external numerical 20、ISA 62、Python CLI/benchmark 集合不变。

其他检查：`cmake.test_classification` 通过；改动 C++ 文件的 clang-format、`git diff --check` 通过；指南的单次加载 C++ 示例以真实 compile_commands 做 syntax 编译通过；文档检查为 0 problems、29 条既有 warnings。

## 验收边界

P1 的限定验证里程碑完成。当前只开放显式 C++ CPU O1/plain FP32 compatibility；普通 Load/Compile/Prepare 不隐式转换，C ABI/CLI 保持 strict FP32。新 CPU 入口使用独立名字，原 `LoadAndPrepareExecutableModel(runtime, path, {})` 编译兼容检查通过；公开 C++ 类型扩展要求调用方重新编译，C ABI v1 未改变。

本轮没有运行全量单测或完整七 profile 最终验收矩阵，没有测量 startup/throughput/RSS，也没有 native W16/A32、16-bit KV、A16 或新 ISA kernel。资源诊断只报告可见 payload 与 owner backing；它不是进程峰值 RSS 或全局多模型限额。11 号整体方案仍为 In Progress，不能由 P1 推导 P2–P4 完成。
