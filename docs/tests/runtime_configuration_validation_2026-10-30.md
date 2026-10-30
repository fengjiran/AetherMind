# 运行前 Runtime 统一配置验证

- 日期：2026-10-30。
- 基线：`4f5c0cef` 加当前既有未提交变更；不暂存、不提交。
- 使用入口：[CPU Runtime 运行前配置](../guides/runtime-configuration.md)。

## 实现与边界

inference 层的 `cpu_inference_options.h` 新增 `CpuRuntimeAssemblyOptions` 和 `MakeCpuRuntimeOptions`，与既有 CPU 编译 preset 共用同一公开头文件。调用方一次给出 context capacity、KV/workspace payload caps 和 CPU feature policy；helper 复用 `MakeFp32KVCacheOptions` 派生模型几何并返回独立值对象，不读权重 payload、不分配 Buffer、不创建 Runtime 或保留模型借用。模型准备与请求预算仍使用各自现有 options。

CLI 私有 `MakeRuntimeOptions` 已删除，CLI 与 model benchmark 共用新入口。原 CLI 的 Load/准入/Compile 先于 RuntimeBuild 顺序、builder 注入、参数语法保持；benchmark 的计时边界和报告 schema 保持。C ABI v1 继续先创建 Runtime、后加载模型，保留显式 KV 几何及数值/overflow 校验，重复的 CPU 默认赋值改为复用 `RuntimeOptions{}`。

`RuntimeBuilder::TryBuild` 在 registry 组装和 pending registration 转移前，集中拒绝未实现的 execution、workspace-manager、tracing 非默认请求；诊断带字段名，返回 `Unimplemented`。CUDA/CANN backend flag 没有匹配的 pending custom factory 时同样拒绝；显式自定义 factory 及其原有覆盖优先级保留，不构成 GPU 执行支持。CPU 实际 required feature 校验仍在 backend 惰性创建时发生。

这是有意的兼容性收紧：之前被忽略的非默认设置现在失败。预检失败保留全部 pending registrations；开始 registry 组装后的失败仍保留原转移规则，已经转入丢弃 registry 的注册不会回到 builder。原 workspace-manager=true 的静默成功测试已迁移：默认 false 继续验证 workspace limit snapshot，新用例验证 true 明确拒绝。用户此前移除 `TryBuild()` 的 `AM_NODISCARD` 改动保持。

## 运行证据

先运行 `CpuRuntimeConfiguration.KvCapRejectsBeforePayloadAndCanRetry`，再扩大集合。19 项配置用例由 9 项原应用映射测试迁移、5 项新的设置/消费测试和 5 项 Builder 预检测试组成，实际新增 GTest 数为 10。

| 配置 | 配置用例 | 相关既有回归 | 结果 |
|---|---:|---:|---|
| Release，AVX2 编译关闭 | 19 GTest | 470 GTest | 全通过 |
| Debug，AVX2 编译开启 | 19 GTest | 470 GTest | 全通过 |
| ASAN / UBSAN / LSan | 19 CTest | 434 CTest（432 GTest + 2 fixture） | 全通过，无 sanitizer 报告 |

全部集合零 failed、disabled、skipped、not-run。ASAN 使用 `halt_on_error=1:detect_leaks=1`，UBSAN 使用 `halt_on_error=1:print_stacktrace=1`；既有 interposer 专项从 sanitizer 集合排除，inventory 与 JUnit 经 `verify_ctest_report.verify_run` 严格核对。

配置用例验证了模型几何、输入对象生命周期、KV exact-cap 准入与无 payload allocation 拒绝、workspace snapshot、CPU feature policy 的实际 backend 消费、预检不创建资源/不销毁注册、修正配置重试、custom provider/factory 覆盖以及禁用默认 CPU allocator 后的自定义 KV 分配。

相关集合覆盖 RuntimeBuilder 的原 OOM/注册转移合同、资源预算、KV、O1/plain 与 O2/packed 准备/生成、FP32 compatibility、外部 fixed trajectory、CLI 直接 Session 对照、C ABI 默认配置/C11合同及参数化分配失败。

另外，真实独立 CLI 的 Python 合同测试 39/39 通过；真实 `aethermind_model_benchmark` 的 Python 正确性/报告合同测试 14/14 通过。后者是 smoke，不作性能对比或收益声明。三个迁移后的 C++ guide 示例均使用真实 compile_commands 做 syntax-only 编译并通过。

## 冻结清单与产物

新配置 suites 的 19 项加入 validation 门禁；其中 9 项从 `InferenceRuntimeOptions` 迁为 `CpuRuntimeAssembly` 并提升门禁。原 `RuntimeResourceLimits.WorkspaceLimitIsOwnedAndIndependentOfWorkspaceManager` 明确迁为 `WorkspaceLimitIsOwnedWithoutWorkspaceManager`，计数不变，拒绝 true 的语义由新预检用例覆盖。

baseline/AVX production 519→538，ASAN production 421→440，TSAN applicable 343→362。更新前从 live inventory 移除新增名字、反向映射这一个旧测试名，必须逐项匹配原 count/SHA256；没有无解释地删除旧集合。其他 numerical/ISA、Python CLI/benchmark 方法集合保持。AVX 注册取本轮 Debug AVX-enabled 构建；TSAN 仅更新已核对的预期名字，本轮未构建或运行。

产物目录：`build/validation/runtime-configuration-20261030/`。`run_checks.py` 保存精确 filters，`*.command.json` 保存命令和环境；`*-final.json`、inventory、JUnit、verified 报告保存实际结果。`policy-delta.json` 保存新增和更名映射。`summary.json` 记录 source/binary/config/policy hashes，`source.patch` 与 `untracked/` 保留 dirty source 快照。

本轮未运行全量单测或完整七 profile 最终矩阵；benchmark 工具仅作上述正确性 smoke，没有性能比较或收益判定。没有并发/同步、kernel/ISA、GPU、原生 16-bit 推理或全局资源账本实现变更。

## 文件合并验证

`cpu_runtime_options.h/.cpp` 已删除，`CpuRuntimeAssemblyOptions`、`MakeCpuRuntimeOptions` 与 CPU 编译 preset 统一由 `cpu_inference_options.h/.cpp` 提供；CLI、model benchmark 与测试均改用合并后的头文件，原文件名引用为零。测试文件同步更名为 `test_cpu_inference_options.cpp`，测试名称和冻结清单不变。

合并后重新构建 `AetherMind`、`aethermind_unit_tests`、`aethermind_infer`、`aethermind_benchmark` 和 `aethermind_model_benchmark`，并运行 `CpuRuntimeAssembly.*:CpuRuntimeConfiguration.*:InferenceCpuPreset.*`，17/17 通过；`clang-format` 与 `git diff --check` 通过。
