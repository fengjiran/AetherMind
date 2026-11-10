# BF16 批量转换验证记录 — DESKTOP-54H5MMI

| 项 | 值 |
|---|---|
| CPU | Intel Core Ultra 9 285H；WSL 暴露 16 个 logical CPUs，x86-64 |
| OS / kernel | WSL2，Linux 6.6.87.2-microsoft-standard-WSL2 |
| Compiler / build | GCC 14.2.0；主工程 Release `-O3 -DNDEBUG`、C++20；standalone Debug `-O0 -g`；standalone ASan/UBSan `-O1 -g` |
| 范围 | portable scalar correctness 与 P1 production-path 回归；没有性能测量、ISA candidate 或加速结论 |

## 双向 scalar、prepared 函数表与模型准备接入

- commit：`f92faedf` + dirty；本条对应新增 BF16 batch 基础包与集成测试。
- 构建：`cmake --build build --target AetherMind -j 12`，再 `cmake --build build --target aethermind_unit_tests -j 12`。新增源码与单测由已有 `CONFIGURE_DEPENDS` 收集，无新增 ISA flags 或依赖。
- 首个门禁：Debug、Release 分别单跑 `Bf16BatchConversion.EveryFiniteMidpointAndItsNeighborsUseNearestEven`，均通过，再扩大到相关回归。
- Release 命令：`TMPDIR=/tmp ./build/tests/unit/aethermind_unit_tests --gtest_filter='Bf16BatchConversion.*:Bf16BatchEnvironment.*:CpuBf16Conversion.*:Fp16BatchConversion.*:CpuFp16Conversion.*:Fp16BatchEnvironment.*:BFloat16Test.*:BFloat16MathTest.*:BFloat16IntegerArithmetic/*.*:RawWeightFp32Conversion.*:RawWeightFp32BatchConversion.*:Fp32CompatibilityBatch.*:Fp32CompatibilityPolicy.*:CompatibleCheckpoints/Fp32CompatibleInference.*:Fp32CompatibilityAdmission.*:ModelGraphBuildOptions.*:ModelCompilerGraphBuildOptions.*' --gtest_brief=1`。
- Release 结果：138 tests / 18 suites 通过，包含全部 BF16 位模式、有限 midpoint/邻点、独立 nearest-value oracle 的 200,003 个 FP32 位模式、非对齐/重叠/保护页/FENV，以及模型 callback 的去重、dtype 分流、分配前拒绝和 owner lifetime。已有四组 tiny checkpoint FP32 trajectory/Generate 对照仍通过；未运行全套单测。
- Debug：从真实 `bf16_conversion.cpp`、`bfloat16.cpp` 和新 batch 单测直接构建 standalone executable，14 tests / 2 suites 通过。完整编译命令在 raw `commands.json`。
- ASan/UBSan：相同真实源码，以 `-fsanitize=address,undefined -fno-omit-frame-pointer` 构建 standalone executable。默认运行的 14 个测试通过，但 LeakSanitizer 退出检查因 ptrace 限制失败；保留该失败日志。以 `ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1` 再跑，14 个测试通过。该 sanitizer 结果只覆盖 batch 与 wrapper 源码，不代表完整 model/inference 链 sanitizer 或泄漏检查通过。
- raw：`benchmark-results/operators/bf16_conversion/20261110T095444Z_f92faedf_desktop-54h5mmi_scalar/`，包含 build/test logs、GoogleTest XML、metadata 与 commands；未发布外部 artifact。
- 结论：Accepted（O0/O1 与 P1 接入的正确性范围）。当前两个方向均为 portable scalar；性能工作尚未开展，P2 W16/A32、SIMD 和跨平台运行仍须各自验收。
