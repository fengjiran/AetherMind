# BF16 批量转换验证记录 — DESKTOP-54H5MMI

| 项 | 值 |
|---|---|
| CPU | Intel Core Ultra 9 285H；WSL 暴露 16 个 logical CPUs，x86-64 |
| OS / kernel | WSL2，Linux 6.6.87.2-microsoft-standard-WSL2 |
| Compiler / build | GCC 14.2.0；主工程 Release `-O3 -DNDEBUG`、C++20；standalone Debug `-O0 -g`；standalone ASan/UBSan `-O1 -g` |
| 范围 | scalar/AVX2 checked batch 正确性、P1 production-path 回归与本机 batch benchmark；不跨机器，不推导模型或生成吞吐 |

## 双向 scalar、prepared 函数表与模型准备接入

- commit：`f92faedf` + dirty；本条对应新增 BF16 batch 基础包与集成测试。
- 构建：`cmake --build build --target AetherMind -j 12`，再 `cmake --build build --target aethermind_unit_tests -j 12`。新增源码与单测由已有 `CONFIGURE_DEPENDS` 收集，无新增 ISA flags 或依赖。
- 首个门禁：Debug、Release 分别单跑 `Bf16BatchConversion.EveryFiniteMidpointAndItsNeighborsUseNearestEven`，均通过，再扩大到相关回归。
- Release 命令：`TMPDIR=/tmp ./build/tests/unit/aethermind_unit_tests --gtest_filter='Bf16BatchConversion.*:Bf16BatchEnvironment.*:CpuBf16Conversion.*:Fp16BatchConversion.*:CpuFp16Conversion.*:Fp16BatchEnvironment.*:BFloat16Test.*:BFloat16MathTest.*:BFloat16IntegerArithmetic/*.*:RawWeightFp32Conversion.*:RawWeightFp32BatchConversion.*:Fp32CompatibilityBatch.*:Fp32CompatibilityPolicy.*:CompatibleCheckpoints/Fp32CompatibleInference.*:Fp32CompatibilityAdmission.*:ModelGraphBuildOptions.*:ModelCompilerGraphBuildOptions.*' --gtest_brief=1`（记录时 `BFloat16Test.*` 含 6 个标量转换用例；它们后随新增 `tests/unit/dtypes/test_bf16_conversion.cpp` 改名为 `BFloat16ToFP32Test.*` 与 `BFloat16FromFP32Test.*`，复现同等覆盖需把这两个 suite 加入 filter，用例总数不变）。
- Release 结果：138 tests / 18 suites 通过，包含全部 BF16 位模式、有限 midpoint/邻点、独立 nearest-value oracle 的 200,003 个 FP32 位模式、非对齐/重叠/保护页/FENV，以及模型 callback 的去重、dtype 分流、分配前拒绝和 owner lifetime。已有四组 tiny checkpoint FP32 trajectory/Generate 对照仍通过；未运行全套单测。
- Debug：从真实 `bf16_conversion.cpp`、`bfloat16.cpp` 和新 batch 单测直接构建 standalone executable，14 tests / 2 suites 通过。完整编译命令在 raw `commands.json`。
- ASan/UBSan：相同真实源码，以 `-fsanitize=address,undefined -fno-omit-frame-pointer` 构建 standalone executable。默认运行的 14 个测试通过，但 LeakSanitizer 退出检查因 ptrace 限制失败；保留该失败日志。以 `ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1` 再跑，14 个测试通过。该 sanitizer 结果只覆盖 batch 与 wrapper 源码，不代表完整 model/inference 链 sanitizer 或泄漏检查通过。
- raw：`benchmark-results/operators/bf16_conversion/20261110T095444Z_f92faedf_desktop-54h5mmi_scalar/`，包含 build/test logs、GoogleTest XML、metadata 与 commands；未发布外部 artifact。
- 结论：Accepted（O0/O1 与 P1 接入的正确性范围）。当前两个方向均为 portable scalar；性能工作尚未开展，P2 W16/A32、SIMD 和跨平台运行仍须各自验收。

## AVX2 双方向 checked batch 与权重准备

- commit：`52dd7214` + dirty；baseline/candidate 为同一实现构建，差别仅是 prepared 函数表。baseline 是本次 portable scalar batch，不是对旧提交二进制的测量。
- 命令：`cmake --build cmake-build-release --target aethermind_benchmark aethermind_unit_tests -j 8`；`taskset -c 0 aethermind_benchmark --benchmark_filter=BM_Bf16 --benchmark_min_time=0.02s --benchmark_repetitions=9 --benchmark_enable_random_interleaving=true --benchmark_out_format=json`。完整命令、metadata 与 JSON 在 raw `commands.json` / `metadata.json`。
- 比较：一轮同进程随机交错 A/B，每 case 九次 repetitions；下表报告每 case CPU-time 的中位数与 min–max 范围，不采用最好单次结果。bytes 口径为每元素读写合计 6 bytes，不等于实际 DRAM traffic。

| Case，N = 1,048,576，offset = 0 | Scalar median | Selected median | 比值 |
|---|---:|---:|---:|
| checked BF16→FP32 | 0.249 ms (0.232–0.287) | 0.162 ms (0.146–0.228) | 1.53× |
| checked FP32→BF16 | 0.412 ms (0.400–0.666) | 0.172 ms (0.156–0.214) | 2.40× |
| `ConvertRawModelWeightsToFp32`，含分配/去重/释放 | 0.375 ms (0.355–0.390) | 0.278 ms (0.265–0.300) | 1.35× |

- 中小尺寸：widening 在 N=1024/65536 达约 3.0×，N=8/31 在噪声内；narrowing 在 N=1024–1,048,576 稳定约 2.5–2.6×；权重准备 N=65536 约 1.94×、N=1024 约 1.20×。offset=1 与小输入的完整数据在 raw JSON。
- widening 的收益低于 fp16 的 F16C 硬件 widening（约 6.8–7.7×）：bf16 widening 本身只是移位，scalar 基线已较快，AVX2 主要受益于向量化访存；narrowing 的 scalar 基线含逐元素 NaN 分支，SIMD 收益更高。
- 正确性：Release 聚焦门禁 67 tests / 7 suites 通过，包括全部 BF16 位模式、midpoint、随机位型、保护页/FENV、callback 去重与四组 bf16/fp16 tiny checkpoint trajectory/Generate 对照。
- 未测量：真实 HF checkpoint `Load→Compile→Prepare` wall-time 调用链（工作区无现成 fixture）；该路径的正确性由上述 CompatibleCheckpoints 用例覆盖。不做跨机器结论。
- raw：原记录引用的 `benchmark-results/operators/bf16_conversion/20261110T165354Z_52dd7214_desktop-54h5mmi_vector/` 在当前工作区未找到，其数值无法复核；复跑证据与结论见下节。
- 结论：Accepted，限定本机 batch primitive 与 P1 权重准备路径；narrowing 当前无生产消费方，不推导整模型或生成吞吐收益。AVX512-BF16 narrowing 与其他 ISA 后续独立验收。

## AVX2 benchmark 复跑（2026-11-11）

- 原因：上一节引用的 raw 目录在当前工作区不存在。本节以 `0eb18441` + dirty（scalar 实现集中；AVX2 代码未变）在同一 `build`（Release、AVX2/F16C ON）重新测量，保留全部 repetitions。
- 命令：`cmake --build build --target AetherMind aethermind_unit_tests aethermind_benchmark -j 12`；`taskset -c 0 ./build/tests/benchmark/aethermind_benchmark --benchmark_filter='BM_Bf16(Batch/(Widen|Narrow)/N:1048576/Selected:[01]/Offset:0|WeightPreparation/N:1048576/Selected:[01])$' --benchmark_min_time=0.05s --benchmark_repetitions=9 --benchmark_enable_random_interleaving=true --benchmark_out_format=json`。
- 环境：同一 CPU/kernel/compiler 口径；focused 轮在 load average 约 1.1 时运行。先前全量 filter 轮在 final validation 结束后立即运行（load average > 10），scalar narrowing median 0.461 ms、cv 44%，保留在 raw 但不在下表采用。
- 每 case 为 9 次 repetitions 的 CPU-time median 与 min–max：

| Case，N = 1,048,576，offset = 0 | Scalar median | Selected median | 比值 |
|---|---:|---:|---:|
| checked BF16→FP32 | 0.182 ms (0.170–0.196) | 0.161 ms (0.151–0.187) | 1.13× |
| checked FP32→BF16 | 0.327 ms (0.310–0.360) | 0.152 ms (0.146–0.168) | 2.15× |
| `ConvertRawModelWeightsToFp32`，含分配/去重/释放 | 0.287 ms (0.272–0.335) | 0.268 ms (0.237–1.124) | 1.07× |

- 解读：narrowing 的整数 SIMD 收益稳定；widening 已接近内存带宽，收益有限。权重准备受分配/释放噪声影响，AVX2 组有一次 1.124 ms outlier，不能据此声明稳定加速。复跑值与上一节记录不同，说明该 primitive 的标量基线依赖负载与 cache 状态，单轮比值不能当作通用加速倍数。
- raw：`benchmark-results/operators/bf16_conversion/20261111T025118Z_0eb18441_desktop-54h5mmi_vector/`，含 metadata、commands、两轮完整 JSON 与 console 输出；未发布外部 artifact。
- 结论：Accepted，限定本机 batch primitive 复跑；不改变上一节的限定范围，O3/O4 性能门禁仍未建立。

## Scalar 实现集中到 BFloat16 数值源文件

- commit：`0eb18441` + dirty。两个 scalar batch 迁入 `bfloat16.cpp`，与单值转换共享 anonymous namespace 中的整数 helper，checked spans 继续由 `bf16_conversion.cpp` 校验。
- 命令与 raw：`benchmark-results/operators/bf16_conversion/20261111T014502Z_0eb18441_desktop-54h5mmi_scalar-location/` 的 `commands.json`、测试 XML、build logs、benchmark JSON 和 `bfloat16.asm`。
- 正确性结果：Release 首跑全部有限 midpoint 用例通过，再运行 118 tests / 12 suites 通过。standalone Debug scalar 运行 22 tests / 3 suites 通过，覆盖位级 oracle、FENV、范围及 fallback。Release 回归同时覆盖 AVX2 tail 和 P1 准备/生成。
- 汇编检查：两个 scalar batch 循环均无 call 指令或 PLT relocation，私有 helper 已内联。
- benchmark smoke：N=65,536、offset=0，两个方向和权重准备分别运行 scalar/selected 共 6 cases，随机交错、每 case 三次 repetitions，均成功。该轮只检查路径可运行，没有采集变更前基线。
- 结论：Accepted，限定数值实现复用与源码组织调整。
