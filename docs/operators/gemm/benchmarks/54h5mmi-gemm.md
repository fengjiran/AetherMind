# GEMM 基准记录 — 54H5MMI

| 项 | 值 |
|---|---|
| CPU | Intel Core Ultra 9 285H（Arrow Lake-H，16 核，SMT off，AVX2+FMA+AVX-VNNI，无 AVX-512/AMX） |
| OS / kernel | WSL2，kernel 6.6.87.2-microsoft-standard-WSL2（与宿主共享物理核） |
| Compiler / build | GCC 14.2.0，C++20，Release（-O3） |
| 备注 | WSL2 下 `taskset` 仅咨询性 |

## reference 基线（direct GEMM）

- commit：`5cbd378695bb`、`fc5371323bec`（工作树含未提交改动）
- 命令：
  - `taskset -c 2 ./build-release/tests/benchmark/aethermind_benchmark --benchmark_filter='BM_GemmF32Reference' --benchmark_min_time=1s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`（单线程，每组独立进程）
  - 同参数，filter 为 `'BM_GemmF32Reference.*(/M:(8|9)/|/M:1/K:11008/|/M:1/K:4096/N:(6144|22016|32000))'`
- correctness：`aethermind_unit_tests` 80/80 通过；框架不变量测试 15/15 通过
- 结果（median of 10；GFLOP/s = 2·M·N·K / t）：

| 组 | case | median | 派生指标 |
|---|---|---|---|
| GEMM · K-contiguous | M=1, K=4096, N=4096 | 10.91 ms | 3.07 GFLOP/s |
| | M=1, K=4096, N=6144 | 16.68 ms | 3.02 GFLOP/s |
| | M=1, K=4096, N=11008 | 27.99 ms | 3.22 GFLOP/s |
| | M=1, K=4096, N=22016 | 60.11 ms | 3.00 GFLOP/s |
| | M=1, K=11008, N=4096 | 30.13 ms | 2.99 GFLOP/s |
| | M=1, K=4096, N=32000 | 87.75 ms | 2.99 GFLOP/s |
| | M=4, K=4096, N=4096 | 41.22 ms | 3.26 GFLOP/s |
| | M=8, K=4096, N=4096 | 88.06 ms | 3.05 GFLOP/s |
| | M=9, K=4096, N=4096 | 99.99 ms | 3.02 GFLOP/s |
| | M=16, K=4096, N=4096 | 165.66 ms | 3.24 GFLOP/s |
| GEMM · N-contiguous（整体比 K-contiguous 慢 10–14×） | M=1, K=4096, N=4096 | 150.04 ms | 0.22 GFLOP/s |
| | M=1, K=4096, N=6144 | 179.46 ms | 0.28 GFLOP/s |
| | M=1, K=4096, N=11008 | 286.01 ms | 0.32 GFLOP/s |
| | M=1, K=4096, N=22016 | 620.50 ms | 0.29 GFLOP/s |
| | M=1, K=11008, N=4096 | 478.64 ms | 0.19 GFLOP/s |
| | M=1, K=4096, N=32000 | 796.74 ms | 0.33 GFLOP/s |
| | M=4, K=4096, N=4096 | 603.64 ms | 0.22 GFLOP/s |
| | M=8, K=4096, N=4096 | 939.26 ms | 0.29 GFLOP/s |
| | M=9, K=4096, N=4096 | 1081.59 ms | 0.28 GFLOP/s |
| | M=16, K=4096, N=4096 | 1970.84 ms | 0.27 GFLOP/s |
| 边界形状 | M=1, K=33, N=31 | — | 4.6–5.5 GFLOP/s（L1 驻留） |
| | M=2, K=32, N=33 | — | 4.7–5.4 GFLOP/s |
| Roofline 上限（本机实测） | cpufp FMA f32；STREAM 单线程 | — | 123.74 GFLOP/s；Triad 22.11 / Copy 37.05 GB/s |
| 重复性检查（同二进制、另一进程） | microkernel | — | 5% 阈值自动判定被误判：9/12 REGRESS（逐 case 见 raw 的 `repeatability-comparison.txt`） |

- raw：
  - `benchmark-results/operators/gemm/20260918T012602Z_5cbd378695bb_DESKTOP-54H5MMI_baseline/`（同一 run 的 Linear 组未收录）
  - `benchmark-results/operators/gemm/20260921T023356Z_fc5371323bec_DESKTOP-54H5MMI_reference-shapefill/`
- 结论：reference 基线 Accepted（本地观察）。`perf stat`/governor/microcode 在 WSL2 不可用，百分比级门禁需裸机重采

## reference 自身交错 A/B（环境噪声 floor）

- commit：`5cbd378695bb`
- 命令：`taskset -c 2 ./build-release/tests/benchmark/aethermind_benchmark --benchmark_filter='BM_GemmF32Reference' --benchmark_min_time=1s --benchmark_repetitions=10`（A/B 各独立进程，相邻成对、保留 raw repetitions；脚本见 raw 目录 `collect_ab.sh`）
- 结果：|Δ| = |B−A| / A，aggregate mean real_time

| 组 | median \|Δ\| | p90 \|Δ\| | max \|Δ\| |
|---|---|---|---|
| microkernel | 2.62% | 7.09% | 13.20% |

- raw：`benchmark-results/operators/gemm/20260918T023421Z_5cbd378695bb_DESKTOP-54H5MMI_ab/`（含 Linear 组，未收录）
- 结论：同一二进制自比即有 case 超过 5%（max 13.20%，median 2.62%），5% 自动门禁在本机不可直接使用，判读需按 case 人工

## blocked GEMM（4x16 micro-kernel，G2a 诊断级）

- commit：`42e2df28`（工作树 dirty）
- 命令：`./build/tests/benchmark/aethermind_benchmark --benchmark_filter='M:16/K|M:64/K|M:128/K'`（WSL2，默认多进程无 taskset；本机 noise floor 结论见上）
- correctness：`aethermind_unit_tests --gtest_filter='*GemmAvx2*'` 33/33 通过（含 BlockedShapes 参数化）
- 实现形态：MR=4 × NR=16（acc 沿 N，每 k 步 A 标量广播 × B 行 16 列横向向量，B 转置 k-major 打包 [kc][NR]）；KC=512 面板、MB=48 行驻留；M/N 尾边条复用行对/多目标内核；运行时对 M≥9 K-contiguous 分派（G2b 注册 prepared path 后重验）
- 结果（单次 run，median n=1；GFLOP/s = 2·M·N·K / t）：

| case | baseline（scalar fallback，同入口） | candidate（blocked） | 提速 |
|---|---|---|---|
| M=16, K=4096, N=4096 | 58.78 ms（9.13 GF） | 10.19 ms（**52.7 GF**） | **5.8x** |
| M=64, K=4096, N=4096 | 238.58 ms（9.00 GF） | 29.79 ms（**72.1 GF**） | **8.0x** |
| M=128, K=4096, N=4096 | 487.53 ms（8.81 GF） | 53.70 ms（**80.0 GF**） | **9.1x** |
| M=128, K=4096, N=11008 | 1274.99 ms（9.05 GF） | 150.39 ms（**76.8 GF**） | **8.5x** |

- raw：`benchmark-results/operators/gemm/`（本次未存档 raw repetitions，单次观察）
- 结论：Needs More Data（方向 Accepted）——单次采样、未跑 repetitions/交错 A/B；80 GF 约达本机 FMA 上限 123.7 GF 的 65%。下一步：MB/NR/KC 调参矩阵 + 交错 A/B 复跑 + G2b 接入 Linear prepared path 后以 `BM_LinearPreparedHot` 为主门禁重验

## CPU B-panel packed candidate diagnostic — Needs More Data

- commit：`3d36700d`（工作树 dirty）
- 命令：`./build/tests/benchmark/aethermind_benchmark --benchmark_filter='BM_LinearPreparedHot/M:1/K:4096/N:4096|BM_LinearPackedBpanelHot/M:1/K:4096/N:4096|BM_LinearPreparedStreaming/M:1/K:4096/N:4096|BM_LinearPackedBpanelStreaming/M:1/K:4096/N:4096|BM_LinearPreparedHot/M:16/K:4096/N:4096|BM_LinearPackedBpanelHot/M:16/K:4096/N:4096|BM_LinearPreparedStreaming/M:16/K:4096/N:4096|BM_LinearPackedBpanelStreaming/M:16/K:4096/N:4096|BM_WeightPackingCpuBpanel/N:4096/K:4096|BM_WeightPackingCpuBpanel/N:4096/K:768' --benchmark_min_time=0.02s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true --benchmark_out=benchmark-results/operators/gemm/20260923T103216Z_3d36700d_DESKTOP-54H5MMI_candidate/results.json --benchmark_out_format=json`
- 结果：median `real_time`，prepared plain Linear reference 对照 AVX2 bpanel candidate；candidate 是直接隔离注册 descriptor benchmark，尚未成为 global 默认 kernel。

| mode | shape | plain reference | bpanel candidate | candidate CV |
|---|---|---:|---:|---:|
| hot | M=1, K=4096, N=4096 | 10.47 ms | 3.64 ms | 4.42% |
| streaming（256 MiB packed working set） | M=1, K=4096, N=4096 | 11.57 ms | 3.71 ms | 2.13% |
| hot | M=16, K=4096, N=4096 | 176.57 ms | 7.82 ms | 5.42% |
| streaming（256 MiB packed working set） | M=16, K=4096, N=4096 | 177.33 ms | 7.89 ms | 6.50% |

Cold packing：`N=4096,K=4096` median 66.83 ms，packed size 64 MiB（amplification 1.00×，1.97 GB/s effective read+write）；`N=4096,K=768` median 7.92 ms，logical size 12 MiB、packed size 16 MiB（1.33×，3.64 GB/s）。benchmark setup 的 8 次交错短样本给出 break-even estimate：M=1 约 9 次、M=16 小于 1 次；这些数字仅作诊断，不是 promotion 门禁。

- raw：`benchmark-results/operators/gemm/20260923T103216Z_3d36700d_DESKTOP-54H5MMI_candidate/results.json`
- 结论：**Needs More Data**。本次在 WSL2 运行，case 间 CV 约 2–12%，且没有当前同实现进程噪声 floor、KC=256 对照或完整端到端 Generate 数据；K=768 的 1.33× padding 已验证，其他形状放大率需按真实模型继续检查。保持 identity descriptor 默认，KC512 仍是候选命名，不据此宣称性能收益。

## PackingLayout 枚举身份迁移

- commit：`46627feb7c42`（dirty，含已有 include 间距修改）。
- 命令：`cmake --build build --target AetherMind aethermind_unit_tests aethermind_benchmark -j 8`；单测先运行 `--gtest_filter=KernelRegistry.RegisterRejectsUnknownPackedLayout`，再运行 PackingRecipe / KernelRegistry / CpuBackend / CpuWeightPrepacker / PackedWeightCollectionOwnership / WeightPacking / PackingRequestResolution / ExecutionPlanBuilder / ExecutionPlanImmutability / ExternalReadRequirements / ExecutableModel / LoadAndPrepareExecutableModel / DirectPrefillDecode / InferenceSession 与 CPU packed consumer 的聚焦 filter（252 例）。
- smoke 命令：`./build/tests/benchmark/aethermind_benchmark --benchmark_filter='^(BM_LinearPackedBpanelHot/M:1/K:4096/N:4096|BM_WeightPackingCpuIdentity/N:4096/K:4096|BM_WeightPackingCpuBpanel/N:4096/K:768)$' --benchmark_min_time=0.01s --benchmark_repetitions=1`，JSON 与完整输出归档在 raw 目录。
- 结果：三类目标构建成功；252/252 测试通过，含两种布局的 production-path 数值/尾部 padding、生命周期与 tiny Llama Prefill→Decode/稳态零分配验证。三个 smoke 用例成功：registered/prepared B-panel Linear 的 packed size amplification=1；identity 的诊断 recipe 名和 amplification=1 保持；B-panel N=4096/K=768 的 12 MiB logical → 16 MiB packed 与原字节公式一致。枚举与视图使用同一布局身份，未知值在边界拒绝。
- 输出复核：第一次检查脚本错误地要求 prepared case 带 recipe label；基准本身成功，prepared label 实际为 kernel/cache_mode。按各 case 的既有标签合同复核原始 JSON 通过；未修改实现或重跑取样。复核信息保存在 `validation.json`。
- raw：`benchmark-results/operators/gemm/20261009T024252Z_46627feb7c42_54h5mmi_packing-layout-enum/`（`metadata.json`、`smoke.json`、`smoke.log`、`validation.json`）。
- 结论：Accepted（布局身份/正确性迁移）。单次短 smoke 仅验证接通与标签，不作性能改善、回退或 priority 调整结论；tile、padding、物理字节格式、alignment 与 ISA policy 保持。

## PackedWeightCollection 入库边界修复

- commit：`3836d79cffd8`（dirty，含本次修复与回归测试）。
- 命令：`cmake --build build --target aethermind_unit_tests -j 4`；先运行 `--gtest_filter=PackedWeightCollectionOwnership.InsertRejectsUnspecifiedLayoutWithoutBindingSource`，再运行 `--gtest_filter='PackingRecipe.*:KernelRegistry.*:CpuWeightPrepacker.*:CpuIdentityPackedWeight.*:CpuBpanelPackedWeight.*:PackedWeightCollectionOwnership.*:ZeroDimensions/PackedWeightCollectionZeroSize.*:WeightPacking.*:PackingRequestResolution.*:CpuGemmPackedB.*:CpuQkvLinearPackedB.*:CpuGateUpLinearPackedB.*'`。另对 collection 实现与对应测试启用 `-fsanitize=undefined -fno-sanitize-recover=undefined`，编译和运行命令见 raw `metadata.json`。
- 结果：构建成功，84/84 聚焦测试通过，包含真实 CPU prepack 与 registered B-panel consumer；18/18 collection UBSan 测试通过，无 sanitizer 诊断。覆盖不同位置的零维度、零维度后的负维度、元素数与字节数溢出、scalar/不足存储，以及 `kNone` 拒绝后集合仍可绑定其他 source。
- raw：`benchmark-results/operators/gemm/20261009T044846Z_3836d79cffd8_54h5mmi_packed-weight-boundaries/`（`metadata.json`、`patch.diff`、`focused.log`、`ubsan.log`）。
- 结论：Accepted（入库正确性修复）。本次修改限于准备期校验；性能未测量，无 kernel priority 调整。

## MakeRowMajorView stride 溢出拒绝

- commit：`3836d79cffd8`（dirty，含前序入库修复、已有 RawWeightView 头文件改动及本次 stride 修复）。
- 命令：`cmake --build build --target aethermind_unit_tests -j 4`；先运行 `--gtest_filter=WeightPacking.PrepackWeightRequestsRejectsStrideOverflowBeforeBackendCall`，再运行 `--gtest_filter='WeightPacking.*:PackedWeightCollectionOwnership.*:ZeroDimensions/PackedWeightCollectionZeroSize.*:PackingRequestResolution.*:CpuWeightPrepacker.*'`。另对 collection 实现与 weight-packing 测试启用 `-fsanitize=undefined -fno-sanitize-recover=undefined`，运行 `--gtest_filter='WeightPacking.*Stride*:WeightPacking.PrepackWeightRequestsAcceptsZeroSizedWeights'`；完整编译命令见 raw `metadata.json`。
- 结果：构建成功，57/57 聚焦测试和 4/4 UBSan 测试通过，无 sanitizer 诊断。直接权重与 composite component 的 `{0, INT64_MAX, 2}` 在 backend 打包前返回 `Overflow`，调用次数为 0；`{0, INT64_MAX, 1}` 与正常零尺寸输入仍可打包。
- raw：`benchmark-results/operators/gemm/20261009T055145Z_3836d79cffd8_54h5mmi_stride-overflow/`（`metadata.json`、`patch.diff`、`focused.log`、`ubsan.log`）。
- 结论：Accepted（准备期 stride 正确性修复）。性能未测量，无 kernel priority 调整。

## WeightPackingRequest 统一 components

- commit：`9b5b7ff52376`（dirty，含本次迁移与既有 collection 实现局部修改）。
- 命令：`cmake --build build --target aethermind_unit_tests -j 4`；先运行 `--gtest_filter=WeightPacking.PrepackWeightRequestsRejectsEmptyComponentsBeforeBackendCall`，再运行 `--gtest_filter='WeightPacking.*:PackedWeightCollectionOwnership.*:ZeroDimensions/PackedWeightCollectionZeroSize.*:PackingRequestResolution.*:CpuWeightPrepacker.*:ExternalReadRequirements.*:ExecutableModel.*:LoadAndPrepareExecutableModel.*:DirectPrefillDecode.*:InferenceSession.*:CPUKernelQkvLinear.*:CPUKernelGateUpLinear.*:CPUKernelAddRmsNorm.*'`。另对 request builder、collection 实现和 weight-packing 测试启用 `-fsanitize=undefined -fno-sanitize-recover=undefined`，运行 `--gtest_filter='WeightPacking.*'`；完整编译命令见 raw `metadata.json`。
- 结果：构建成功，126/126 聚焦测试和 27/27 UBSan 测试通过，无 sanitizer 诊断。直接绑定统一为单组件，rank-1 norm、Q/K/V 与 Gate/Up 顺序、tied embedding、backing 生命周期、空批次、零尺寸与溢出拒绝均验证通过；空组件列表在 backend 调用前拒绝。tiny Llama Prefill→Decode 与同步生成符合 scalar oracle，稳态零分配测试通过。
- raw：`benchmark-results/operators/gemm/20261009T063154Z_9b5b7ff52376_54h5mmi_components-only/`（`metadata.json`、`patch.diff`、`focused.log`、`ubsan.log`）。
- 结论：Accepted（请求表示迁移）。删除 `raw_weight` 改变 C++ 请求结构的 API/ABI，调用方须迁移初始化并重新编译；性能未测量，物理布局与 kernel priority 无调整。

## Backend 打包扩展点收敛

- commit：`9b5b7ff52376`（dirty，含 components 迁移与本次 backend 接口修改）。
- 命令：`cmake --build build --target aethermind_unit_tests -j 4`；先运行 `--gtest_filter=BackendWeightPacking.ConvenienceForwardsSelectedRecipeAndInputsToVirtualPacking`，再运行 `--gtest_filter='BackendWeightPacking.*:CpuBackend.*:CpuBackendFactory.*:WeightPacking.*:PackedWeightCollectionOwnership.*:ZeroDimensions/PackedWeightCollectionZeroSize.*:PackingRequestResolution.*:CpuWeightPrepacker.*:ExternalReadRequirements.*:ExecutableModel.*:LoadAndPrepareExecutableModel.*:DirectPrefillDecode.*:InferenceSession.*:CPUKernelQkvLinear.*:CPUKernelGateUpLinear.*:CPUKernelAddRmsNorm.*'`。另对 Backend 协议测试、weight-packing 测试及请求构建/执行实现启用 `-fsanitize=undefined -fno-sanitize-recover=undefined`，运行 `--gtest_filter='BackendWeightPacking.*:WeightPacking.*'`；完整编译命令见 raw `metadata.json`。
- 结果：构建成功，140/140 聚焦测试和 32/32 UBSan 测试通过，无 sanitizer 诊断。三参数便利函数查询并转发所选 recipe，查询失败（含 Unimplemented）在打包前传播；四参数默认实现返回 Unimplemented，显式调用直接虚派发。CPU 实例的三参数调用可见且成功，recipe policy 校验、tiny Llama Prefill→Decode 与稳态零分配测试通过。
- raw：`benchmark-results/operators/gemm/20261009T072256Z_9b5b7ff52376_54h5mmi_backend-pack-entry/`（`metadata.json`、`patch.diff`、`focused.log`、`ubsan.log`，补丁包含新增 Backend 协议测试）。
- 结论：Accepted（打包入口收敛）。packing 仍为可选能力；三参数版本由 virtual 改为 non-virtual 会改变 C++ vtable/ABI，原三参数实现须迁移到四参数入口，并重新编译调用方。性能未测量，物理布局与 kernel priority 无调整。

## Backend 仅保留显式 recipe 打包入口

- commit：`9b5b7ff52376`（dirty，含此前 components/backend 迁移与本次便利重载删除）。
- 命令：`cmake --build build --target aethermind_unit_tests -j 4`；先运行 `--gtest_filter=BackendWeightPacking.ExplicitPackingForwardsRecipeAndInputsToVirtualImplementation`，再运行 `--gtest_filter='BackendWeightPacking.*:CpuBackend.*:CpuBackendFactory.*:WeightPacking.*:PackedWeightCollectionOwnership.*:ZeroDimensions/PackedWeightCollectionZeroSize.*:PackingRequestResolution.*:CpuWeightPrepacker.*:ExternalReadRequirements.*:ExecutableModel.*:LoadAndPrepareExecutableModel.*:DirectPrefillDecode.*:InferenceSession.*:CPUKernelQkvLinear.*:CPUKernelGateUpLinear.*:CPUKernelAddRmsNorm.*'`。另对协议与 weight-packing 测试及相关构建/执行实现启用 `-fsanitize=undefined -fno-sanitize-recover=undefined`，运行 `--gtest_filter='BackendWeightPacking.*:WeightPacking.*'`；完整编译命令见 raw `metadata.json`。
- 结果：构建成功，137/137 聚焦测试和 29/29 UBSan 测试通过，无 sanitizer 诊断。代码只保留带 recipe 的四参数 `PackWeights`，CPU 不再需要 using 声明；独立 `GetPackingRecipe` 查询保留。显式参数/错误传递、默认 Unimplemented、CPU recipe policy、tiny Llama Prefill→Decode 与稳态零分配验证通过。
- raw：`benchmark-results/operators/gemm/20261009T074302Z_9b5b7ff52376_54h5mmi_explicit-pack-only/`（`metadata.json`、`patch.diff`、`focused.log`、`ubsan.log`，补丁包含协议测试最终版本）。
- 结论：Accepted（单一打包入口）。三参数源码调用须改为显式传入 recipe；相对此前单虚入口版本，删除非虚便利函数不再调整 vtable。性能未测量，物理布局与 kernel priority 无调整。

## GEMM 参数统一为 byte 指针与明确精度入口

- commit：`cab0c145`（dirty；源码补丁及 SHA-256 见 raw）。非模板 `GemmArgs` 替代旧 FP32 参数名，A/B/output 统一为 byte 指针、stride 保持元素单位；`RunGemm*` entry 与 reference/optimized 各自数值合同保留。`PackedGemmF32Args` 独立描述 F32 B-panel，边缘切片先按 float 元素偏移再转 byte 地址。
- 命令：`cmake --build build --target aethermind_unit_tests aethermind_benchmark -j 4`；先跑 `--gtest_filter=CpuGemmPackedB.BlockedHandlesMAndNTailsWithoutOverwritingEdges`，再跑 `--gtest_filter='*Gemm*:*MatMul*:*QkvLinear*:*GateUpLinear*:*TinyLlama*:*DirectPrefillDecode*:*HfFixedTrajectory*:*CPUKernelLinear*:*LinearA16*'`。同过滤器运行 Debug ASan/UBSan（`ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1`）；完整命令、构建日志与 XML 见 raw。
- 结果：Release **302/302 通过**；ASan/UBSan **291 通过、11 skip、0 失败**，无地址/UB 诊断。11 项为 sanitizer 下既有 malloc-interposer 不兼容测试，Release 均已通过；本轮未启用 LSan。覆盖 plain/packed、M/N/K tail、strides/guards、zero-size、fallback、fused consumers、HF/tiny-model Prefill/Decode/Generate 与 Release Decode 零分配。
- 构建纠正：首次 Debug 链接发现适配层的 `TensorView::data<std::byte>()` 请求了未支持的 dtype 映射；改用无类型 `data()` 的 byte cast 后，Release 与 Debug 均重新构建并完成上述回归。初次失败日志与源码快照保留。
- benchmark smoke：`--benchmark_filter='^(BM_GemmF32(Reference|ScalarOptimized|Avx2Fma).*M:1/K:33/N:31|BM_LinearPreparedHot/M:1/K:33/N:31|BM_LinearPackedBpanelHot/M:(1|16)/K:4096/N:4096)$' --benchmark_min_time=0.01s --benchmark_repetitions=3`；8 组 × 3 repetitions 完成，独立正确性守卫全部通过、无 error。包含 direct reference/scalar/AVX2 诊断、真实 prepared Linear 与 B-panel candidate。默认 affinity、单线程；未做交错 A/B，不据此判定性能变化。
- raw：`benchmark-results/operators/gemm/20261117T021356Z_cab0c145_54h5mmi_gemm-byte-args/`（metadata/checksums、源码补丁、构建/测试日志、XML、benchmark JSON；本地 gitignored 留存，无远端 retention）。
- 结论：**Accepted（参数合同迁移与限定正确性回归）**。内部参数类型及 F32 primitive 的 C++ 符号发生变化，引用内部头的调用方须重新编译；不新增精度入口，不改变 packing bytes、recipe、线程、资源所有权或 production priority，不作性能收益结论。

## Scalar N-contiguous 的 K-panel 调度与连续 N 遍历

- commit：`cab0c145c845`（dirty；进入任务前已有 byte 参数迁移及其他修改，分别保存在 raw 的 `initial-worktree.patch` / `initial-status.txt`；最终源码、二进制及校验和亦保留）。
- 实现：K-contiguous 算法保留；N-contiguous 独立使用 `NC=128, KC=128`，在同一 K panel 内处理相邻 N 块，保持升序 K 的 FP32 accumulator，首 panel 不读原输出。一个 N 块可覆盖输出时不分割 K；多行且 `N<=64,K<=128,rhs_k_stride<=64` 时使用轻量 4 列路径。固定 accumulator 暂存最多 2 KiB，无 heap/workspace、intrinsic、线程或执行期 autotuning。
- 命令：`cmake --build build --target aethermind_unit_tests aethermind_benchmark -j 4`；`TMPDIR=/tmp ./build/tests/unit/aethermind_unit_tests --gtest_filter='*Gemm*:*MatMul*:*CPUKernelLinear*'`。性能命令由 raw 中 `sweep.sh`、`shortlist.sh`、`paired.sh` 保存；最终 `paired.sh` 为独立进程 A₁/B₁/B₂/A₂，`--benchmark_min_time=0.1s --benchmark_repetitions=5`，默认 affinity、单线程。baseline 使用修复前 scalar，最终 candidate 为 `candidate-tiny/`；loader 路径与二进制 SHA-256 已核对。
- correctness：Release **183/183 通过、0 skip**，覆盖 M/N/K 边界、NC/KC 尾块、padded/unaligned views、NaN 原输出覆盖写、输入与输出 padding guards、跨 K panel 的抵消次序、重复调用、零尺寸、AVX2 fallback 与直接 scalar 稳态零 malloc。prepared Linear hot/tail 与 streaming smoke 共 2 组 × 3 repetitions 通过，实际 resolve 仍为 `cpu::linear_f32_w32_reference`；该 smoke 不执行新 scalar 路径，不作为其生产收益证据。
- 调参过程：保留仅增大 NC 的初版，以及 `NC∈{16,32,64,128} × KC∈{32,128,512}` 共 12 组诊断数据；每组 8 个 shape、3 repetitions。随后对 3 个候选及旧实现复测 13 个 shape、5 repetitions。`NC32/KC32` 对部分大 M=1 用例较快，但对 small/generic-M 和小工作集不均衡；选择 `NC128/KC128` 作为本机较均衡的静态候选。初版宽暂存对 `M=2,K=32,N=33` 产生额外开销，故增加 tiny 多行分支；初版 AB 与失败/撤销实验全部留存，后段曾与增量 build 重叠，不用于最终数值。最终 32 组 shape/layout 完整重新采集，保留每侧 2×5 个 raw repetitions。
- 结果：下表为每进程 5 次 `real_time` 的 median，再取两进程 median 的 median；时间单位 ms。全部是同布局、同 shape 的旧/新实现比较，logical GB/s 不代表 DRAM traffic。

| layout | M / K / N | baseline | candidate | baseline/candidate |
|---|---|---:|---:|---:|
| N-contiguous | 1 / 4096 / 4096 | 41.725 | 4.609 | 9.05× |
| N-contiguous | 4 / 4096 / 4096 | 77.215 | 8.761 | 8.81× |
| N-contiguous | 8 / 4096 / 4096 | 161.819 | 17.329 | 9.34× |
| N-contiguous | 16 / 4096 / 4096 | 186.619 | 20.471 | 9.12× |
| N-contiguous | 1 / 4096 / 512 | 1.681 | 0.231 | 7.29× |
| N-contiguous | 1 / 1024 / 4096 | 2.901 | 0.629 | 4.61× |
| N-contiguous | 16 / 512 / 512 | 0.274 | 0.216 | 1.27× |
| N-contiguous | 1 / 1024 / 4093 | 1.687 | 0.708 | 2.38× |
| N-contiguous，stride=N+13 | 1 / 4096 / 4096 | 21.093 | 4.702 | 4.49× |
| N-contiguous，stride=N+13 | 4 / 4096 / 4096 | 54.340 | 9.740 | 5.58× |

- tiny 检查：`M=2,K=32,N=33` 最终 baseline/candidate 为 0.148/0.137 µs；只用于确认轻量分支移除了初版设置开销，不将百分比级差异作为可信收益。`M=4,K=129,N=65` 为 2.535/1.903 µs。
- 控制组与限制：6 个 K-contiguous 控制 case 的 A₁→A₂ 自比最大变化为 **35.26%**；最终 K-contiguous `M=16,K=4096,N=4096` 为 53.115/61.544 ms（+15.87%），不能隐去或宣称全部 shape 零回退。补充检查确认 MR=1/2 的 K-contiguous helper `.text` 字节一致；generic-M 主函数的自动向量化生成代码有变化。使用仅含 primitive 的共享对象，在同进程、相同数据、guest CPU 2 上交错 12 对调用，该 generic-M case 为 53.116/54.175 ms（+1.99%，命令与明细见 raw）。两种协议不一致，该组仍需裸机复核；guest affinity 不能保证宿主核类别。此处未建立新的逐组可信噪声 floor，5% 不作自动通过/回退判据。
- 诊断边界：反汇编包含 compiler-generated SSE，未建设 strict non-SIMD 变体；malloc interposer 与符号检查均未发现新 heap 调用。WSL2 的 `perf` 缺少匹配 kernel 工具，未获得 PMU/TLB/DRAM 计数。补充交错 harness 初次同时加载两份完整引擎库时发生 SIGSEGV，该实验失败、无 kernel 结论；之后改为仅 primitive 共享对象、共用单份引擎依赖，成功完成交错测量。未采集新 scalar 的 streaming、跨机器或端到端 token 数据。
- raw：`benchmark-results/operators/gemm/20261117T052606Z_cab0c145c845_54h5mmi_scalar-n-contiguous/`（source snapshots、初始/最终 patch、完整 sweep、最终 `final-*.json` / `final-paired-summary.txt`、183 项测试 XML、prepared smoke、反汇编、交错 harness、context 与 SHA-256；本地 gitignored 留存，无远端 retention）。
- 结论：**Accepted（限定实现与正确性修复）；Needs More Data（生产性能门禁）**。大 N/small-M 的完整 K 跨步扫描已改为 panel 内连续 N 遍历，本机诊断显示明显改善；NC/KC 与 tiny 阈值不是跨平台最优值。生产 descriptor、priority、packing bytes 与资源合同不变；不得把这些 microkernel 数据转为端到端收益或 production promotion 依据。


## CPU-03 FP32 GEMM 生产闭环与布局 promotion 决策

- 身份：baseline 保存于 `cab0c145` + 当时 GemmArgs 工作树；candidate 为 `e502fe11` + F32 cold-driver 修正。全量源码补丁、逐文件/二进制 SHA-256 见 raw `qualified-identity.json`、`baseline/identity.json`、`candidate/source.patch`。测量时 HEAD 已是 e502fe11，官方 baseline collector 以该 HEAD + 恢复旧源码的差异表达同一旧树，不能仅看 commit 判断实现相同。
- 机器/配置：本文件头的 Core Ultra 9 285H / GCC 14.2 / WSL2；Release，单线程，affinity=`{2}`（不推导物理核类别），default effective ISA；无 AVX-512。A/B 各用匹配其头文件的测量程序与独立保存的库，`LD_LIBRARY_PATH='$ORIGIN'`；4 对独立进程交错 A/B（AB/BA/AB/BA），Google Benchmark 每进程 3 repetitions（模型报告另按下文 5 repetitions），保存全部原始值。表中为进程内 median 再取 4 进程 median，单位 ms；`hot` 为复用一份权重，不代表超过 LLC 的工作集驻留 L3，streaming 工作集 256 MiB。
- 正确性/平台：最终 Release 聚焦集 **935/935**（含 C ABI 与 glibc malloc 注入）；原长上下文 ASan/UBSan 集 **795 pass / 33 skip / 0 fail**，覆盖 W16/KV16/BF16 A16 的冻结 2048 门禁；F32 cold 参数修正后的适用回归 **562 pass / 16 skip / 0 fail**。LSan 未启用。额外 ASan C ABI 扩围曾有 13 项 `MallocInterposerAvailable()==false` 前置失败，原始失败保留，不计作通过；这 13 项由 Release 实际验证，适用 sanitizer 集明确排除它们。
- ISA 变体：AVX2/F16C 均编译禁用，191 项中 187 pass、4 个 native 专用项 skip；AVX2 开启/F16C 禁用，189 pass、2 个 F16 native 专用项 skip。该证据是在本机执行编译 fallback，不冒充物理无 AVX 主机验收。

| 精度 / layout | Decode M=1 baseline → candidate | 倍数 | Prefill M=16 baseline → candidate | 倍数 |
|---|---:|---:|---:|---:|
| F32 / plain | 10.135 → 2.408 | 4.21× | 164.483 → 8.675 | 18.96× |
| F32 / identity | 10.600 → 2.168 | 4.89× | 167.358 → 7.691 | 21.76× |

模型测量使用 CPU-01 冻结的 tiny MHA untied 几何（hidden 16 / intermediate 64 / vocab 32000 / 2 层 / context 2048）及同源 retained-W16 checkpoint：F32 为 `.models/inference-numerical/mha-untied/model`，F16/BF16 为 `.models/kv16-acceptance/mha-near-limit-wf16|wbf16`。prompt 为 128 个固定 IDs、max_new_tokens=4、KV=F32、warmup=1、repetitions=5；每策略/preset 有 4 对进程，48 份官方报告与 24 份同 workload 比较全部校验通过，token 轨迹一致。内部 Execute 与公开 Generate 边界分开；本表不属于跨精度 A/B、不验收冷文件缓存启动，也不代表 7B/真实大模型质量或吞吐。主要收益来自原默认 scalar/reference 被共享优化 engine 替代，不能归因为“16-bit 必然更快”。

| 策略 / preset | 内部 Prefill baseline → candidate | 内部 3 Decode 合计 baseline → candidate | 公开 Generate baseline → candidate | Generate 倍数 |
|---|---:|---:|---:|---:|
| f32-o1-plain | 32.628 → 11.714 | 0.947 → 0.676 | 35.875 → 14.998 | 2.39× |
| f32-o2-packed | 51.865 → 9.616 | 1.332 → 0.510 | 56.527 → 11.659 | 4.85× |

- cold dispatch：初始纯 SIMD descriptor 在 M=1/K<8 的 F32 小形状出现显著回退，原始探索保留。最终 binding 冻结 `LinearA32W32KernelArgs + driver`，小形状/不适用布局使用原独立 Linear reference，执行期只调用该已冻结函数。最终 4 对进程的 M1/K1/N3、M1/K7/N5、M8/K7/N19 回退大多处于进程噪声范围；不对纳秒级边界强设全平台百分比阈值。M1/K33/N31 仍稳定改善。
- 注册合同：`RegistrationKey` 加入 exact `{layout, alignment}`，同一 selector/ISA 的 identity 与 B-panel 可共存；相同 exact recipe 仅更换 name/priority 仍判重复。没有改变注册锁/冻结同步策略。F32 scalar 与 reference 函数独立；reference 保留作 oracle/兼容 fallback，不用修改 oracle 来制造优化收益。

| B-panel / identity 对照，K=N=4096 | identity | B-panel | B-panel / identity |
|---|---:|---:|---:|
| compute-only hot Decode | 2.140 ms | 3.616 ms | 1.690× |
| compute-only hot Prefill M16 | 7.929 ms | 6.778 ms | 0.855× |
| compute-only streaming Decode | 2.306 ms | 3.717 ms | 1.612× |
| cold packing（分配/释放包含） | 14.416 ms | 47.479 ms | 3.294× |
| pack+bind+1 Decode+release | 16.153 ms | 39.073 ms | 2.419× |
| pack+bind+16 Decode+release | 47.319 ms | 91.113 ms | 1.926× |
| pack+bind+1 Prefill+release | 21.688 ms | 41.172 ms | 1.898× |
| pack+bind+16 Prefill+release | 146.462 ms | 144.578 ms | 0.987× |

- 上表每边 4 个独立进程、每进程 3 repetitions，均包含独立数值 guard。compute-only 的 M16/M64 Prefill 有局部改善，但 Decode 稳定慢 61%–69%；packing-inclusive 在 1 次调用下明显不利，16 次 Prefill 也只落在噪声区间。冷 packing 的两个 source generator 虽不同，均做等字节纯拷贝/重排；最严格的 inclusive 对照在同一 helper、相同数据上完成。
- **Rejected（全局 B-panel promotion）**：identity optimized priority 20，B-panel candidate 10。resolve 尚无 concrete shape，而当前模型共用 Prefill/Decode recipe；不为局部 Prefill 收益默认复制全模型双布局。B-panel 的 existing numerical/composite-slice 覆盖保持，KC256 与 B-panel 整模型质量/性能未因本表自动验收。
- instrumentation 纠正：首次 cold packing 发现两个匿名函数注册同名 Identity benchmark，聚合器拒绝该批（status=rejected-instrumentation），8 份原始进程输出保留。删除重复入口，复用既有 canonical identity packing benchmark后重采；未把歧义数据混入上表。此前 B-panel compute 的旧测量 binary 仍独立保留。
- 固定栈：F32 plain/identity blocked 单函数 `-fstack-usage` 实测 131584 B；该成本与 caller frames 均不同于 tensor workspace/payload。没有新增线程或全局 autotuner。
- raw：`benchmark-results/operators/gemm/20261117T084348Z_cab0c145_54h5mmi_cpu02-03/`；`qualified-identity.json`、`packing-instrumentation-cleanup.json`、各 phase 原始 JSON/log、model report/comparison、源码快照/二进制、stack-usage 与验证 XML 全部保留。命令见采集脚本与各 `commands.json`；本地 gitignored，无远端 retention。
- 结论：**Accepted（CPU-03 identity/scalar/AVX2 生产闭环）**；**Rejected（B-panel 全局提升）**；fusion 自身收益与其他机器/ISA 仍按 Needs More Data/not-run 解释。不以 tiny fixture 代替 7B 产品质量、内存或性能验收。

## K-contiguous scalar 复用单行模板

- commit：`33de2046460a`（工作树 dirty）。`M=1` 和多行 driver 的单行尾块共用 `RunKContiguousRowBlock<1>`，保持 4 列分块、奇数 K 处理、升序 FP32 累加和覆盖写。
- 构建与 correctness：`cmake --build build --target aethermind_benchmark aethermind_unit_tests -j 4`，Release 下 `--gtest_filter='*Gemm*:*Linear*:*MatMul*'` **539/539 通过**。同进程对照的 11 组输入逐元素比较，旧版与最终版输出相等。
- 性能诊断：`collect.sh` 测量 19 组 direct hot case，旧/新两个独立进程各 5 次 repetitions，顺序 A₁/B₁/B₂/A₂。GCC 14.2 将直接调用共享 `RunRowBlock<1, true>` 的尝试生成了独立函数调用和不同的内循环，`1/1024/4093` 与 `1/128/64` 在该轮的延迟中位数分别增加约 46% 和 43%。最终仅在 K-contiguous 行块调度上保留内联，使编译器能看到 `first_row=0`。两组复测中位数分别为 1.113/1.040 ms 和 1.846/1.595 μs。其余组存在波动，例如 `1/4096/6144` 慢 17.2%，同一候选的两进程相差 25.9%。
- 同进程复核：`interleaved.sh` 将同配置的两个 primitive 放在同一进程下交替执行，每组 10 对，数据与输出缓冲相同。最终版的 8 个单行 K-contiguous 用例，成对延迟差的中位数为 −6.1% 至 +2.3%。两种采集方式对部分形状仍有差异，生产性能门禁维持 Needs More Data。
- Prepared Linear smoke：用 `CpuFeaturePolicy` 屏蔽 AVX2，经实际 prepare/bind/execute 解析到 `cpu::linear_f32_w32_scalar`。旧版与最终版的 `1/33/31` hot 均通过原 `1e-5` guard，`1/4096/4096` streaming 均触发该 guard。临时 harness 使用 direct scalar 已有 `1e-4` 容差后，两组各 3 次重复均通过。原 guard 失败记录保留，仓库 benchmark 的容差保持原值。
- raw：`benchmark-results/operators/gemm/20261118T023144Z_33de2046460a_54h5mmi_kcontig-single-row-dedup/`，包含两轮 direct 采集、源码快照、反汇编、同进程对照、prepared smoke 与测试 XML。本地 gitignored。

## AVX2 临时 panel、单行 GEMV 与同型 A16 blocking

- 基线为 `bacdaca9afb2` 加实施前已有工作树差异。数据使用本机 Release、单线程、`taskset -c 2` 和 default effective ISA。保存每个阶段的源码差异、共享库、测量程序及 SHA-256。`measure.py` 运行 4 对 AB/BA 交错独立进程、每进程 3 repetitions、`--benchmark_min_time=0.02s`。以下取进程中位数的中位数，包含 prepared driver 调用。streaming 对照轮转 256 MiB 权重。`perf` 缺本 WSL2 kernel 对应工具，硬件 PMU 数值未采集。
- 幅值分链与 direct A 分别对照。FP32 仅改 SIMD 转置的早期候选在代表形状上没有稳定收益，最终加入 direct A 和跨完整 M tiles 的 B panel 复用。W16/A32 的 Prefill 保留原 `KC=512` 和 `MB=48`，本轮测独立 B panel 转置。Decode 实验以已有转置的 driver 为基线，只比较单行幅值归约分链。A/A control 中部分短 K 形状的中位数差约 9%，纳秒级变化仅作观察。

| Prepared Linear | M / K / N | 基线 → 候选（ms） | 证据 |
|---|---|---:|---|
| FP32 plain Prefill | 64 / 4096 / 4096 | 27.575 → 21.740 | `f32-direct-a-ab.csv` |
| FP32 identity Prefill | 64 / 4096 / 4096 | 25.623 → 21.155 | 同上 |
| FP16 权重 / A32 plain Prefill | 16 / 4096 / 4096 | 15.633 → 8.475 | `transpose-ab.csv` |
| BF16 权重 / A32 plain Prefill | 16 / 4096 / 4096 | 15.666 → 8.349 | 同上 |
| FP16 权重 / A32 identity streaming Decode | 1 / 4096 / 4096 | 2.545 → 1.489 | `w16-decode-ab.csv` |
| BF16 权重 / A32 identity streaming Decode | 1 / 4096 / 4096 | 2.579 → 1.487 | 同上 |
| 同型 FP16，FP16 output | 16 / 4096 / 4096 | 49.394 → 8.799 | `a16-ab.csv` |
| 同型 BF16，BF16 output | 16 / 4096 / 4096 | 55.581 → 9.238 | 同上 |

- 同型 A16 的阻塞对照还含 F32 输出、`M=9/49, K=N=513` 和 LM head `128/16/32000`。独立 FP64 oracle 由已存储的 dtype 解码，窄输出以对区间两端做同 dtype 舍入验证。FP32 `M=16` 多数组进程区间重叠。W16 QKV/GateUp、plain/packed、短 K 及调整前后的观察值保留在 raw 文件。
- 模型 A/B 使用 `measure_models.py`，从固定 tiny Llama checkpoint 走真实 Load、Compile、Prepare、内部 ExecutionPlan 和同步 Generate。每种同精度、同 preset 的 4 对交错进程各完成 1 次 warmup 与 5 次测量。对照组覆盖 FP32、FP16/BF16 权重 A32、BF16 A16，含 O1/plain 和 O2/identity-packed。每轮核对报告 schema、同步 Generate 与内部执行轨迹、相同 workload 的输出 token IDs。模型权重、prompt、编译配置与全部值见 `model-ab-direct-a/` 及最终复采记录。
- 定量栈开销以 Release `-fstack-usage` 记录为准。FP32 blocked 单函数为 33216 B，原 CPU-03 记录为 131584 B。同型 FP16/BF16 blocked 单函数为 156160 B，另有上层调用帧。工作区使用固定栈缓冲，无额外模型 artifact 或运行期堆分配。
- Release 聚焦集 653/653 通过，含 FP32、W16、BF16 A16/KV16、冻结 near-limit 数值预算与 Decode malloc-family 回归。ASan/UBSan 共 621 项通过、2 项跳过、0 项失败，禁用 LSan，单独排除依赖 malloc interposer 的测试。新增测试覆盖跨 K-panel 只舍入一次、M/N/K 尾块与未对齐地址，以及 BF16 极端输入触发全量 reference 回退。
- ASan 首轮的 BF16 回退案例触发 SIGILL。反汇编证明 GCC 选择了 AVX-512 编译单元中弱链接的 `LoadA16Element`，其 ASan 栈清理包含 EVEX `vmovdqu8`。A16 helpers 改用内部链接后，通过隔离复现与聚焦集复验。ISA fixture 同步补齐 W16 两个 optional TU 的 flags 和宏校验。AVX2 ON/F16C OFF 的 prepare/bind/execute 探针验证 FP16 reference、BF16 AVX2，选择身份和数值均通过。
- AVX2/F16C OFF 构建复验 234 项通过、6 项依 ISA 跳过。Release 与上述两类非默认构建的 ISA flags 检查均通过。
- 最终共享库以 `final/source.patch`、`final/sha256.txt` 定位。以同一 benchmark 程序驱动两侧共享库，`final-prepared-ab.csv` 的 `M=16, K=N=4096` W16/A32 Prefill 分别为 FP16 15.175→9.068 ms、BF16 15.450→8.734 ms。对应 identity streaming Decode 为 2.558→1.592 ms、2.587→1.621 ms。同型 A16 窄输出 Prefill 为 FP16 46.919→9.942 ms、BF16 52.351→8.869 ms。FP32 `M=64, K=N=4096` 首轮复采区间重叠，再次复采从 28.388→23.653 ms 且进程区间分离。FP32 `K=N=512` 与同型 A16 `M=8` 的最终多轮结果有波动，性能门禁保持 Needs More Data。
- 最终模型复采的 BF16 A16 O1/plain Generate 从 45.051→31.848 ms，两个进程区间分离。BF16 A16 O2/identity 及 FP32、W16/A32 的 tiny Generate 区间重叠。本次短 FP32 O2 内部 Decode 的中位数偏差复现为约 +20%，同实现 A/A 的偏差约 +25%。补做 64 个 Decode step 后，内部 Decode 为 0.077→0.084 ms/token，公开 Generate 为 13.773→13.968 ms，双方进程区间重叠。短 Decode 偏差仍归 Needs More Data。所有同 workload 的 token 轨迹一致。
- 结论为 Accepted（WSL2 限定的 W16/A32 Prefill/Decode、A16 Prefill 与 FP32 代表性长 Prefill）。保持现有 recipe、ISA priority 与单线程执行，原生 AVX-512/AMX、代表规模模型、更多 KC/MB 组合与裸机 PMU 保留独立验收。原始记录位于 `benchmark-results/operators/gemm/20261118T063844Z_bacdaca9afb2_54h5mmi_cpu-panel/`，含命令、源码快照、样本、失败日志及其复验。

## W16/A32 长 Prefill panel 复用与 Decode 带宽对照

- 基线为 `bacdaca9afb2` 加本轮开始前的工作树差异，候选基于同一配置与源树。四组独立进程交错 AB/BA，每进程 3 repetitions，`taskset -c 2`，Release，已注册的 prepared Linear 路径。使用同一个 benchmark 程序驱动基线和最终共享库，oracle 与权重准备均在计时外。所有身份、命令和原始数据位于 `benchmark-results/operators/gemm/20261118T091318Z_bacdaca9afb2_54h5mmi_w16-reuse/`。
- 长 Prefill 采用 direct A，每个 K panel 中同一 B tile 复用至多 256 行。与旧 48 行相比，`M=512` 的同一 B tile 从 11 次 widening/transpose 降到 2 次。256 行将当次 K panel 的逻辑 A 元素量限制在 512 KiB。实际地址跨度由输入 stride 决定。有界 scratch 仍在栈上，GCC `-fstack-usage` 的函数帧为 32992 B，packed-A 分支为 131296 B。最终仅当完整 M 行数大于 48 且 K 至少为 512 时使用 direct A。
- 首 K panel 直接写 C 独立于 B panel 复用比较。用寄存器中的 +0 保留原加法语义。候选的相关 prepared 中位数为 −5.2% 至 +2.5%，未形成跨形状的稳定收益，保留原清零与累加实现。原始数据见 `overwrite-ab.csv`。
- 单核读取对照用 `g++ -O3 -std=c++20 -mavx2 -DNDEBUG read-bandwidth.cpp` 编译，`taskset -c 2` 扫描 256 MiB 已预触碰缓冲并核对 checksum。四进程的中位数为 19.93–21.37 GB/s。相同 affinity 下，FP16/BF16 权重的 identity streaming Decode 每次读取 32 MiB 逻辑权重，基线 A/A 各边的中位数为 1.50–1.60 ms，折合 20.9–22.4 GB/s。该量级比较支持保留现有 Decode kernel。测量取决于本机 WSL2 与访问模式，从逻辑吞吐无法反推物理 DRAM 带宽。
- 初版 direct A 对 K 无下限。512-token tiny Llama O2 复测中，FP16/BF16 W16/A32 内部 Prefill 从 38.625/37.064 ms 增至 59.868/54.772 ms。该模型的 LM head 为 K=16、N=32000，重排 B 的成本低，C 约占 62.5 MiB。此组未达门禁，候选、原始数据和失败结果保存在 `final/` 与 `model-long-ab-final/`。最终增加 K 下限，重新比较实际模型与 prepared 短 K 边界。
- 最终 Release 聚焦集 560/560 通过，ASan/UBSan 聚焦集 251/251 通过，LSan 关闭。用例覆盖 48 和 256 行分界、K/N 尾块、padded stride、非对齐权重与全输出回退。Release ISA flags 校验通过。禁用 F16C 的构建经 `M=53, K=521` 的真实 prepare/bind/execute 探针验证，FP16 走 reference、BF16 走 AVX2，输出均为预期值。

最终 prepared Linear 数据来自 `qualified-ab.csv`。各值为进程中位数的中位数，单位 ms。

| Layout / phase | M / K / N | FP16 权重 | BF16 权重 |
|---|---|---:|---:|
| identity Prefill | 64 / 4096 / 4096 | 26.340 → 22.254 | 27.025 → 22.440 |
| plain Prefill，非整除 shape | 53 / 521 / 517 | 0.469 → 0.388 | 0.434 → 0.374 |
| plain Prefill，多 M block | 512 / 512 / 512 | 3.275 → 2.736 | 3.348 → 3.058 |
| plain 短 K 控制组 | 512 / 16 / 4096 | 1.693 → 1.599 | 1.650 → 1.631 |
| identity streaming Decode | 1 / 4096 / 4096 | 1.526 → 1.547 | 1.560 → 1.523 |

代表性 `M=64` 的 Prefill 两种精度进程区间均分离。短 K 与 Decode 对照区间重叠。二进制、源码和 SHA-256 见 `qualified/`。

最终 `model-ab-qualified/` 与 `model-long-ab-qualified/` 各用 4 对独立进程、每进程 1 次 warmup 与 5 次重复，验证 128/512-token tiny Llama、FP16/BF16 权重、O1/O2 的内部 ExecutionPlan 和公开 Generate。全部同 workload 的 token 轨迹一致。加入 K 下限后，512-token O2 内部 Prefill 的 FP16/BF16 中位数分别为 40.259→39.139 ms 和 38.709→42.223 ms，双方进程区间均有重叠，初版的大幅回退未复现。128/512-token 各配置的公开 Generate 区间均重叠。

结论为 Accepted（本机代表形状的 W16/A32 长 Prefill 复用）。tiny 模型端到端加速仍为 Needs More Data。首 panel 直接写 C 保持实验结论，Decode 保留现有 GEMV。

## A16/W16 GEMM driver 提取

- 基线为 `4a048033fcaa` 加迁移前的工作树差异。reference、AVX2、AVX-512 FP16 与 AVX-512 BF16 的计算从 Linear 迁入 GEMM。候选使用同一编译配置与工作树。源码差异、构建命令、可执行文件和 SHA-256 位于 `benchmark-results/operators/gemm/20261119T080832Z_4a048033fcaa_54h5mmi_a16-driver/`。
- Release 聚焦集 582/582 通过，其中包含真实注册与绑定、非单位 K/N stride 及输出 guard、跨 panel 累加和异常回退、BF16 A16 与 FP16 P4-B 的独立 oracle 和近容量 Generate 回归。ISA 编译检查确认 reference 的 `-ffp-contract=off` 与各可选 TU 隔离。禁用 F16C 的构建在本机通过 FP16/BF16 A16 的 prepare/bind/execute 探针，均解析为 FP32-output reference。
- ASan/UBSan 聚焦集 21/21 通过，LSan 关闭。AVX-512 的验证范围为编译与合成能力选核。
- prepared Linear 使用同一 benchmark 程序驱动两侧 Release 共享库，`taskset -c 2`，四对独立进程 AB/BA 交错，每进程 3 repetitions。覆盖同型 FP16/BF16、16-bit 和 FP32 输出，以及 `M=1/16, K=N=4096`。`prepared-ab.csv` 八组的进程中位数区间均有重叠。FP16 窄输出 Prefill 首轮为 8.745→9.840 ms，同实现 A/A 的对应中位数波动约 7.9%，定向复测为 9.518→9.544 ms。验收为本机职责提取和行为回归通过，性能对照未证实稳定变化。

## Scalar K 连续 Prefill 的行块与临时权重块对照

- 比较对象为 FP32 plain/identity Linear 绑定到 scalar driver 后的稳态调用。基线为 `208d76dc` 的 4×4，直接 2×4 候选按两行遍历完整权重。另测了列块优先的 2×4，以及临时权重块的不同列宽。最终实现对 `M>8,K>=32,N>=32` 的完整 32 列，在栈上将最多 128 个 K 元素转为连续列布局，再由全部 M 行复用。最后不足 32 列沿用直接行块。GCC 14.2 的 `-fstack-usage` 报告新函数单帧 16,832 B，含 16 KiB 权重块。
- 使用文件头的机器、编译器与 Release 配置。每个进程固定 guest CPU 2。prepared harness 源自仓库 Linear benchmark，以 `CpuFeaturePolicy` 屏蔽 AVX2，核对 `cpu::linear_f32_w32_scalar` 和 `cpu::linear_f32_w32_packed_identity_scalar` 标签。输入生成周期从 251 改为 61，使本组 K 下的定点乘积及部分和可由 FP32 精确表示，保留原有独立 double oracle 和 `1e-5` 检查。direct benchmark 使用仓库原始输入与误差检查。
- 2×4 与 4×4 在相同 prepared plain hot 输入上的独立进程对为 AB、BA，每进程 3 次。以每进程的中位延迟计算，`M=16,K=N=4096` 的两对分别慢 3.2% 和 3.4%，`M=64` 为快 2.9% 和慢 15.7%。仅缩小行块缺少稳定收益。列块优先的 2×4、固定宽度 4/16/32/64/128 与动态尾块候选只做筛选，原始记录含较慢结果。
- 下表为完整基线库与最终库的 prepared A/B。AB/BA 交错 4 对，每进程预热 0.01 s、最短计时 0.025 s、3 次重复；取各进程 real-time 中位数，再列两侧的中位数。变化范围为四对各自的延迟变化。计时仅包含已绑定的执行，含临时权重转置。streaming 轮换总计 256 MiB 权重。

| Prepared Linear | M / K / N | 4×4 基线 ms | 最终 ms | 配对延迟变化范围 |
|---|---|---:|---:|---:|
| plain hot，面板边界 | 9 / 32 / 32 | 0.00210 | 0.00078 | −67.6% 至 −56.0% |
| plain hot，含 M/K/N 非整齐维度 | 9 / 513 / 517 | 0.523 | 0.190 | −66.9% 至 −61.5% |
| plain hot | 16 / 512 / 512 | 0.941 | 0.279 | −72.6% 至 −66.4% |
| plain hot | 16 / 4096 / 4096 | 56.539 | 21.841 | −63.5% 至 −61.1% |
| plain hot | 64 / 4096 / 4096 | 236.411 | 63.017 | −76.0% 至 −71.6% |
| plain streaming | 16 / 4096 / 4096 | 58.535 | 20.541 | −65.9% 至 −62.6% |
| identity hot | 16 / 4096 / 4096 | 56.832 | 21.644 | −65.8% 至 −56.2% |

- 默认实现的 A/A 另采两对，plain hot 的 `M=16,K=N=4096` 为 −7.9%、+0.3%。小 M 和窄 N 对照有噪声：`M=17,K=4097,N=5` 仍用直接行块，四对 A/B 延迟变化为 −5.9% 至 +63.4%，中位 +7.0%，两对 A/A 为 −5.5%、+9.1%。不据此裁决该控制组的性能变化。相同两个完整库的 direct benchmark 另测两对，每进程 3 次，K 连续 `M=16,K=N=4096` 两对分别下降 66.3% 和 55.3%；N 连续 `M=16,K=N=512` 对照为 +8.5%、−2.0%。
- Release 聚焦集 173/173 通过，含 scalar、Linear 与融合调用方。新增用例覆盖面板准入、跨 K 面板的抵消与 NaN 输出覆盖、剩余 31 列及 padding guard。另用 ASan/UBSan 编译 scalar TU 和独立探针，LSan 关闭；随机有限 FP32 输入的 168 个 padded shape 与升序标量 oracle 相等，输出保护区完整。
- 后续阈值扩展：派发条件由 `M>8` 放宽为 `M>=3`（`K>=32,N>=32` 不变），`M=2` 保留原 4 列路径，因为其 B 只有单次遍历，临时 panel 只增加转置开销。直接 primitive 探针 5 轮交替进程取每轮最小值：K 连续 `M=8,K=4096,N=4096` 27.48→11.31 ms、`M=4,K=4096,N=4096` 13.27→9.04 ms、`M=3,K=4096,N=4096` 10.79→8.13 ms、`M=2,K=4096,N=4096` 6.89→7.41 ms（排除依据）；缓存驻留 `M=8,K=512,N=512` 382.4→149.3 µs、`M=4,K=512,N=512` 189.5→105.8 µs、`M=3,K=512,N=512` 142.1→95.7 µs。该组是审查期间的指示性直接测量，未按 §6.7 归档 raw，不作为生产门禁；prepared Linear 与端到端结论沿用上表。
- 原始目录为 `benchmark-results/operators/gemm/20261124T085913Z_208d76dc_54h5mmi_scalar-prefill/`。包含 `experiment.py build`、`qualify.py run`、`direct.py run`、`verify.py run`、候选源码、最终补丁、构建/测量命令、共享库校验和、每次运行 JSON 与测试 XML。
- 结论为 Accepted（本机覆盖形状的 scalar Prefill 临时权重块）；直接 2×4 替换未采用。端到端 Generate、多机与多线程性能尚未测量。

## Scalar N 连续路径的四行分块

- 决策范围为 FP32 单线程、RHS 的 N 维连续、固定权重复用。基线在 `M=2..8` 使用两行块；候选保留单行入口，多行按 4/2/1 分解。算术顺序与 NC/KC 沿用基线。目标是同一 `M=8,K=N=4096` 的 prepared 延迟稳定下降，同时检查其他受影响的 M、tiny、padded 和分界控制组。
- 基线为 `f8beb464` 加采集开始时已有的 K 连续源码差异，保存于 `base.patch`。候选在此基础上调整 N 连续分派，见 `candidate.patch`。机器与文件头一致。Release，GCC 14.2，guest CPU 2。两侧分别构建完整共享库，由同一可执行文件通过 `LD_LIBRARY_PATH` 加载。
- 通过 CPU feature policy 屏蔽 AVX2，检查解析结果为 plain `cpu::linear_f32_w32_scalar`、已绑定函数为 `RunLinearF32Scalar`。权重逻辑形状为 `[N,K]`，实际 strides 为 `{1,N+padding}`，显式核对绑定后的两个 stride。绑定后先用同一逻辑权重的普通行主序副本与 double GEMM oracle 核对 NaN 预填的全部输出，容差为 `1e-5`。周期 61 的二进制分数使所测范围的乘加可精确表示。预热后单独计时 prepared 执行与 direct primitive。
- 首轮扫描 `M=1..9`，覆盖 `K/N=4096/4096、512/512、32/33、129/65`（最后一组 padding 为 13）。每形状 4 对独立进程，AB/BA 交错，每进程 3 次重复，另有 2 对基线 A/A。`M=8,K=N=4096` 的 prepared 四对延迟变化为 −45.0% 至 −30.1%，direct 为 −43.1% 至 −38.3%。`K=N=512` 的同一 M 在 prepared 首轮的配对中位变化为 −35.3%，其中一对为 +12.2%。
- 下表为大形状 prepared 定向复测。每侧 6 个独立进程，交替 AB/BA，每进程 5 次重复，每次包含 6 次调用。另有 3 对同配置基线 A/A。两侧时间分别为各进程中位数的中位数，延迟变化逐对计算。

| M，K=N=4096 | 基线 ms | 候选 ms | 配对中位变化 | 配对变化范围 |
|---:|---:|---:|---:|---:|
| 4 | 8.378 | 4.980 | −40.9% | −53.3% 至 −27.2% |
| 5 | 12.141 | 9.003 | −27.1% | −29.9% 至 −15.8% |
| 6 | 12.840 | 9.193 | −29.2% | −34.1% 至 −9.8% |
| 7 | 16.130 | 12.935 | −19.6% | −27.4% 至 −9.9% |
| 8 | 16.677 | 9.683 | −40.5% | −47.2% 至 −36.2% |
| 9（控制） | 13.947 | 13.436 | −4.9% | −6.9% 至 +11.2% |

- `M=8` 的 A/A 延迟变化绝对值最大 14.9%。该形状将 B 的逻辑遍历从四次减到两次。两组实现的缓存命中率和 DRAM 流量未采集。控制组 `M=1、2、3、9` 的大形状首轮 prepared 差值均跨零。
- tiny 和 padded 的定向复测同样使用 6 对 A/B、每进程 5 次重复与 3 对 A/A。tiny `M=5` 首轮配对中位数为 +10.2%，复测为 −4.6%；tiny `M=6` 复测为 +11.6%，范围 −10.9% 至 +26.7%，该组 A/A 的绝对变化最大 13.7%。padded 的 `M=4..8` 配对中位数在 −5.1% 至 +4.1% 之间，均有正有负。此次没有观察到足以支持 tiny 单独选核的稳定回退。
- `*GemmScalar*:*CPUKernelLinear*` Release 聚焦集 100/100 通过，覆盖四行块与 1/2 行余数、带 padding 的输入输出、跨 K panel 抵消及 NaN 覆盖写。
- 原始记录位于 `benchmark-results/operators/gemm/20261125T090152Z_f8beb464_54h5mmi_scalar-n-rows/`。`run.sh`、`short.sh`、`large.sh` 包含构建及运行命令；同目录含 harness 源码、进程级 CSV、汇总脚本、源码差异、共享库及校验和、测试 XML。
- Accepted（本机大形状 N 连续 scalar 与 prepared Linear 的四行分块）。小形状性能受测量波动影响，未据此增设阈值；端到端 Generate 与其他平台未测。
