# Linear 基准记录 — desktop-54h5mmi

| 项 | 值 |
|---|---|
| CPU | Intel Core Ultra 9 285H；客体可见 16 logical CPU；AVX2/FMA/F16C |
| OS / kernel | Ubuntu 24.04.1 LTS / 6.6.87.2-microsoft-standard-WSL2 |
| Compiler / build | GCC 14.2.0；build，Release -O3 -DNDEBUG，C++20，AETHERMIND_ENABLE_AVX2=ON，AETHERMIND_ENABLE_F16C=ON；sanitizer=none |
| 备注 | WSL2；taskset -c 0 单核固定；数值为有限样本观测，不推导真实模型加速 |

## P2 O1/plain W16 reference kernel baseline（F16/BF16 权重，F32 激活/累加）

- commit：`ea95702e`，dirty；含 P2 步骤 2–7 工作树补丁。
- 命令：`taskset -c 0 ./build/tests/benchmark/aethermind_benchmark --benchmark_filter='BM_LinearW16.*|BM_LinearPreparedHot.*' --benchmark_out=<raw>/linear-w16-benchmark.json --benchmark_out_format=json`。
- 正确性守卫：每个 case 计时前对独立 FP64 oracle 按冻结合同 `gamma(2K)*sum|A·W| + 1e-6` 校验（K 升序、分离 F32 乘加）；全部通过，无 SkipWithError。
- 结果（prepared hot，median real_time，n 见 JSON iterations；同构建同 affinity）：

| M/K/N | F32 reference | W16 F16 reference | W16 BF16 reference |
|---|---:|---:|---:|
| 1/4096/4096（Decode GEMV） | 11.70 ms | 65.49 ms | 51.11 ms |
| 1/4096/32000（LM head） | 135.73 ms | 551.34 ms | 399.30 ms |
| 16/4096/4096（Prefill） | 288.89 ms | 1035.73 ms | 781.52 ms |
| 64/4096/4096（Prefill） | 643.11 ms | 4358.97 ms | 3286.86 ms |

  streaming（256 MiB 权重工作集轮换）与 hot 基本一致：1/4096/4096 F16 64.07 ms、BF16 50.26 ms，说明当前瓶颈是逐元素 scalar widening 而非权重缓存缺失。binding specialization（冷路径 params 构建）约 0.37–0.45 µs。
- raw：`benchmark-results/operators/linear/20261111T044713Z_ea95702e_desktop-54h5mmi_w16-o1-baseline/`（metadata.json、commands.json、JSON、log）。
- 结论：**Accepted，限定 O1 correctness baseline**。W16 reference 比 F32 reference 慢约 4–7 倍，成本集中在内层逐元素 `LoadW16`（memcpy + dtype 分支 + 软件转换）；BF16 快于 F16（widening 为移位）。这是正确性基线，不是性能候选；不宣称吞吐收益。优化顺序决策见提案 P2 步骤 8 记录：先 widening SIMD 化（F16C/AVX2），再 Decode GEMV 与 Prefill 分块。

## W16 B-panel 布局评估（P2 步骤 9 决策）

- commit：`ea95702e`，dirty；依据上述 O1 baseline 与 O2/packed 验收。
- 观察：streaming 与 hot 时延基本一致，瓶颈是 scalar widening 而非权重布局/缓存缺失；identity packing 已保持 2 字节元素并支持 QKV/GateUp composite 拼接，O2 fused consumer 全部 resolve。
- 决策：**暂不新增 W16 B-panel**。现有 `kCpuBPanelF32Kc512Nr16` 按冻结合同仅限 F32，W16 不得复用其布局身份；新物理格式须独立 `PackingLayout` 并同步 byte formula、packer、validator、artifact key 与 consumer。在 widening SIMD 化落地且 profiling 证明布局收益之前，新增 W16 B-panel 缺乏证据支撑，列为后续 O4 优化批次候选。
- 结论：**Accepted（评估决策）**；O2/packed 首批以 identity 布局交付，W16 B-panel 不进入本批支持范围。

## A16 实际选核身份与验证边界核对

- commit：`cab0c145`，dirty（产品/路线图及证据说明修订，无 kernel 或 priority 变更）。
- 命令：`cmake --build build --target AetherMind -j 4`、`cmake --build build --target aethermind_unit_tests -j 4`；实际选核探针 `/tmp/aethermind-a16-dispatch-probe`，源码及编译命令保留于 raw；聚焦回归 `./build/tests/unit/aethermind_unit_tests --gtest_filter='A16LoadAdmission.*:LinearA16Dispatch.*:LinearA16Avx2Fma.*:CPUKernelRmsNormA16.*:A16ModelAcceptance.*'`。
- 配置：Release，默认 CPU feature policy；真实 `DetectCpuCapabilities` 返回 `{SSE4.1, AVX, AVX2, FMA, F16C, AVXVNNI}`，没有 AVX-512。探针只调用正式 `PrepareKernel` / `GetPackingRecipe`，不执行 kernel；不沿用文件头旧 benchmark 的绑核设置。
- 结果：Prefill/Decode 的 16 个 Linear selector 全部 resolve 为 AVX2/FMA；plain recipe 为 `none`/alignment 0，packed 为 `cpu_identity`/alignment 64，packing query 与 resolved recipe 一致。各 phase 的实际名称相同：

| A=W | output | plain kernel | identity-packed kernel |
|---|---|---|---|
| BF16 | BF16 | `cpu::linear_bf16_wbf16_avx2_fma` | `cpu::linear_bf16_wbf16_packed_identity_avx2_fma` |
| BF16 | F32 | `cpu::linear_bf16_wbf16_out32_avx2_fma` | `cpu::linear_bf16_wbf16_out32_packed_identity_avx2_fma` |
| F16 | F16 | `cpu::linear_f16_wf16_avx2_fma` | `cpu::linear_f16_wf16_packed_identity_avx2_fma` |
| F16 | F32 | `cpu::linear_f16_wf16_out32_avx2_fma` | `cpu::linear_f16_wf16_out32_packed_identity_avx2_fma` |

- 回归：27 项通过、0 skip，其中 Linear dispatch 6 项、AVX2 数值 1 项；其余覆盖 RMSNorm、Load 准入和 BF16 O1/O2 模型路径。合成 AVX-512 capabilities 的测试只验证 dispatch；本轮不重跑 2048 near-limit 校准。
- raw：`benchmark-results/operators/linear/20261117T013309Z_cab0c145_desktop-54h5mmi_a16-dispatch-audit/`（逐 selector TSV、探针源码、metadata、测试 XML/log；本地 gitignored 留存，无远端 retention）。
- 结论：**Accepted（本机选核身份与限定回归）**。native BF16/FP16 的实际执行、数值与性能均为 **not-run**；FP16 A16 高层继续由 Load 拒绝。本次没有性能采样，不调整 priority，不推导加速收益。


## CPU-02 W16/A32 生产优化与 CPU-03 关联闭环

- 身份：baseline 保存于 `cab0c145` + 当时 GemmArgs 工作树；candidate 为 `e502fe11` + F32 cold-driver 修正。全量源码补丁、逐文件/二进制 SHA-256 见 raw `qualified-identity.json`、`baseline/identity.json`、`candidate/source.patch`。测量时 HEAD 已是 e502fe11，官方 baseline collector 以该 HEAD + 恢复旧源码的差异表达同一旧树，不能仅看 commit 判断实现相同。
- 机器/配置：本文件头的 Core Ultra 9 285H / GCC 14.2 / WSL2；Release，单线程，affinity=`{2}`（不推导物理核类别），default effective ISA；无 AVX-512。A/B 各用匹配其头文件的测量程序与独立保存的库，`LD_LIBRARY_PATH='$ORIGIN'`；4 对独立进程交错 A/B（AB/BA/AB/BA），Google Benchmark 每进程 3 repetitions（模型报告另按下文 5 repetitions），保存全部原始值。表中为进程内 median 再取 4 进程 median，单位 ms；`hot` 为复用一份权重，不代表超过 LLC 的工作集驻留 L3，streaming 工作集 256 MiB。
- 正确性/平台：最终 Release 聚焦集 **935/935**（含 C ABI 与 glibc malloc 注入）；原长上下文 ASan/UBSan 集 **795 pass / 33 skip / 0 fail**，覆盖 W16/KV16/BF16 A16 的冻结 2048 门禁；F32 cold 参数修正后的适用回归 **562 pass / 16 skip / 0 fail**。LSan 未启用。额外 ASan C ABI 扩围曾有 13 项 `MallocInterposerAvailable()==false` 前置失败，原始失败保留，不计作通过；这 13 项由 Release 实际验证，适用 sanitizer 集明确排除它们。
- ISA 变体：AVX2/F16C 均编译禁用，191 项中 187 pass、4 个 native 专用项 skip；AVX2 开启/F16C 禁用，189 pass、2 个 F16 native 专用项 skip。该证据是在本机执行编译 fallback，不冒充物理无 AVX 主机验收。

| 精度 / layout | Decode M=1 baseline → candidate | 倍数 | Prefill M=16 baseline → candidate | 倍数 |
|---|---:|---:|---:|---:|
| F16 / plain | 58.437 → 2.615 | 22.35× | 979.774 → 14.335 | 68.35× |
| F16 / identity | 60.852 → 2.590 | 23.49× | 986.013 → 13.884 | 71.02× |
| BF16 / plain | 46.630 → 2.758 | 16.91× | 740.980 → 14.215 | 52.13× |
| BF16 / identity | 45.516 → 2.439 | 18.66× | 761.643 → 13.234 | 57.55× |

- 上表为真实 registered/prepared plain/identity；各主组合的 4 进程区间不重叠。streaming 同矩阵分别取得稳定改善，详细值见 summary。另有 M=2/4/8/32/128/512、K=N=512 的 plain sweep，独立 oracle 均通过；不穷举 dtype×shape×layout 的笛卡尔积。

模型测量使用 CPU-01 冻结的 tiny MHA untied 几何（hidden 16 / intermediate 64 / vocab 32000 / 2 层 / context 2048）及同源 retained-W16 checkpoint：F32 为 `.models/inference-numerical/mha-untied/model`，F16/BF16 为 `.models/kv16-acceptance/mha-near-limit-wf16|wbf16`。prompt 为 128 个固定 IDs、max_new_tokens=4、KV=F32、warmup=1、repetitions=5；每策略/preset 有 4 对进程，48 份官方报告与 24 份同 workload 比较全部校验通过，token 轨迹一致。内部 Execute 与公开 Generate 边界分开；本表不属于跨精度 A/B、不验收冷文件缓存启动，也不代表 7B/真实大模型质量或吞吐。主要收益来自原默认 scalar/reference 被共享优化 engine 替代，不能归因为“16-bit 必然更快”。

| 策略 / preset | 内部 Prefill baseline → candidate | 内部 3 Decode 合计 baseline → candidate | 公开 Generate baseline → candidate | Generate 倍数 |
|---|---:|---:|---:|---:|
| f16-o1-plain | 287.614 → 8.280 | 6.740 → 0.459 | 291.567 → 10.752 | 27.12× |
| f16-o2-packed | 303.191 → 9.400 | 6.927 → 0.459 | 324.107 → 11.786 | 27.50× |
| bf16-o1-plain | 247.711 → 8.590 | 6.196 → 0.471 | 254.193 → 12.255 | 20.74× |
| bf16-o2-packed | 248.439 → 8.686 | 6.195 → 0.531 | 250.505 → 13.260 | 18.89× |

- fusion 对照：相同 packed identity、相同精度/数据/输出，在 prepared QKV/GateUp 与 3/2 个 standalone Linear 之间做独立进程 A/B（M=1/8/32、K=N=512）。18 个组合均通过独立 FP64 guard，进程区间全部重叠；**Needs More Data（fusion 自身性能）**。本批只认定 engine 被各 consumer 正式复用，不声称跨 projection 的 input-tile 复用收益。
- 数值安全：W16 输入始终保持 F32，不以窄化 activation 换取 native dot。unsafe magnitude/non-finite 路径完整重算原 reference；新增可区分 FMA 隐藏“先乘溢出后相消”的回归。shape/stride/zero-size/alias 合同保持，owner/recipe 与 Decode 零分配由模型/对齐回归验证。
- 固定栈成本：A/B panels 128 KiB；`-fstack-usage` 同配置实测 F16 blocked 单函数 131488 B、BF16 131392 B，另有上层调用帧。没有全模型 F32 副本、没有 worker pool；后续并行须先规划 worker scratch。
- raw：`benchmark-results/operators/linear/20261117T084348Z_cab0c145_54h5mmi_cpu02-03/` 的索引、summary 与验证 XML；完整不可变二进制、prepared/model/fusion/ISA 数据关联到同批 `benchmark-results/operators/gemm/20261117T084348Z_cab0c145_54h5mmi_cpu02-03/`。全部为本地 gitignored 留存，无远端 retention。
- 命令：`collect_prepared.py prepared|msweep|fusion`、`collect_models.py` 及每 phase 的 `commands.json` 保留精确参数；不是脱离上下文的加速数字。
- 结论：**Accepted（CPU-02，本机、已记录精度/layout/workload 范围）**。其他平台/ISA 未验收的组合仍 not-run；模型质量与产品 7B 门禁继续独立。
