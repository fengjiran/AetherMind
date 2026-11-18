# CPU 算子覆盖与生产基线台账

本文是 [CPU 算子演进路线图](cpu-operator-evolution.md) CPU-01 的公共基准与跨算子覆盖台账：冻结参考机器/构建身份与代表模型 revision，记录真实生产准备入口的选核身份，并标记覆盖缺口。§1/§2 保留 CPU-01 冻结基准，§3 的矩阵身份随 CPU-02/03 更新；原始基线与后续变更证据分别留存。算子语义合同、专项提案与单算子性能证据仍归各算子子目录；本文只做跨算子关联，不复制合同，不新建 capability runtime registry。

证据遵循 [算子开发与优化工作流](../guides/operator-development-workflow.md) §4–§5：记录追加式、不覆盖；机器级数值不跨机器引用。本文 §6 是跨算子审计记录；单算子性能实验仍记入 `docs/operators/<op>/benchmarks/<machine>-<op>.md`。

状态标记含义：`reference`＝保守实现/独立正确性 oracle（是否为默认以实际选核表为准）；`optimized`＝已有 SIMD/ISA 实现且被生产选中；`candidate`＝已注册 eligible 但未被默认选中；`unsupported`＝该 selector 无 eligible kernel（准备期 NOT_FOUND）；`not-run`＝缺少目标硬件或未采集，不等于不支持。

## 1. 参考机器与构建身份（冻结）

| 项 | 值 |
|---|---|
| 机器名 | `desktop-54h5mmi` |
| CPU | Intel Core Ultra 9 285H（family 6 / model 197 / stepping 2）；客体可见 16 logical CPU，1 thread/core，1 socket，1 NUMA node |
| 缓存 | L1d 768 KiB（16×）、L1i 1 MiB（16×）、L2 48 MiB（16×）、L3 24 MiB（1×） |
| 关键 ISA | AVX2 / FMA / F16C / AVX_VNNI；**无 AVX-512**（硬件不存在） |
| OS / kernel | Ubuntu 24.04.1 LTS / 6.6.87.2-microsoft-standard-WSL2 |
| 内存 | 62 GiB（WSL2 客体） |
| Compiler / build | GCC 14.2.0；`build/` Release `-O3 -DNDEBUG`，C++20，sanitizer=none |
| 构建开关 | `AETHERMIND_ENABLE_AVX2=ON`、`F16C=ON`、`AVX512BF16=ON`、`AVX512FP16=ON`；`GEMM_HAS_AVX2_FMA_KERNEL=1` |
| effective features | 真实 `DetectCpuCapabilities` 返回 `{SSE4.1, AVX, AVX2, FMA, F16C, AVXVNNI}`；feature policy=default |
| 构建身份 | commit `cab0c145`，dirty（仅文档改动，无源码变更）；该项描述 CPU-01 冻结时的库；后续 CPU-02/03 源码/二进制身份另记于新增证据 |
| 备注 | WSL2；`taskset` 仅咨询性，混合拓扑可能被伪造成对称 CPU；核类别/SMT 结论只在裸机采集 |

冻结约束：

- 后续 CPU-* 的同机性能对照以本机此配置为公共基准；已有机器记录不足以支持跨机器排名，不据此推定 AVX-512/AMX 更快。
- AVX-512 BF16/FP16 TU **已编译**，但本机 effective features 不含 AVX-512，native descriptor 不会被选中；native 的实际执行、数值与性能均为 **not-run**，须待目标硬件补验（CPU-10）。合成 capabilities 的 dispatch 测试只证明选择逻辑，见 [Linear A16 选核核对](linear/benchmarks/desktop-54h5mmi-linear.md#a16-实际选核身份与验证边界核对) 与 [RMSNorm A16 选核核对](rmsnorm/benchmarks/desktop-54h5mmi-rmsnorm.md#a16-实际选核身份与验证边界核对)。
- `build/CMakeCache.txt` 仍残留 `AETHERMIND_ENABLE_GEMM_AVX2_CANDIDATE=OFF`、`AETHERMIND_ENABLE_GEMM_SCALAR_CANDIDATE=OFF`；这两个选项已在 `66f61523` 从源码移除，不再 gate 任何编译。当前 GEMM AVX2 TU 经 `GEMM_HAS_AVX2_FMA_KERNEL` 编译，残留缓存项不构成“candidate 被关闭”的证据。
- 部署方如需暂时排除 native，可用既有 `CpuFeaturePolicy.disabled_features` 削减特征；这不是新增准入门。

## 2. 代表模型与 fixture revision（冻结）

公共基准模型为 pinned HF checkpoint 衍生的 tiny fixture，用于正确性/基线可追溯，**不是** 7B 产品质量、总内存或吞吐验收的代表模型（后者按路线图 §6.3 单独验收）。

源身份（[admission recipe](../../tests/fixtures/inference_admission/README.md) 唯一权威）：

| 项 | 值 |
|---|---|
| 源 | `peft-internal-testing/tiny-random-LlamaForCausalLM` |
| revision | `d0b81c5e23f2d61bb7a5f430a8cd6f3e711c0cb3` |
| `config.json` SHA-256 | `ea68007f47752b68f0f58e5fc2d51fa380bb730af51faff8c1df9fccbe608bc2`（720 B） |
| `model.safetensors` SHA-256 | `49c20f32c6c597480fcaec5df2f86c645eabea765cbea1e67886dbae45e5c992`（4131280 B） |

本地已验收 bundle（本轮实际运行校验器，exit 0）：

| Bundle | 校验命令 | 结果 | 身份 |
|---|---|---|---|
| `.models/inference-numerical` | `verify_llama_numerical_reference.py --expected tests/fixtures/inference_numerical/checksums.json` | verified，5 cases / 120 tensors | `manifest.json` SHA-256 `1e87d57b04d5612bcfe2fdb5b0af9f99e5f6d702f1e2cb7bdbf217463ad5d1b3` |
| `.models/kv16-reference` | `prepare_kv16_reference.py verify` | verified，50 files | manifest SHA-256 `af66d31a47d64b82df09c6ce42aefee09111a029a6f1a3b9eed2d9508400d23b` |
| `.models/inference-admission` | recipe + `checksums.json`（20 variants） | 已制备 | `generated-manifest.json` SHA-256 `46df9def37d7bc609172de7f94540d69865782937402252e2eac8ee0906c3cbd` |

代表几何（`inference-numerical/mha-untied/model`，F32 untied）：hidden 16、intermediate 64、vocab 32000、2 layers、4 Q/KV heads、head_dim 4、max_pos 2048、rms_norm_eps 1e-6、RoPE theta 10000 default。其 `model.safetensors` SHA-256 `a75eafd663e29632ddfe06bf4ebab90985dbb9c28a43dee368a565377e6d47c6`、`config.json` SHA-256 `8b0b513b2c8709da249074537153124033fd8e2f9cbc6479dabb2dd57bb22dfb`。

KV16 near-limit 验收模型（`.models/kv16-acceptance`，2048 capacity）由同一 mha-untied 源衍生，矩阵转 F16/BF16、Norm 保留 F32：

| 变体 | output `model.safetensors` SHA-256 | source 一致性 |
|---|---|---|
| `mha-near-limit-wf16` | `3f2c56fe2a4d9c86063f1a7c889130c9749a990d3d03f1caa52fad6c8891915b` | source_model_sha256 = `a75eafd6…`（与 mha-untied 一致） |
| `mha-near-limit-wbf16` | `ca4771b9824974faa1767b4d4f7ba58af09494884f3da13f774e26d1e41360df` | 同上 |

测量边界：model benchmark 的 cache policy 为 `pre_read_by_identity_collection`，loading 数值**不是**冷文件缓存启动；公开 Generate 与内部 prepared 诊断区间分开（见 [model benchmark README](../../tests/benchmark/models/README.md)）。

## 3. 生产选核身份台账（真实检测）

方法：跨算子探针只调用正式 `CpuBackend::PrepareKernel` / `GetPackingRecipe`，真实 `DetectCpuCapabilities` + default policy，不执行 kernel、不做性能采样。逐 selector 完整 TSV（含 phase、output/state dtype、plain/packed、recipe、alignment、recipe-query 一致性）见 §6 raw。所有注册 descriptor 的 phase 为 `kBoth`，故 Prefill/Decode 选核身份相同；下表按组合给出 plain 与 packed-identity 的实际 `ResolvedKernel::name`。plain recipe 一律 `none`/alignment 0；packed 一律 `cpu_identity`/alignment 64，且 `GetPackingRecipe` 与 `PrepareKernel` 全部一致（agree）。

### 3.1 矩阵计算

| 算子 | act / weight / output | plain | packed-identity |
|---|---|---|---|
| Linear | F32 / F32 / F32 | `cpu::linear_f32_w32_avx2_fma` | `cpu::linear_f32_w32_packed_identity_avx2_fma` |
| Linear | F32 / F16 / F32 | `cpu::linear_f32_wf16_avx2_fma` | `cpu::linear_f32_wf16_packed_identity_avx2_fma` |
| Linear | F32 / BF16 / F32 | `cpu::linear_f32_wbf16_avx2_fma` | `cpu::linear_f32_wbf16_packed_identity_avx2_fma` |
| Linear | BF16 / BF16 / BF16 | `cpu::linear_bf16_wbf16_avx2_fma` | `cpu::linear_bf16_wbf16_packed_identity_avx2_fma` |
| Linear | BF16 / BF16 / F32 | `cpu::linear_bf16_wbf16_out32_avx2_fma` | `cpu::linear_bf16_wbf16_out32_packed_identity_avx2_fma` |
| Linear | F16 / F16 / F16 | `cpu::linear_f16_wf16_avx2_fma` | `cpu::linear_f16_wf16_packed_identity_avx2_fma` |
| Linear | F16 / F16 / F32 | `cpu::linear_f16_wf16_out32_avx2_fma` | `cpu::linear_f16_wf16_out32_packed_identity_avx2_fma` |
| QkvLinear | F32 / F32 / F32 | **unsupported** | `cpu::qkv_linear_f32_w32_avx2_fma` |
| QkvLinear | F32 / F16 / F32 | **unsupported** | `cpu::qkv_linear_f32_wf16_avx2_fma` |
| QkvLinear | F32 / BF16 / F32 | **unsupported** | `cpu::qkv_linear_f32_wbf16_avx2_fma` |
| GateUpLinear | F32 / F32 / F32 | **unsupported** | `cpu::gate_up_linear_f32_w32_avx2_fma` |
| GateUpLinear | F32 / F16 / F32 | **unsupported** | `cpu::gate_up_linear_f32_wf16_avx2_fma` |
| GateUpLinear | F32 / BF16 / F32 | **unsupported** | `cpu::gate_up_linear_f32_wbf16_avx2_fma` |
| MatMul | F32 / F32 / F32 | `cpu::matmul_f32_reference` | **unsupported** |

F32 Linear packed 另有 `cpu::linear_f32_w32_packed_bpanel_candidate`（QKV/GateUp 同名 candidate）已注册且 eligible（AVX2/FMA 满足），B-panel priority 保持 10，identity optimized 为 20，默认保持 identity；exact recipe 纳入注册去重键，两个布局可在相同 ISA 要求下共存。无 AVX2/FMA 时 F32 为 scalar，W16 为 reference；F16C 缺失只使 A32/F16 回退，A32/BF16 仍可使用 AVX2/FMA。

### 3.2 Embedding 与 Norm

| 算子 | act / weight / output | plain | packed-identity |
|---|---|---|---|
| Embedding | F32 / F32 / F32 | `cpu::embedding_f32_w32_reference` | `cpu::embedding_f32_w32_packed_identity_reference` |
| Embedding | F32 / F16 / F32 | `cpu::embedding_f32_wf16_f16c`（optimized） | `cpu::embedding_f32_wf16_packed_identity_f16c` |
| Embedding | F32 / BF16 / F32 | `cpu::embedding_f32_wbf16_avx2`（optimized） | `cpu::embedding_f32_wbf16_packed_identity_avx2` |
| Embedding | BF16 / BF16 / BF16 | `cpu::embedding_bf16_wbf16_reference` | `cpu::embedding_bf16_wbf16_packed_identity_reference` |
| Embedding | F16 / F16 / F16 | `cpu::embedding_f16_wf16_reference` | `cpu::embedding_f16_wf16_packed_identity_reference` |
| RmsNorm | F32 / F32 / F32 | `cpu::rmsnorm_f32_w32_avx2_fma`（optimized） | `cpu::rmsnorm_f32_w32_packed_identity_reference`（**reference**） |
| RmsNorm | BF16 / F32 / BF16 | `cpu::rmsnorm_bf16_w32_avx2_fma` | `cpu::rmsnorm_bf16_w32_packed_identity_avx2_fma` |
| RmsNorm | F16 / F32 / F16 | `cpu::rmsnorm_f16_w32_avx2_fma` | `cpu::rmsnorm_f16_w32_packed_identity_avx2_fma` |
| AddRmsNorm | F32 / F32 / F32 | `cpu::add_rmsnorm_f32_w32_reference` | `cpu::add_rmsnorm_f32_w32_packed_identity_reference` |
| AddRmsNorm | BF16 / F32 / BF16 | **unsupported** | **unsupported** |
| AddRmsNorm | F16 / F32 / F16 | **unsupported** | **unsupported** |

### 3.3 Attention / KV / RoPE（plain 为生产请求；无 weight packing）

| 算子 | act / weight / state / output | plain |
|---|---|---|
| Attention | F32 / F32 / F32 / F32 | `cpu::attention_f32_kv32_reference` |
| Attention | F32 / F32 / F16 / F32 | `cpu::attention_f32_kvf16_reference` |
| Attention | F32 / F32 / BF16 / F32 | `cpu::attention_f32_kvbf16_reference` |
| Attention | BF16 / BF16 / BF16 / BF16 | `cpu::attention_bf16_kvbf16_reference` |
| Attention | F16 / F16 / F16 / F16 | `cpu::attention_f16_kvf16_reference` |
| KVCacheUpdate | F32 / F32 / F32 / — | `cpu::kvcache_update_f32_kv32_reference` |
| KVCacheUpdate | F32 / F32 / F16 / — | `cpu::kvcache_update_f32_kvf16_reference` |
| KVCacheUpdate | F32 / F32 / BF16 / — | `cpu::kvcache_update_f32_kvbf16_reference` |
| KVCacheUpdate | BF16 / BF16 / BF16 / — | `cpu::kvcache_update_bf16_kvbf16_reference` |
| KVCacheUpdate | F16 / F16 / F16 / — | `cpu::kvcache_update_f16_kvf16_reference` |
| RoPE | F32 / F32 / — / F32 | `cpu::rope_f32_reference` |
| RoPE | BF16 / BF16 / — / BF16 | `cpu::rope_bf16_reference` |
| RoPE | F16 / F16 / — / F16 | `cpu::rope_f16_reference` |

KVCacheUpdate 为 state-only step，output_dtype undefined；Attention 的 weight fallback 跟随 activation，KV dtype 由 state_dtype 携带。

### 3.4 小算子与语义-only 算子

| 算子 | act / output | plain | 备注 |
|---|---|---|---|
| Silu | F32 / F32 | `cpu::silu_f32_reference` | |
| SiluMul | F32 / F32 | `cpu::silu_mul_f32_reference` | O2 fusion 产物 |
| SiluMul | BF16 / BF16 | `cpu::silu_mul_bf16_reference` | |
| SiluMul | F16 / F16 | `cpu::silu_mul_f16_reference` | |
| ElementwiseMul | F32 / F32 | `cpu::elementwise_mul_f32_reference` | |
| Add | F32 / F32 | `cpu::add_f32_reference` | |
| Add | BF16 / BF16 | `cpu::add_bf16_reference` | |
| Add | F16 / F16 | **unsupported** | 无 F16 descriptor（FP16 A16 residual 缺口） |
| Argmax | F32 / Int64 | `cpu::argmax_f32_reference` | |
| Softmax | F32 / F32 | **unsupported** | 语义定义存在；Attention 内部 softmax 已执行 |
| Reshape | F32 / F32 | **unsupported** | 语义-only |
| Permute | F32 / F32 | **unsupported** | 语义-only |
| Reorder | F32 / F32 | **unsupported** | 语义-only |
| Cast | F32→F16、BF16→F32 | **unsupported** | 语义-only；FP16 A16 模型路径依赖 |

Add 另注册 Double/Int32/Int64（非 Llama 主链，未列入）；语义-only 算子的常量折叠成功不代表动态执行可用。

## 4. 覆盖缺口与路径不对称（源码核对）

| 观察 | 源码依据 | 后续工作包 |
|---|---|---|
| FP32 Linear 已接入 scalar/AVX2 生产入口；MatMul 仍为 reference，不扩张未测 consumer | [linear_entry.cpp](../../src/backend/cpu/kernels/linear/linear_entry.cpp)、[matmul_entry.cpp](../../src/backend/cpu/kernels/matmul/matmul_entry.cpp)、`GEMM_HAS_AVX2_FMA_KERNEL=1` | CPU-03 |
| F32 B-panel candidate priority 10，identity optimized 20；布局 promotion 继续由完整证据决定 | [linear_entry.cpp:999](../../src/backend/cpu/kernels/linear/linear_entry.cpp#L999)、[cpu_backend.cpp:405](../../src/backend/cpu/cpu_backend.cpp#L405) | CPU-03 promotion 决策 |
| RmsNorm F32 plain＝AVX2/FMA（priority 20），packed-identity＝reference（priority 10）→ O2/packed F32 RMSNorm 落标量 | [rmsnorm_entry.cpp:534](../../src/backend/cpu/kernels/rmsnorm/rmsnorm_entry.cpp#L534)、[rmsnorm_entry.cpp:671](../../src/backend/cpu/kernels/rmsnorm/rmsnorm_entry.cpp#L671) | CPU-05（fused-vs-unfused） |
| W16/A32 Linear/QKV/GateUp 已接入 SIMD/blocking，reference 保留为 ISA/布局/数值 fallback | [Linear 优化方案](linear/linear-optimization.md) | CPU-02（证据见机器记录） |
| Embedding W16 已有 F16C/AVX2 optimized 默认 | [embedding_entry.cpp](../../src/backend/cpu/kernels/embedding/embedding_entry.cpp) | 复用，不重复建设 |
| Add F16 缺失（BF16 存在） | [add_entry.cpp](../../src/backend/cpu/kernels/add/add_entry.cpp) | CPU-06 / 精度专题 P4-B |
| QKV/GateUp plain unsupported（O1 不融合，by design）；MatMul packed unsupported | [optimize_graph.cpp:14](../../src/compiler/optimize_graph.cpp#L14) | 非缺口，记录边界 |
| Attention/KVUpdate/RoPE/小算子主链＝reference | 各 entry | CPU-04 / CPU-05 |
| Softmax/Reshape/Permute/Reorder/Cast standalone kernel 缺失 | [op_type.h](../../include/aethermind/operators/op_type.h) | CPU-06（consumer 驱动） |
| A16 编译关闭 QKV/GateUp/AddRmsNorm fusion；AddRmsNorm A16 kernel 缺失 | [model_compiler.cpp:23](../../src/compiler/model_compiler.cpp#L23) | CPU-05 / 精度专题 |
| FP16 A16 kernel 存在但高层 Load 拒绝（F32 residual/Cast 未完成） | [executable_model.cpp](../../src/inference/executable_model.cpp) | CPU-06 / 精度专题 P4-B |
| AVX-512 native 已编译、本机不可选；真实硬件数值/性能 not-run | §1 effective features | CPU-10 |

## 5. 编译预设 / fusion / packing 映射

| 预设 | opt_level | semantic pipeline | 权重 | fusion |
|---|---|---|---|---|
| O1Plain | 1 | ConstantFolding + DCE | plain | 无 QKV/GateUp/SiluMul/AddRmsNorm |
| O2Packed | 2 | + QkvLinear/GateUpLinear/SiluMul/AddRmsNorm fusion + DCE | identity-packed | 启用（A16 除外） |

A16（activation=F16/BF16）在 [model_compiler.cpp:23](../../src/compiler/model_compiler.cpp#L23) 强制关闭 QKV/GateUp/AddRmsNorm fusion，只跑 standalone A16 kernel 集。预设编号、fusion 与 packing 本身不构成加速证据；每次优化须同时对照 Prefill/Decode、O1/O2、fused/unfused。

## 6. 证据记录（追加式）

### CPU-01 跨算子选核身份审计与公共基准冻结

- commit：`cab0c145`，dirty（仅文档改动，无源码变更；library 与 HEAD 源码一致）。
- 命令：
  - 探针编译 `g++ -std=c++20 -O2 -Iinclude /tmp/aethermind-cpu01-dispatch-probe.cpp -Lbuild/src -lAetherMind -Wl,-rpath,$PWD/build/src -o /tmp/aethermind-cpu01-dispatch-probe`；运行 `/tmp/aethermind-cpu01-dispatch-probe`。
  - `python3 tools/verify_llama_numerical_reference.py --root .models/inference-numerical --expected tests/fixtures/inference_numerical/checksums.json`（verified）。
  - `python3 tools/prepare_kv16_reference.py verify --output .models/kv16-reference`（verified）。
  - 源码核对：rmsnorm packed F32 reference-only、B-panel priority tie、`optimize_graph.cpp` 预设 pipeline、`add_entry.cpp` 无 F16。
- 结果：157 行 selector（Prefill/Decode × plain/packed），121 resolved / 36 unsupported；packed recipe-query 全部 agree。机器/构建身份与代表模型 revision 冻结于 §1–§2。关键不对称：FP32 矩阵生产＝reference、RmsNorm F32 packed 落标量、Add F16 缺失、语义-only standalone kernel 缺失。
- raw：`benchmark-results/operators/cpu01/20261117T014540Z_cab0c145_desktop-54h5mmi_cpu01-dispatch-audit/`（探针源码、逐 selector TSV、metadata、stderr；跨算子 workstream 归集，非单算子性能记录；本地 gitignored，无远端 retention）。
- 结论：**Accepted（覆盖台账 + 基准冻结 + 选核身份）**。本次无性能采样，不调整 priority，不推导加速收益；native AVX-512 执行/数值/性能 not-run。同实现噪声与主路径耗时分布见下条；单算子性能证据见各算子 `benchmarks/`。

### CPU-01 同实现噪声 floor 与主路径耗时分布（tiny fixture）

- commit：`cab0c145`，dirty（仅文档）。model benchmark 二进制本轮重建至 HEAD（`build/validation/benchmark-model` preset，Release、sanitizer=none、AVX2=ON；旧二进制 11-11 已过期）。
- 命令：4 次独立进程 `python3 tools/run_model_benchmark.py --binary build/validation/benchmark-model/tests/benchmark/aethermind_model_benchmark --model .models/inference-numerical/mha-untied/model --prompt 1,7,3 --max-new-tokens 3 --preset o1-plain --warmup 1 --repetitions 20 --mode timing --variant cpu01-noise-aa-{1..4}`；随后 `python3 tools/verify_model_benchmark.py --report <aa-1>/report.json --noise-floor <aa-2>/report.json <aa-3>/report.json <aa-4>/report.json`。
- workload：O1/plain、require-fp32、KV F32、activation F32；cache policy `pre_read_by_identity_collection`（loading 非冷启动）。
- 结果（4 进程 median，单位 ns；noise floor 为同实现跨进程离散观测）：

| 区间 | run1 | run2 | run3 | run4 | noise floor |
|---|---:|---:|---:|---:|---:|
| `generate.duration_ns`（公开请求） | 1421290 | 1286584 | 1334722 | 1138165 | 24.88% |
| `internal.first_token_request_ns` | 978678 | 811128 | 752426 | 712709 | 37.32% |
| `internal.prefill_execute_ns` | 919352 | 762267 | 708069 | 669894 | 37.24% |
| `internal.decode_execute_total_ns` | 556282 | 478311 | 434126 | 419070 | 32.74% |
| `internal.request_setup_ns` | 55787 | 48155 | 46987 | 39708 | 40.49% |
| `preparation.load_ns` | 132265 | 111123 | 105086 | 92545 | 42.92% |
| `preparation.compile_ns` | 202754 | 182768 | 174057 | 154897 | 30.90% |
| `preparation.prepare_ns` | 71651 | 61588 | 57943 | 52497 | 36.49% |
| `preparation.runtime_ns` | 5453 | 4919 | 4453 | 3952 | 38.00% |

  `generate.actual_tokens_per_second` noise floor 24.87%；`internal.decode_setup_ns` 最高 49.21%。完整逐指标 median/pairwise 见 raw `noise-floor.json` / `noise-summary.txt`。
- 关键告警：tiny fixture 为开销主导，几乎所有指标在 run1→run4 呈**系统性下漂**（WSL2 跨进程非平稳，即便 identity 预读）。这是局部离散观测，`status=observed_noise`、`performance_claim=none`、判读 **Needs More Data**；不得据此设全平台阈值或宣称收益。主路径耗时分布已可追溯，但 promotion 阈值须按机器×组在真实形状/streaming 上另行冻结（工作流 §7.2）。
- raw：`benchmark-results/operators/cpu01/20261117T015438Z_cab0c145_desktop-54h5mmi_cpu01-noise-floor/`（noise-floor.json、summary、4 份 driver log、metadata）；模型级原始 artifact 在 `benchmark-results/models/20261117T015331Z…aa-1-9bd19702/` 等 4 个 run 目录（含 report/summary/machine/invocation/source.patch）。
- 结论：**Accepted（同实现噪声与主路径耗时分布观测）**。CPU-01 退出条件的噪声/分布项已取得限定证据；不调整 priority，不推导加速。后续 CPU-02/03/04 的 A/B 须用独立进程交错、真实形状与 streaming，并以本机该组噪声 floor 判读。

### CPU-02/03 后续生产闭环

- candidate 为 `e502fe11` + F32 cold-driver 修正；旧源码、旧库与 candidate 分别保存，M0 基准不被覆盖。
- §3/§4 已同步当前矩阵 default 身份；实际 12 组 baseline/candidate × F32/F16/BF16 × O1/O2 model plan/Generate trace 均已保存，完整 selectors/recipes 与 kernel 名称可追溯。
- [CPU-02 Linear 证据](linear/benchmarks/desktop-54h5mmi-linear.md#cpu-02-w16a32-生产优化与-cpu-03-关联闭环) 与 [CPU-03 GEMM 证据](gemm/benchmarks/54h5mmi-gemm.md#cpu-03-fp32-gemm-生产闭环与布局-promotion-决策) 分别记录 935 项 Release 回归、适用 sanitizer、两类编译 fallback、同机交错 A/B、48 份官方模型报告和 B-panel 拒绝提升的决策。该条只关联证据，不把 CPU-01 其余待办或产品 7B/4k 验收标成完成。
