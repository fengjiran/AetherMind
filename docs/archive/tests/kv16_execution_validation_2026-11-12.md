# P3 CPU KV16 执行验收

状态为 Current。本记录对应 `592df0e4` 加当前工作树补丁，平台为 Linux/WSL x86-64、GCC 14.2.0、C++20。入口与支持矩阵见 [W16/KV16 指南](../guides/w16-fp32-inference.md)，合同与冻结预算见 [P3 提案](../improvement-plan/11-fp16-bf16-execution-support.md)。

## 交付范围

| 项目 | 当前结果 |
|---|---|
| 权重与 KV | retained-W16 policy 可显式选择 F32/F16/BF16 KV；W 与 KV 独立，所有层 K/V 同型 |
| 方向 | 仅 F16/BF16 矩阵 policy 准入 KV16；F32 matrix + KV16、strict FP32 compatibility、C ABI/CLI 仍拒绝 |
| state 身份 | `KernelSelector::state_dtype` 严格匹配；覆盖 inference、lowering、lowered 验证、plan、registry/key 与 untrusted 校验 |
| 配置链 | `CpuRuntimeAssemblyOptions::kv_dtype`、`ModelLoadOptions::kv_dtype` 和 `MakeKVCacheOptions` 显式选择；Load 在转换与 packing 前校验既有 Manager 的物理 dtype |
| 写边界 | KVCacheUpdate 在 K 经 F32 RoPE 后、V 从 projection 输出时按冻结位合同 narrow；padding 与历史 bytes 不受影响 |
| 读边界 | Attention 从 cache widening 后以 F32 归约；reference TU 禁用 FMA contraction |
| 布局账本 | 逻辑两字节 payload 与 64-byte 对齐后的物理 KV 分别记账 |

## 独立校准与 fixture

- 校准器 [calibrate_kv16_reference.py](../../tools/calibrate_kv16_reference.py) 的 sha256 为 `0c1b0be3…aba6`，与 `tests/fixtures/kv16_reference/checksums.json` 的固定 pin 一致；它独立于 AetherMind 生成量化边界的 FP64-reduction oracle。
- 转换位合同覆盖全部有限 pattern、midpoint 邻点与随机模式；比较器 negative checks 与 5 个 accepted FP32 golden 逐位交叉验证通过，两次独立进程的 65 个规范产物逐位一致。
- 生产 fixture 由 [prepare_kv16_reference.py](../../tools/prepare_kv16_reference.py) 从 pinned 校准 bundle 导入：5 个 case（MHA tied/untied、GQA/MQA tail、2046→2048 near-limit），共 50 个受 SHA-256 约束的文件。`verify` 返回 `status: verified`，manifest 与 reviewed checksums 完全一致。
- 验收分为实现正确性（独立量化 oracle）与相对同 W16 KV32 的质量两层；阈值与排名规则来自 §2.8，不在候选失败时放宽。

## 本轮验证

- Release 完整单测 4,804 项 / 411 suites 通过；KV16 聚焦套件覆盖 kernel、事务、Load 链、reduced 模型验收与 calibrated fixture。
- `KV16CalibratedAcceptance.NearLimitFourCombinationsMatchKv32WithinFrozenBudgets` 在 mha-near-limit（2048）上通过：四种 W×KV 组合 × O1/O2 共 16 条轨迹，比较 2046 prefill + 2 decode 的 logits、完整 K/V、watermark、行身份与固定排名。
- 模型验收覆盖四组合的 Prefill/多次 Decode/Generate、EOS/容量、失败重试、owner lifetime、每层写入成功后才 commit，以及 binding 后 Decode malloc-family 零分配。
- ASan/UBSan 运行 KV16 全套件：31 项中 29 通过，2 项 malloc-interposer 用例因 sanitizer 拦截显式跳过（普通 Release 运行通过），无地址、UB 或泄漏错误；raw 保留 `ASAN_OPTIONS=detect_leaks=1` 的运行日志。
- Python 合同测试：benchmark verifier/driver 99 项、真实二进制 CLI probe 16 项、fixture 导入工具 6 项通过。`verify_model_benchmark.py` 已按 schema v2 校验 `kv_dtype`、padded KV 字节与 activation 资源字段。

## 性能与资源

- prepared-operator baseline（scalar reference，n=3，同一 build）：KVUpdate F16 decode 约 13.5µs、128-token prefill 约 1.73ms；BF16 约 13.6µs / 1.54ms。Attention Q=F32、KV16 的 decode（1024 visible）约 24ms，128-query prefill 约 2.0s 且 CV 约 38%，只作量级参考。记录见 [KVCacheUpdate](../operators/kv_cache_update/benchmarks/desktop-54h5mmi-kv-cache-update.md) 与 [Attention](../operators/attention/benchmarks/desktop-54h5mmi-attention.md)。
- 2-layer KVUpdate→Attention 加完整 plan commit 的 production execution baseline 共 8 个 case，正确性 guard 通过；n=3，单机观察，不代表模型或 token 吞吐。
- 模型级 12 份 report（F16/BF16 权重 × F32/F16/BF16 KV × O1/O2，每份 3 次重复）由 driver/verifier 接受，记录 Load、conversion、Compile、Prepare、Execute、Generate 与资源字段。该数据是同一 WSL2 机器上的诊断 baseline，不含候选加速或模型质量结论。
- 小 head_dim 下两字节元素的 padded row 可能与 F32 占用相同的 64-byte 对齐行，不能保证物理 KV 或进程内存减半；logical 与 physical bytes 分别报告。

## Raw 证据

```text
benchmark-results/models/kv16-calibration/20261112T053100Z_592df0e4-dirty_desktop-54h5mmi_kv16-step1/
benchmark-results/models/kv16-step10/
benchmark-results/models/kv16-closeout-model/20261112T085017Z_592df0e4d425_desktop-54h5mmi_baseline/
benchmark-results/operators/kv_cache_update/20261112T081904Z_592df0e4-dirty_desktop-54h5mmi_kv16-baseline/
benchmark-results/operators/kv_cache_update/20261112T085844Z_592df0e4_desktop-54h5mmi_kv16-execution/
```

上述目录保存脚本与环境身份、输入/oracle hash、固定预算下的全部测量、重复性报告、测试日志与 GoogleBenchmark JSON；未发布外部 artifact。

## 限制

- 仅验证本机 tiny fixture 与 scalar reference 路径；不代表其他 CPU、OS、context 或生产规模模型的质量、内存或时延。
- production ExecutionStep/模型级性能优化与 SIMD widening 尚未开展；本记录不声称 KV16 带来吞吐收益。
- TSAN 不适用于当前同步单请求 KV16 路径，本记录不作并发声明。
- C ABI v1 与独立 CLI 继续 strict FP32；KV16 只通过 C++ `ExecutableModel::Load` 显式选择。
