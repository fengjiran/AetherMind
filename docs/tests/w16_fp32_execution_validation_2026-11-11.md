# P2 CPU W16/A32 执行验收

状态为 Current。本记录对应 `ea95702e` 加当前工作树补丁，平台为 Linux/WSL x86-64、GCC 14.2.0、C++20。实现入口与支持矩阵见 [W16 指南](../guides/w16-fp32-inference.md)，分阶段合同见 [P2 提案](../improvement-plan/11-fp16-bf16-execution-support.md)。

## 交付范围

| 项目 | 当前结果 |
|---|---|
| 权重 policy | 保留 F16 或 BF16 矩阵，仅转换 Norm，actual dtype 按角色准入 |
| 图与序列化 | Embedding 显式 F32 output，undefined 保留跟随语义，裸 serde 逐字节兼容 |
| O1/plain | Embedding、Linear/LM head、F32 Norm/activation/KV 完整生成 |
| O2/identity-packed | 两字节 identity/composite、W16 QKV/GateUp，完整生成 |
| Embedding 实现 | portable reference、effective-feature-gated F16C/AVX2 行级转换 |
| Linear 数值 | K 升序、F32 独立乘加，reference TU 的 `-ffp-contract=off` 已核对 |
| backing 与预算 | owner identity 去重、W16 retained backing、Norm-only 预检、OOM 原子性与 lifetime |
| Decode malloc-family | O1/O2 在 binding 后的执行区间均为零，Release 单独验证 |

## 本轮验证

- Release 最小受影响目标 `aethermind_unit_tests`、`aethermind_benchmark`、`aethermind_model_benchmark` 构建通过。先跑最窄 kernel/预算门禁，再汇合模型与相关回归，最终 468 tests / 39 suites 全部通过。
- Debug ASan/UBSan 对相同 468 项运行，457 通过，11 项 malloc-interposer 相关测试因 sanitizer 拦截环境显式跳过。这些测试在上面的 Release 运行全部通过。设置为 `ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1`，本轮结果覆盖地址与 UB 检查，LSan 泄漏检查另列环境验收。
- model benchmark report verifier / driver 的 87 项 Python 合同测试通过，真实二进制 CLI 的 14 项数值资产/驱动合同测试通过。ISA compile-command manifest 检查通过。
- 468 项回归覆盖 scalar/selected Embedding 的完整 65,536 位模式、未对齐/尾部/feature-policy fallback、W16 Linear 的 stride/shape/alias/零 M/N/K 与 exact selector、QKV/GateUp/packed dtype 验证、Embedding inference/serde/builder，以及 P1 转换行为。
- 模型测试覆盖两种矩阵 dtype 与三种 Norm source dtype、MHA/GQA、tied/untied、known-hint mismatch、逐阶段 logits/KV 与排名误差、Generate 确定性、EOS/容量、OOM 重试、owner lifetime 和 O1/O2 Decode malloc-family 计数。

完整命令、环境、源码 SHA-256、GoogleTest XML、build/test logs、初次失败与修复后的运行结果均保存在本地 raw。

```text
benchmark-results/operators/w16_execution/
  20261111T072153Z_ea95702e_desktop-54h5mmi_closeout/
```

该记录针对受影响范围的聚焦回归。source hashes 与当前 dirty snapshot 绑定，其他历史测试数字按各自记录解释。

## 性能与资源

Embedding 的同构建 scalar/selected prepared A/B 使用 CPU affinity 0、同进程随机交错，每个 case 三次 repetitions，涵盖 tokens=1/16、hidden=4096、F16/BF16。所有 correctness guard 通过，具体中位数和范围见 [Embedding 机器记录](../operators/embedding/benchmarks/desktop-54h5mmi-embedding.md)。

另以现有 admission tiny fixture 对 F16/BF16 的 O1/O2 各运行一个生产 model benchmark report，每个 report 三次 repetitions，driver/verifier 均接受。报告保留实际 Load、Norm conversion、Compile、Prepare、Execute、Generate 和资源字段，路径列于 raw 的 `aethermind-p2-model-runs.json`。

矩阵逻辑存储为两字节，Norm 保持四字节，O2 另外持有对应 dtype 的 packed artifacts。最终 retained owner 账本按完整 backing footprint 计量。Linear 当前为 reference，实现及现有 baseline 的性能限制见 [Linear 机器记录](../operators/linear/benchmarks/desktop-54h5mmi-linear.md)。新 W16 B-panel 的布局评估采用后续 profiling 门禁。

当前验收覆盖本机与 tiny 测试模型，资源与时延按各 report 的计时区间解释。其他平台和生产规模模型质量由对应测试矩阵验收。

## W16 FP32 I/O 自然对齐修复

本节记录 `98604d79` 加当前工作树补丁的独立修复验证。W16 plain/identity-packed Linear、QKV 和 GateUp 在 params builder 检查实际读写的 F32 地址满足 `alignof(float)`，不依赖未知或较弱的 view alignment。空输出不要求对齐；零 K 仍检查写入的 output，未读取的 input 不受该检查约束。W16 权重继续使用 byte-safe 读取。

- Debug `aethermind_unit_tests` 目标构建通过；新增 56 项参数化边界测试全部通过，覆盖两种权重 dtype、四种 projection 路径、每个输出端口、零 M/N/K、空输出 split 和自然对齐但未满足 SIMD 对齐的合法地址。
- 扩大的受影响回归共 366 项，全部通过且无跳过，包含 W16 O1/O2 生成、转换预算、owner lifetime、Embedding 和 Linear/composite 回归。
- 原四个 F16/BF16 输入/输出未对齐 UBSan harness 均在准备阶段返回 `InvalidArgument`，不再进入 typed load/store。
- 独立构建完整 ASan/UBSan `AetherMind` 库，单独编译新增 GTest 源文件并链接该库。设置 `ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1`，56 项全部通过且无跳过。开启 leak detection 时，56 项测试仍通过，随后 LSan 因当前 ptrace 环境不受支持而 fatal 退出；本次不声明泄漏检查通过。
- 格式检查与 `git diff --check` 通过。源码 SHA-256、完整命令、GoogleTest XML、build/test logs 和 LSan 失败日志保存在 `benchmark-results/operators/linear/20261111T154712Z_98604d79_desktop-54h5mmi_w16-alignment-fix/`。
