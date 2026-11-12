# Attention 基准记录 — desktop-54h5mmi

| 项 | 值 |
|---|---|
| CPU | Intel Core Ultra 9 285H；客体可见 16 logical CPU；AVX2/FMA/F16C |
| OS / kernel | Ubuntu 24.04.1 LTS / 6.6.87.2-microsoft-standard-WSL2 |
| Compiler / build | GCC 14.2.0；build，Release -O3 -DNDEBUG，C++20；sanitizer=none |
| 备注 | WSL2；prefill 样本 CV 偏高（系统噪声），数值为有限样本观测，不推导真实模型加速 |

## P3 KV16 prepared reference kernel baseline（Q=F32，K/V cache F16/BF16，F32 归约）

- commit：`592df0e4`，dirty；含 P3 步骤 1–10 工作树补丁。
- 命令：`./build/tests/benchmark/aethermind_benchmark --benchmark_filter=RegisterKv16 --benchmark_repetitions=3 --benchmark_report_aggregates_only=true`
- 几何：8 Q/KV heads，head_dim 128，visible context 1024；prepared params 一次构建，循环仅 `kernel.fn`（borrowed kv_read binding）；reference TU 禁 FMA contraction。
- 结果（median，n=3）：
  - attention_f16/1（decode）：~24ms（1 query × 1024 context）
  - attention_f16/128（prefill）：~2.0s，CV ~38%
  - attention_bf16/1：24.4ms
  - attention_bf16/128：~2.0s，CV ~37%
- raw：`benchmark-results/operators/kv_cache_update/20261112T081904Z_592df0e4-dirty_desktop-54h5mmi_kv16-baseline/`（与 KVUpdate 同一 run 的 JSON）
- 结论：Accepted（诊断 baseline）。scalar reference，compute-bound，prefill 受 WSL2 噪声影响（高 CV，仅作量级参考）。SIMD widening/F32 归约优化为后续工作，须另有 prepared/production/端到端证据与该机噪声 floor 判定，不以此 microbenchmark 推导模型收益。