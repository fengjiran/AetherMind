# KVCacheUpdate 基准记录 — desktop-54h5mmi

| 项 | 值 |
|---|---|
| CPU | Intel Core Ultra 9 285H；客体可见 16 logical CPU；AVX2/FMA/F16C |
| OS / kernel | Ubuntu 24.04.1 LTS / 6.6.87.2-microsoft-standard-WSL2 |
| Compiler / build | GCC 14.2.0；build，Release -O3 -DNDEBUG，C++20；sanitizer=none |
| 备注 | WSL2；数值为有限样本观测，不推导真实模型加速 |

## P3 KV16 prepared reference kernel baseline（F32 activation → F16/BF16 cache）

- commit：`592df0e4`，dirty；含 P3 步骤 1–10 工作树补丁。
- 命令：`./build/tests/benchmark/aethermind_benchmark --benchmark_filter=RegisterKv16 --benchmark_repetitions=3 --benchmark_report_aggregates_only=true`
- 几何：8 KV heads，head_dim 128，capacity 2048；prepared params 一次构建，循环仅 `kernel.fn`（borrowed kv_append binding）。
- 结果（median，n=3）：
  - kvupdate_f16/1（decode）：13.5µs，75.8M items/s，144.6 MiB/s
  - kvupdate_f16/128（prefill）：1.73ms，75.7M items/s，144.4 MiB/s
  - kvupdate_bf16/1：13.6µs，75.4M items/s，143.9 MiB/s
  - kvupdate_bf16/128：1.54ms，85.1M items/s，162.2 MiB/s
- raw：`benchmark-results/operators/kv_cache_update/20261112T081904Z_592df0e4-dirty_desktop-54h5mmi_kv16-baseline/`
- 结论：Accepted（诊断 baseline）。scalar reference 路径，吞吐稳定（低 CV）；F16/BF16 相近。SIMD 窄化/写入优化为后续工作，须另有 prepared/production/端到端证据，不以此 microbenchmark 推导模型收益。