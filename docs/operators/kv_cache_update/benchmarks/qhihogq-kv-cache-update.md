# KVCacheUpdate / Attention 绑定基准记录 — DESKTOP-QHIHOGQ

| 项 | 值 |
|---|---|
| CPU | Intel Core i9-12900H；WSL2 guest 暴露 20 个 logical CPUs |
| OS / kernel | WSL2，6.18.54.1-microsoft-standard-WSL2 |
| Compiler / build | GCC 14.2.0，C++20；harness、layout、session view 使用 `-O3 -DNDEBUG`，其余组件使用同一份冻结 Debug supporting library，非完整 Release build |
| 备注 | 单线程，affinity 为 CPU 19；不据此推断宿主物理核或稳定的百分比收益 |

## 单次操作内复用已验证的 layout

- commit：`751686776a6801f05964d87d41131e390ed8eea1`，工作树 dirty。baseline 为去重前、已含诊断 OOM 修复的源码快照；candidate 为去重后的四个 layout/view 源码及头文件。逐文件 hash、supporting library hash 与确切命令保存在 raw artifact 的 metadata 中。
- 合同：每个公开绑定或数据访问入口重新验证当前 layout、session 和 backing buffers；内部 helper 只复用同次操作的验证结果，保留索引、乘加溢出检查，不缓存跨调用结果。K/V 共用一次 layer offset 计算。
- 命令：raw 中的 `build_benchmark.py baseline`、`build_benchmark.py candidate`、`run_benchmark.py`；构建及运行时的绝对路径完整保留在 `benchmark_*/metadata.json` 和 `benchmark_results.json`。复现应使用新的输出目录，避免覆盖原始样本。
- workload：FP32，2 层，每层 2 个 Q/KV heads，head dimension 2，Decode 前 committed context 为 8 / 128。真实 kernel 经 `Backend::PrepareKernel` 解析为 `cpu::kvcache_update_f32_reference` 和 `cpu::attention_f32_reference`；prepared contexts 在进程初始化后复用。
- 计时范围：一次 `Executor::Execute(decode_plan, context)`，包含 KV update、Attention 和完整 plan commit。每步计时前通过 `ResetSession` 和仅含 KV update 的 Prefill 恢复 context；初始化、Prefill、结果检查均在计时外。query/key 为零、Prefill value 为 3、Decode value 为 5，Attention 输出检查为 `3 + 2 / (context + 1)`。这是固定深度的执行片段，不是完整模型或连续 Generate。
- repetitions：每进程 warmup 100 步、保留 4000 步；每个 context 4 组独立进程 A/B，AB/BA 顺序交错，共 16 个进程、64000 条逐步样本。全部样本保留。

| context | 各进程对的 median 延迟比 `baseline / candidate - 1` 范围 | 判读 |
|---|---:|---|
| 8 | −0.84% ～ +3.41% | 方向不稳定 |
| 128 | −1.46% ～ +3.79% | 方向不稳定 |

- correctness：Debug 聚焦测试 167/167，含原 25 类诊断 OOM、9 项绑定/状态回归及 O1/plain、O2/packed tiny Llama；Decode 零分配检查通过。受影响源码及回归测试以 `-O2 -DNDEBUG` 单独重编，34/34 通过。基准检查 Attention 输出、最终 KV 值与每次 commit watermark；父代理另行复核并运行两种 variant、两种 context 的 smoke。
- 调用计数：在 Debug 生产执行入口包围计数，单层 KV update + Attention 的 Prefill 和 Decode 均从 20 次 layout 校验降到 4 次；双层仅 KV update 从 26 次降到 5 次。每层一个 update + Attention 的完整成功执行从 `18L + 2` 降为 `3L + 1`。此计数不是延迟测量。
- raw：`benchmark-results/operators/kv_cache_update/20261017T030430Z_751686776a68_DESKTOP-QHIHOGQ_kv-layout-validation/`。含 baseline/candidate 快照、独立 harness、全部样本、构建/符号绑定证据、correctness 与调用计数记录，以及 `ARCHIVE_SHA256SUMS`。
- 结论：去除单次操作内的重复验证已验证；延迟收益 **Needs More Data**。本次短样本和混合构建不足以证明稳定加速，不作完整 Release 或完整模型的性能结论。
