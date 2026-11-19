# 在研方向

当前优先事项如下。专题工作包、草案和历史记录见 [演进提案索引](improvement-plan/README.md)。

| 方向 | 当前目标 | 实施入口 |
|---|---|---|
| CPU 算子覆盖与性能 | 按生产路径 profiling 推进剩余热点和算子覆盖 | [CPU 算子路线图](operators/cpu-operator-evolution.md) |
| KV / Attention 演进 | 维护静态 KV 的布局预算和提交合同，按真实负载选择下一阶段 Attention 优化 | [KV Cache 与 CPU Attention](improvement-plan/08-kv-cache-and-attention-capability-evolution.md) |
| FP16 / BF16 执行 | 补齐 A16 的 AVX-512 真机数值与性能证据 | [FP16 / BF16 执行方案](improvement-plan/11-fp16-bf16-execution-support.md#64-性能资源与发布状态) |
| 生产验证与质量 | 补齐剩余模型数值、平台组合和交付验证 | [生产验证实施方案](improvement-plan/13-deepinfer-quality-and-validation.md) |

已知缺陷见 [问题清单](issues.md#未解决)，产品范围和验收标准见 [产品需求](product.md)。
