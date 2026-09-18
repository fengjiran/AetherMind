# GEMM 实验记录与验证报告索引

- **专项提案**: [CPU GEMM 优化方案](../../../improvement-plan/04-cpu-gemm-optimization.md)
- **工作流**: [算子开发与优化工作流](../../../guides/operator-development-workflow.md)

本目录是本算子实验过程记录与正式验证/分析报告的**唯一存放位置**（2026-09-18 由 `docs/logs/operators/gemm/` 与 `docs/tests/operators/gemm/` 两处合并）。

| 日期 | 类型 | 文件 | 结论 | 状态 |
|---|---|---|---|---|
| 2026-09-18 | 实验日志 | [g0-baseline-log.md](g0-baseline-log.md) | 本机 reference 基线、独立复跑与配对 A/B 噪声 floor；production 百分比级门禁仍需裸机补采 | Current |
| 2026-09-18 | 实验日志 | [g1s-scalar-log.md](g1s-scalar-log.md) | scalar 单入口 candidate 与 opt-in Linear descriptor（strict/exact 机制已按简化决策移除）；默认仍为 reference，未获 production acceptance | In Progress / Needs More Data |
| 2026-09-18 | 验证报告 | [gemm_g0_baseline_validation_2026-09-18.md](gemm_g0_baseline_validation_2026-09-18.md) | 本机 reference baseline Accepted；production gate Needs More Data | Current |
| 2026-09-18 | 验证报告 | [gemm_g0_paired_ab_validation_2026-09-18.md](gemm_g0_paired_ab_validation_2026-09-18.md) | 关闭 raw repetitions/streaming repeat/交错 A/B 三项；production gate 仍 Needs More Data | Current |
| 2026-09-18 | 分析报告 | [gemm_g0_roofline_analysis_2026-09-18.md](gemm_g0_roofline_analysis_2026-09-18.md) | Roofline 定位：canonical 形状 ≤27% cap；M=1 记忆侧 / M≥16 计算侧；访问顺序主导 | Current |
