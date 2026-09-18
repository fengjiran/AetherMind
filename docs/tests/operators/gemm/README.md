# GEMM 实验记录与验证报告索引

- **专项提案**: [CPU GEMM 优化方案](../../../improvement-plan/04-cpu-gemm-optimization.md)
- **工作流**: [算子开发与优化工作流](../../../guides/operator-development-workflow.md)

本目录是本算子实验过程记录与正式验证/分析报告的**唯一存放位置**（2026-09-18 由 `docs/logs/operators/gemm/` 与 `docs/tests/operators/gemm/` 两处合并）。

## 机器归属与可复核性

本目录的报告**按机器分别成立**，目前有两台采集机，二者数据并列、互不替代：

| 采集机 | CPU | 环境 | G0 状态 |
|---|---|---|---|
| `DESKTOP-54H5MMI` | Intel Core Ultra 9 285H（Arrow Lake-H，同构 16 核，SMT off） | WSL2 kernel 6.6.87.2，GCC 14.2.0 | Baseline Complete / production gate Needs More Data |
| `DESKTOP-QHIHOGQ` | Intel Core i9-12900H（Alder Lake-H，6P+8E 混合） | WSL2 kernel 6.18.33.2，GCC 14.2.0 | Baseline Complete / production gate Needs More Data |
| 其他任何机器 | — | — | **Not Collected** |

- 原始 artifact 目录 `benchmark-results/operators/gemm/<run-id>/` 被 `.gitignore` 忽略，**只存在于各自的采集机本地磁盘**，不随仓库分发。在其他机器上这些 run id 不可恢复、数值不可复核。
- 按提案 §6.7（baseline/candidate 必须同机、同配置）与 §6.5（不得跨微架构比较原始计数），任一采集机的数值**都不能作为另一台的基线或门禁参照**。
- 因此 G0 状态按机器计：某台机器 Baseline Complete 不代表其他机器 Complete；每台目标机必须各自采集并归档自己的 raw artifact 后才能判定。
- 与机器无关、可在任何机器由源码复核的部分：G0 benchmark/测试/采集脚本本身、correctness 契约测试、reference 反汇编为纯标量的事实、G1S candidate 入口与 fallback 边界、opt-in descriptor 注册逻辑。

| 日期 | 类型 | 文件 | 采集机 | 结论 | 状态 |
|---|---|---|---|---|---|
| 2026-09-18 | 实验日志 | [g0-baseline-log.md](g0-baseline-log.md) | `54H5MMI` | reference 基线、独立复跑与配对 A/B 噪声 floor；production 百分比级门禁仍需裸机补采 | Current |
| 2026-09-18 | 实验日志 | [g1s-scalar-log.md](g1s-scalar-log.md) | `54H5MMI` | scalar 单入口 candidate 与 opt-in Linear descriptor（strict/exact 机制已按简化决策移除）；默认仍为 reference，未获 production acceptance | In Progress / Needs More Data |
| 2026-09-18 | 验证报告 | [gemm_g0_baseline_validation_2026-09-18.md](gemm_g0_baseline_validation_2026-09-18.md) | `54H5MMI` | reference baseline Accepted；production gate Needs More Data | Current |
| 2026-09-18 | 验证报告 | [gemm_g0_paired_ab_validation_2026-09-18.md](gemm_g0_paired_ab_validation_2026-09-18.md) | `54H5MMI` | 关闭 raw repetitions/streaming repeat/交错 A/B 三项；production gate 仍 Needs More Data | Current |
| 2026-09-18 | 分析报告 | [gemm_g0_roofline_analysis_2026-09-18.md](gemm_g0_roofline_analysis_2026-09-18.md) | `54H5MMI` | Roofline 定位：canonical 形状 ≤27% cap；M=1 记忆侧 / M≥16 计算侧；访问顺序主导 | Current |
| 2026-09-19 | 实验日志 | [g0-baseline-log-desktop-qhihogq.md](g0-baseline-log-desktop-qhihogq.md) | `QHIHOGQ` | 本机首次 G0 采集；访问顺序归因复现且更强（16–18×，L1 驻留时仅 1.18×）；噪声分解为进程内 CV 0.03–0.04% vs 跨进程偏移 ≤5.17%；WSL2 拓扑伪造致 `taskset` 仅咨询性；G1S 已被 SSE2 自动向量化 | Current |
| 2026-09-19 | 验证报告 | [gemm_g0_baseline_validation_desktop-qhihogq_2026-09-19.md](gemm_g0_baseline_validation_desktop-qhihogq_2026-09-19.md) | `QHIHOGQ` | 本机 reference baseline Accepted（106+30 测试通过）；production gate Needs More Data；同一实现自我比较被判出 4 个 REGRESS，证明 5% 自动门禁在本机三组上不可用 | Current |
| 2026-09-19 | 分析报告 | [gemm_g0_roofline_analysis_desktop-qhihogq_2026-09-19.md](gemm_g0_roofline_analysis_desktop-qhihogq_2026-09-19.md) | `QHIHOGQ` | 本机 ceiling：P=130.21 GFLOP/s、Triad=25.32 GB/s、ridge=5.14；M=1 达记忆侧 27.7–29.4%，M≥16 达峰值 2.6–2.8% | Current |
