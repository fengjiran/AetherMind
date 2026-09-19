# GEMM G0 配对 A/B 噪声 floor 验证报告

- **状态**: Current
- **日期**: 2026-09-18
- **专项提案**: [CPU GEMM 优化方案](../cpu-gemm-optimization.md)
- **工作包**: G0 合同与证据基线（本地补采）
- **Change Profile**: Optimized benchmark infrastructure / Tuning-only measurement protocol
- **Candidate commit**: `5cbd378695bb90d97fa4273a7359bf2bfea20e8f`（与 baseline 同实现）
- **Baseline**: 同实现、独立进程、相邻配对
- **采集机**: `DESKTOP-54H5MMI` — Intel Core Ultra 9 285H，SMT off，WSL2 kernel 6.6.87.2（同前置报告）
- **Raw artifact**: `benchmark-results/operators/gemm/20260918T023421Z_5cbd378695bb_DESKTOP-54H5MMI_g0-ab/`（gitignored、不随仓库分发，**仅存在于上述采集机本地磁盘**）
- **Artifact checksum**: `context.json` SHA256 `902717d37212ecf0946a6d5ecbeffa01f0d6d5112dcf242ad454679f3e2d967a`；数据产物清单见 `checksums.sha256`（`sha256sum -c` 通过）
- **前置报告**: [G0 baseline validation](gemm_g0_baseline_validation_2026-09-18.md)
- **关联 ADR**: 无

> **机器归属**：本报告量化的噪声 floor 是采集机 `DESKTOP-54H5MMI` 的属性，**不是** G1S/G1V candidate 在其他机器上可用的最小可信 delta；换机必须重新量化。下文“本机”一律指该采集机。

## 1. 验证目标与结论

本报告关闭前置报告 §10 门禁清单中的三项本地可补采项：保存全部 repetition 原始数据、streaming 独立复跑、交错 A/B 协议，并量化该 WSL2 机器上同实现相邻配对的噪声 floor。

最终判定：

- **Closed locally**：raw repetitions、streaming 独立复跑、交错 A/B 协议及噪声 floor 量化；
- **Needs More Data（不变）**：稳定百分比级 production 门禁（本机多个组 p90 超 5%）、bare-metal perf/governor/microcode、durable artifact URL。

本报告不证明任何 optimized kernel 收益；噪声 floor 是后续 G1S candidate 判读的最小可信 delta 参考。

## 2. 协议

- 5 组 × 相邻配对：microkernel、linear_hot、linear_streaming、linear_binding、packing；每组 A/B 背靠背采集，组 1/3/5 采 A→B、组 2/4 采 B→A（顺序平衡）；
- 10 个独立进程 run；未使用 `--benchmark_report_aggregates_only`，JSON 保留每 repetition 原始行 + 聚合行；
- 其余同 G0 baseline：单线程 `taskset -c 2`、10 repetitions、`--benchmark_min_time=1s`、Release（GCC 14.2.0）。

## 3. 结果

### 3.1 噪声 floor（|B−A|/A，aggregate mean real_time）

| 组 | cases | >5% | >10% | 中位 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|
| microkernel | 12 | 4 | 1 | 2.62% | 7.09% | 13.20% |
| linear_hot | 12 | 4 | 1 | 1.95% | 6.94% | 14.51% |
| linear_streaming | 10 | 4 | 1 | 3.08% | 17.13% | 17.13% |
| linear_binding | 12 | 11 | 7 | 12.08% | 15.32% | 24.22% |
| packing | 3 | 1 | 1 | 4.70% | 14.84% | 14.84% |

### 3.2 Canonical compute 关键 case（B vs A）

| Case（mean real_time） | A (ns) | B (ns) | delta |
|---|---:|---:|---:|
| K-contiguous microkernel M1 K4096 N4096 | 11 722 384 | 11 253 473 | −4.00% |
| K-contiguous microkernel M4 K4096 N4096 | 46 187 350 | 45 683 001 | −1.09% |
| K-contiguous microkernel M16 K4096 N4096 | 181 852 517 | 181 775 046 | −0.04% |
| N-contiguous microkernel M1 K4096 N4096 | 125 713 856 | 142 308 970 | +13.20% |
| hot M1 K4096 N4096 | 11 785 189 | 11 532 565 | −2.14% |
| hot M1 K4096 N11008 | 31 902 453 | 33 735 196 | +5.74% |
| hot M16 K4096 N4096 | 191 611 315 | 180 437 643 | −5.83% |
| streaming M1 K4096 N4096 | 12 174 654 | 11 458 087 | −5.89% |
| streaming M1 K4096 N22016 | 61 322 743 | 71 827 349 | +17.13% |

### 3.3 与顺序两轮（G0-BASELINE-001）的对比

顺序两轮独立进程（非相邻）的跨进程 delta 达 ±10–25%；相邻配对后 canonical compute 中位噪声降至约 2–3%。系统性漂移主要来自两次采集间隔内的宿主负载/频率状态变化，配对协议将其大部分抵消；残余噪声集中在 binding（亚微秒）、tail/小形状与个别大 N streaming case。

## 4. 门禁清单更新（相对前置报告 §10）

- [x] 保存全部 repetition 原始数据；
- [x] streaming 独立复跑；
- [x] 交错 A/B 协议（相邻配对 + 顺序平衡；噪声 floor 已量化）；
- [ ] 稳定百分比级门禁 —— 本机 p90 仍超 5%（binding/小形状尤其显著），保持 blocked；
- [ ] bare-metal perf/governor/microcode 证据 —— 本机环境不可得，保持 blocked；
- [ ] durable CI/object-storage artifact URL —— 无基础设施，保持 blocked（`checksums.sha256` 已就绪，便于上传后核验）。

## 5. 事实、推断与限制

**已验证事实**：

- 相邻配对把 canonical compute 噪声压到中位约 2–3%；
- binding 组（亚微秒）噪声中位 12%，不适合单次百分比判定；
- 10 个 run 全部 rc=0，`checksums.sha256` 核验通过。

**基于数据的推断**：

- G1S/后续 candidate 比较应采用相邻配对协议，并以各组噪声 floor（而非固定 5%）作为最小可信 delta 参照；
- production 5% 门禁仍需裸机环境。

**限制**：

- 仍为 WSL2 宿主共享环境，无法验证跨机器稳定性；
- 未增加重复配对轮数（仅 2 轮交叉），未覆盖长时频率漂移。

## 6. 后续动作

- G1S candidate 采集沿用 `collect_ab.sh` 配对协议与 `analyze_ab.py` 汇总；
- 裸机环境就绪后：重采完整协议（bare-metal perf、governor/microcode、稳定百分比门禁）；
- raw artifact 上传 CI/object storage 后回填 retention URL 与校验值。
