# AetherMind 算子开发与优化工作流

## 1. 目的与适用边界

本文定义 AetherMind 所有算子开发、reference kernel、optimized kernel、packing/layout、fusion 和 threading 变更的统一工程流程。它冻结的是职责、证据门禁、文档角色和追溯关系，不冻结某个算子的具体优化顺序、tile、阈值或 ISA 选择。每个算子根据机制选择工作包，但不得绕过本文定义的 correctness、production-path 和证据门禁。

## 2. 核心原则

1. **语义先于实现**：先冻结 input/output、dtype、shape、stride/layout、alias、zero-size、overflow 和 numerical semantics，再实现 kernel。
2. **reference 与 optimized 分离**：reference 是 correctness oracle，不为追求性能而修改成难以审计的实现。
3. **microbenchmark 不代表产品性能**：必须同时验证真实 registered/prepared production path。
4. **性能结论必须可追溯**：绑定 commit、机器、benchmark 命令和 raw 数据；无上下文的数字不记录。

## 3. 工作包与门禁（O0–O4）

算子的机制决定选择哪些工作包（优化实现与纯正确性/调参工作投入不同），但 correctness、production-path 与追溯门禁不可绕过。

| 工作包 | 内容与退出条件 |
|---|---|
| **O0 语义与合同** | 明确端口顺序、dtype/rank/shape、stride/layout、alias/in-place、zero-size/overflow |
| **O1 Reference** | 简单、可读、独立的 correctness oracle，覆盖完整合法合同 |
| **O2 Benchmark baseline** | 以 **prepared operator**（registered/resolved kernel + prepared params）为主门禁，microkernel 仅诊断 |
| **O3 方案与路线** | 非平凡的优化方案、备选与工作包依赖记入本算子 `<op>-optimization.md`；工作包状态记入其 README；过程证据不写进提案 |
| **O4 实验与调优** | 在 work file 追加 |

## 4. 算子工作文件骨架

新建 `docs/operators/<op>/benchmarks/<work-package>.md` 时复制以下骨架。

````markdown
# <算子名> <工作包> 工作文件

- **算子索引**: <docs/operators/<op>/README.md>
- **状态**: Open / In Progress / Under review / Closed
- **门禁状态**: correctness [ ] / production-path [ ] / 可追溯 [ ]

## 1. 目标与合同不变式

- 目标（一句话）：
- 合同不变式（端口/dtype/shape/stride/alias/zero-size/overflow/数值预算）：无变化则写"无变化"；有变化必须列明并同步 OperatorSchema/测试。
- 关联代码 / 关联测试：

## 2. 实验记录（追加式，失败保留）

### YYYY-MM-DD — EXP-001：<标题>

- 假设 / 预期机制：
- 验证命令（benchmark/test）与核心元数据（commit、dirty、CPU、OS/kernel、compiler+flags、raw artifact）：

```bash
# exact commands
```

- Correctness：`测试 | 结果 | 备注`（PASS/FAIL/NOT RUN）
- 结果摘要：`Case | Baseline | Candidate | Delta | Raw artifact`（不粘贴完整 JSON/终端输出）
- 分析与决定：Accepted / Rejected / Needs More Data；下一步：

## 3. 正式结论（冻结区，按 commit@date 追加，不覆盖）

### <commit-sha>@<YYYY-MM-DD> — <结论名称>

- 判定：Accepted / Rejected / Needs More Data
- 环境快照（核心 6 项；production 结论附全档）：
- correctness / numerical error 摘要：
- production-path benchmark 摘要：
- layout/alias/fallback、workspace/ownership、dispatch、并发（适用时）：
- 未覆盖项与结论边界：

## 4. 门禁判定

- [ ] correctness 与 safety 测试通过
- [ ] production-path benchmark 已运行且附核心元数据
- [ ] 结果可追溯（commit/机器/命令/raw artifact）
- [ ] 未运行或未证明的内容已标注

## 5. 相关链接

- design / issues / 原始数据路径：
````

## 5. Benchmark 与原始数据规范

### 5.1 两级元数据

- **核心（所有性能结论必填）**：git commit + working-tree dirty、CPU model、OS/kernel、compiler/version + build flags、benchmark command、raw artifact 引用。
- **全档（仅 end-to-end 结论、production 决策或跨机器比较需要）**，在核心之外增加：date/time、baseline/candidate 身份、CPU stepping/microcode、effective CPU features、thread/affinity/NUMA、governor/turbo/SMT、memory configuration、raw JSON/checksum。

缺少核心元数据的结果只能作为本地观察，不能用于 production 决策。

### 5.2 Raw 数据

gitignored `benchmark-results/operators/<op>/<run-id>/`（run ID 格式 `YYYYMMDDTHHMMSSZ_<git-sha>_<host-id>_<variant>`）。大量 machine-specific JSON 不进 `docs/`；需长期保留时用 CI artifact，在 work file 记录 artifact ID、URL、checksum 与 retention。

### 5.3 比较规则

- baseline/candidate 同机同配置、独立进程交错 A/B、保存全部 repetitions；
- Decode/Prefill 与 hot/streaming 分组报告；logical bytes 与实际 DRAM traffic 区分；
- 噪声或置信区间内重叠的差异不得宣称收益；microbenchmark 改善未经 production/integration 验证，不得提高 production priority。

## 6. 决策、评审与反模式

### 6.1 状态流转

work file `Open → In Progress → Under review → Closed`；门禁不过 → 继续实验 / Rejected / Needs More Data；通过 → closeout（descriptor priority、design、CHANGELOG、issues 按触发更新）。

### 6.2 PR 评审门禁

- [ ] reference 与 optimized 独立；语义/layout/alias/zero-size/overflow 合同未被意外缩窄；
- [ ] correctness 测试覆盖目标和 fallback；steady-state 计时边界正确；
- [ ] 性能结论包含核心元数据，microbenchmark 与 production-path 证据已区分；
- [ ] work file、design、CHANGELOG 按触发条件同步；未运行或未证明内容明确标注。

### 6.3 反模式

- 直接优化 reference kernel；只保存最好数据、删除失败实验；
- 用 microkernel GFLOP/s 宣称端到端收益；timed loop 内做无意 allocation/validation；
- baseline/candidate 跨机器或跨配置直接比较；把 raw JSON/长终端输出粘贴进文档；
- 没有 exact recipe/ownership 证据修改 packed layout；没有 runtime threading contract 就在 kernel 内建线程。

## 7. 优化方法

### 7.1 契约与 workload

- **契约清单**：输入输出 shape/dtype/layout/stride；contiguous/in-place/aliasing；align 要求；NaN/Inf 传播；累加精度与 fast-math 是否允许近似；epsilon/rounding/denormal 行为；overflow 与 zero-size；目标 workload（latency vs throughput、单核 vs 多核、ISA 与 fallback）；WSL2/容器会把混合拓扑伪造成对称 CPU（`taskset` 仅咨询性）。
- **Prefill vs Decode**：Prefill 大 GEMM、复用高 → packing/microkernel/多线程分块/epilogue 融合；Decode 每步 1 token、GEMM→GEMV、call overhead 敏感 → small-batch GEMV 专用路径、权重预打包、KV layout、低开销 dispatch、避免小算子起线程。

### 7.2 正确性与基准

- **Reference vs Baseline**：Reference 用于正确性对齐（可用 double/标准库、不追求性能）；Baseline 是性能对比基准（朴素实现）。
- **测试体系**：固定用例（最小/常见/大 shape、非 2 的幂与质数维、SIMD 整除与不整除、cache 边界、LLM 常见尺寸）；随机测试（uniform/normal、符号/量级/稀疏/极值）；特殊数值（`±0`/NaN/±Inf/denormal，fast-math 路径须明确处理）。
- **误差指标**：Elementwise 用 max abs/rel diff；Reduction/RN 用 max diff + double reference；Softmax 加 sum-to-one/argmax；GEMM 用 max diff/rel diff/必要时 ULP；LLM logits 用 top-1/top-k 一致性。
- **噪声 floor（方法权威）**：固定百分比回退阈值（如 5%）不是常量——噪声 floor 是"机器 × 组"属性。① 区分进程内方差（提高 repetitions 有效）与跨进程系统偏移（无效，须多进程 median 或同进程交错 A/B）；② 用同一实现两轮互比校验阈值（被误判则该组只能人工判读）；③ 判读用该机该组 floor，不用全局常数；④ 核类别/SMT 结论只在裸机采集。

### 7.3 理论模型与 Roofline

FLOPs 口径统一（MAC = 2 FLOPs）；Bytes 按实际内存层级流量估算（两遍扫描/写分配/非临时存储改变流量）；`AI = FLOPs/Bytes` 低则大概率 memory-bound，但瓶颈还可能来自 TLB/load-store/shuffle/reduction 链/branch/spilling/同步/NUMA remote；`Attainable = min(Peak Compute, AI × Memory Bandwidth)`，机器平衡点 `= Peak/Bandwidth`，峰值须实测（STREAM、FMA、AVX-512 降频、AMX）。

### 7.4 Profiling（CPU）

贯穿流程：`baseline → profiling → 瓶颈假设 → 单点优化 → 再 profiling`。工具：perf/pmu-tools（计数）、VTune/uProf（pipeline/memory）、likwid（拓扑/带宽）、numactl/taskset（绑定）。关键指标：IPC、L1/L2/LLC miss、DRAM 带宽、branch/TLB miss、vectorization ratio、stalled cycles、port pressure、spilling、remote NUMA。

### 7.5 算法级优化与融合

优先级：`减少计算/访存 > 减少中间 tensor > 提高数据复用 > SIMD/指令级 > 多线程`。融合方向：GEMM epilogue（Bias/Activation/Residual 在写回阶段完成）、RMSNorm/Linear/RoPE 与前后算子、score+mask+softmax、Softmax+value matmul、Dequant+MatMul。Fast math：不默认全局开启、每算子显式声明、LLM 敏感路径端到端验证、近似后通常需 Newton-Raphson。

### 7.6 内存层级优化

连续访问优先（cache line/预取/TLB/SIMD 都受益）；GEMM packing 目标=连续访问+复用+匹配 tile；KV layout 考虑 decode 按 head 连续与对齐；blocking 按寄存器→L1→L2→L3→NUMA 分层；**2 的幂次步长诅咒**（`hidden=2^N` 跨列访问映射同一 cache set，用 padding/swizzling/非 2^N blocking 化解）；align（大 buffer ≥64B、AVX2 32B、AVX-512 64B）；prefetch 只用于硬件难识别的访问；reduction buffer `alignas(64)` 防 false sharing；non-temporal store 只用于短期不读的大输出；量化边反量化边计算、blocking 取 `group_size` 整数倍。

### 7.7 SIMD / 指令级 / Micro-kernel

主循环 + tail（AVX-512 mask、AVX2 scalar/masked/over-read）；多累加器提升 ILP 但防 spilling；FMA 注意与非 FMA 路径结果不 bitwise equal；横向归约只在循环末尾一次；shuffle 高则改 layout/减少跨 lane；dispatch：`Scalar → SSE/NEON → AVX2 → AVX-512 → AMX`，启动测 feature、按 dtype/shape/ISA 选择、预留 fallback、各路径共享测试。

### 7.8 多线程与 NUMA

先让单线程达到合理效率再扩展（顺序：correctness → baseline → profiling → SIMD/memory → scaling → NUMA）。Executor 持长驻线程池（绑核/NUMA 分组/低开销 barrier）；切分维度：norm/softmax 按 batch/row（head）、GEMM 按 M/N tile、attention 按 batch/head/query block；reduction 用线程私有 partial 并对齐、避免 atomic；负载不均用 block cyclic/work stealing/动态调度；NUMA 用 affinity + first-touch + 按 socket 分配。CPU-first 引擎坚持：steady-state zero allocation、single-request low latency、clear fallback。

### 7.9 回归与停止

每次优化做正确性回归、多 shape/ISA 验证、profiling 对比、端到端影响；记录 p50/p90/p99、GB/s、GFLOPS、scaling、decode per-token、MFU/MBU。停止准则：达到目标、接近实测硬件上限、瓶颈落在物理带宽/指令吞吐极限、继续优化收益低于阈值、端到端收益不明显——阈值按算子/shape/平台确定。Autotuning：搜索空间 × 目标函数 × 策略（穷举/贝叶斯）× 缓存最优配置，目标硬件离线运行、保留 fallback。

### 7.10 典型算子策略速查

| 算子 | 关键优化点 |
|---|---|
| RMSNorm | 两遍 SIMD + 多累加器；`rsqrt` 由 contract 控制；hidden 固定值专用路径；batch 并行；与 residual/elementwise 融合 |
| Softmax | max/exp/sum 三段 SIMD、`inv_sum` 代替除法；融合 mask/scale/value matmul；不能直接 `exp(x)`；fast exp 验证误差 |
| GEMM/GEMV | loop order → packing → cache/register blocking → microkernel → epilogue 融合 → small shape/量化专用；decode 退化为 GEMV，重点权重带宽与 cache 复用 |
| RoPE | split-half 配对处理、sin/cos 预计算、减少 shuffle、head_dim 固定专用、decode 低开销 |
| Attention / KV | decode 瓶颈在 KV 读取带宽；block/paged cache、head_dim 对齐、prefetch 历史 block、streaming softmax、多 head/query 并行 |

### 7.11 工程落地

- **Kernel dispatch**：按 op type → dtype/weight_format/phase 结构化过滤 → CPU 按 `effective_features ⊇ cpu_requirements` 过滤 → priority 选择 → shape 特化 kernel 内二次分发 → fallback 必须存在。`CpuFeature` 模型见 `docs/designs/dispatch_design.md`。
- **Steady-state zero allocation**：临时 buffer 由 workspace 预分配；初始化阶段分配、推理稳态零分配；小临时用 stack/寄存器。
- **Benchmark 与 CI 三层**：Correctness CI（每次提交、覆盖各 ISA 与关键 shape）；Performance smoke（每日、宽松阈值）；Full benchmark（定期/手动、供优化决策）。