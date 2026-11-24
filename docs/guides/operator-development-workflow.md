# AetherMind 算子开发与优化工作流

## 1. 目的与适用边界

本文定义 AetherMind 所有算子开发、reference kernel、optimized kernel、packing/layout、fusion 和 threading 变更的统一工程流程。它冻结的是职责、证据门禁、文档角色和追溯关系，不冻结某个算子的具体优化顺序、tile、阈值或 ISA 选择。每个算子根据机制选择工作包，但不得绕过本文定义的 correctness、production-path 和证据门禁。

## 2. 核心原则

1. **语义先于实现**：先冻结 input/output、dtype、shape、stride/layout、alias、zero-size、overflow 和 numerical semantics，再实现 kernel。
2. **reference 与 optimized 分离**：reference 是 correctness oracle，不为追求性能而修改成难以审计的实现。
3. **micro benchmark 不代表产品性能**：必须同时验证真实 registered/prepared production path。
4. **性能结论必须可追溯**：绑定 commit、机器、benchmark 命令和 raw 数据；无上下文的数字不记录。

## 3. 工作包与门禁（O0–O4）

先确定本次交付目标。新增算子先完成 O0、O1 和生产接入检查，即可在已验证范围内交付 reference。已有算子复核其契约和接入证据后，可从 O2 复现基线开始。

| 工作包 | 工作与退出条件 |
|---|---|
| **O0 语义与合同** | 定义端口、dtype、rank/shape、stride/layout、alias、空张量、溢出、数值语义和错误行为。区分算子允许的输入与具体后端支持的子集。 |
| **O1 Reference** | 用可手算样例或独立数值对照验证 reference，覆盖声明支持的范围和错误边界。保留可审计的实现，供优化路径做差分测试。 |
| **生产接入检查（O1 后）** | 图算子用最小图经过 lowering、计划构建、实际绑定和执行，核对选中的 descriptor、输出和错误传播。backend primitive 通过真实 consumer 验证。声明模型覆盖时，再从对应模型入口验证该步骤或融合路径。 |
| **O2 Benchmark baseline** | 分别标识正确性 reference、当前默认实现和候选。用同一 selector 与 workload 测量已选核、已绑定参数的算子；首次建立基线时可用 reference。将绑定、准备和稳态执行分别计时。 |
| **O3 方案与路线** | 写明瓶颈假设、目标 workload、拟做的决定、预期收益和允许回退范围。非平凡方案及备选记入本算子的优化提案，状态记入算子 README。 |
| **O4 实验与调优** | 将候选与独立 reference 对照，并与 O2 的默认路径做配对性能比较。按预定条件裁决，在 §4 格式的记录中给出适用机器、dtype、phase、shape 和决定。提升默认优先级前，验证实际选核会受影响的 shape；覆盖不足时缩小启用范围或保留候选。 |

计划构建时选核，具体视图在绑定时校验，可变 KV 状态在执行时检查。RMSNorm 的完整示例见[算子系统设计](../operators/算子系统设计.md)。

## 4. 记录文件

证据记入 `docs/operators/<op>/benchmarks/<machine>-<op>.md`：**一机器一文件**，文件名只含机器名与算子名。日期、工作包编号、"实验日志/验证报告/分析"等分类都不进文件名，需要区分时用文件内的章节和记录条目。每台机器只写自己的文件，数值不跨文件引用；未采集的机器不建空文件。

机器文件头记录初始环境，每条实验注明与文件头的差异。记录只追加、不覆盖，失败实验同样保留；按追加顺序排列，标题不写日期（时间由 raw run-id 与 commit 定位）。

````markdown
# <算子名> 基准记录 — <机器名>

| 项 | 值 |
|---|---|
| CPU | <型号/核数/SMT/关键 ISA> |
| OS / kernel | <发行版/kernel> |
| Compiler / build | <compiler + build flags> |
| 备注 | <WSL2/affinity/拓扑等注意事项> |

## <被测实现或变更，一句话>

- 决策与范围：<要决定什么；机器、dtype、phase、shape、缓存状态、接受条件>
- 身份：<基线与候选的 commit、构建差异、effective ISA、实际 descriptor/recipe；dirty 源码的恢复信息>
- 命令与边界：<测试与基准命令；计时包含的步骤、预热和线程配置>
- 结果：<正确性；独立 A/B 对数、每进程 repetitions、两侧中位数、配对差值及噪声；非计时项按实际填写>
- raw：`benchmark-results/operators/<op>/<run-id>/`
- 结论：<Accepted / Rejected / Needs More Data；具体对象、适用范围、是否启用和待补项>

````

## 5. Benchmark 与原始数据规范

### 5.1 元数据

- **文件头（每机器一次）**：CPU model、核数/SMT、OS/kernel、compiler + build flags。
- **每条记录**：按 §4 模板记录。compiler、build flags、OS 或微码发生变化时，在该条记录注明。用于生产决策的 dirty 构建附可恢复的补丁（含影响结果的未跟踪文件）或源码快照。
- **全档（仅 end-to-end 结论、production 决策或跨机器比较时补记）**：date/time、baseline/candidate 身份、CPU stepping/microcode、effective CPU features、thread/affinity/NUMA、governor/turbo/SMT、memory configuration、JSON/checksum。

缺少 commit、命令或 artifact 的记录只能作为本地观察，不能用于 production 决策。

### 5.2 Raw 数据

gitignored `benchmark-results/operators/<op>/<run-id>/`（run ID 格式 `YYYYMMDDTHHMMSSZ_<git-sha>_<host-id>_<variant>`）。大量 machine-specific JSON 不进 `docs/`；需长期保留时用 CI artifact，在记录文件中记录 artifact ID、URL、checksum 与 retention。

## 6. 基准比较规则

- 默认在同机同配置下交错启动独立进程，交替使用 A→B 和 B→A 顺序，保存全部 repetitions。先取每个进程的中位数，再按相邻 A/B 对计算差值或百分比，注明指标单位和改善方向。
- 分开报告独立进程对数、每进程重复数和配对差值的分布。调试时可用同进程交错比较以共享进程环境；注明运行顺序及缓存影响，并用独立进程对验证结论能否跨启动复现。
- 按目标 workload 区分 Decode/Prefill、hot/streaming。逻辑访问字节和实际 DRAM 流量各自标明。按 §7.2 估计同机同组噪声；配对差值不足以区分收益与噪声时，记录 Needs More Data。

## 7. 优化方法

### 7.1 针对目标负载验证改动

从生产路径上耗时的 shape 和 phase 入手，一次验证一个瓶颈假设。例如优化 Decode 的矩阵乘时，用单 token 的实际 shape 测延迟；若声称提升生成速度，再测同一模型的每 token 延迟。具体的 tile、指令和缓存策略见[各算子方案与实测记录](../operators/README.md)。

| 改动 | 需要核对的边界 |
|---|---|
| ISA 或 SIMD 内核 | 分别验证缺少必要 ISA 时的选核，以及 ISA 满足但视图布局不满足时的返回状态或回退。数值和性能在支持该 ISA 的机器上验证。以向量宽度前后一项测试尾部读写；只有布局保证最后一行之后仍有足够可读 padding 时才允许 over-read。 |
| 权重打包 | 打包端和 kernel 须匹配同一 recipe；物理字节格式变更使用新布局 ID。验证尾块，记录额外字节和含打包的准备时间，结合实际复用次数判断收益。 |
| 算子融合 | 用同一输入对照融合前后的实际执行计划、数值、workspace 和耗时。 |
| 算子内并行 | 先明确线程与临时存储的所有权、预算、同步完成及失败清理，再验证确定性、TSAN、prepared Decode 稳态零分配、单线程和目标线程数的表现。[CPU 算子路线图](../operators/cpu-operator-evolution.md) |

### 7.2 正确性与基准

按算子契约选数值对照和误差指标。例如归约可与高精度结果对照；改变 Softmax 近似时，检查输出误差、归一化和特殊值行为。

按 §6 的配对方式在同一机器、同一 workload 组上把基线与自身比较（A/A），估计噪声，再判读候选的 A/B 差值。进程内增加重复次数只能估计该次运行的波动；跨进程漂移需用独立进程对衡量。若 A/A 已触发该组回退阈值，转为人工判读。核类别与 SMT 结论需在裸机验证。
