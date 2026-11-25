# 14 号提案审核报告：CPU GEMM 算子内多线程并行方案（CPU-08）

- **被审提案**：[14-cpu-gemm-intra-op-parallelism.md](14-cpu-gemm-intra-op-parallelism.md)（状态：Draft，CPU-08）
- **审核日期**：2026-11-26
- **审核方式**：文档逻辑审查 + 代码事实核对（未运行性能测试）
- **审核依据**：[CPU 算子路线图](../operators/cpu-operator-evolution.md)、[GEMM 优化方案](../operators/gemm/cpu-gemm-optimization.md)、[GEMM 证据索引](../operators/gemm/README.md)、[算子开发与优化工作流](../guides/operator-development-workflow.md)、[产品边界](../product.md)、`src/backend/cpu/kernels/gemm/` 与 `src/execution/` 实现
- **审核结论**：**修改后再审**（加权总分 6.9/10）

---

## 1. 提案核心内容梳理

### 1.1 核心目标

在 Runtime 层引入常驻 worker 池，为 FP32 GEMM/Linear 建立算子内（intra-op）多线程并行能力，用多核共享权重读取带宽来降低 Decode 单 token 延迟与 Prefill 延迟。首个实验限定为 **FP32 identity Linear 的单 token Decode**，核心待验证命题：多核读取权重节省的时间能否覆盖每次投影的派发与同步开销。

### 1.2 实施路径

| 阶段 | 内容 | 退出证据 |
|---|---|---|
| O2 串行基线 | 冻结 FP32 Linear 的 descriptor/recipe/形状/模型，测 prepared Decode/Prefill/Generate | 同机 A/A 噪声、投影耗时占比 |
| FP32 Decode | 常驻线程池 + 同步接口，M=1 identity Linear 按 N 分片；首组实验轮转 > LLC 权重 + hot 对照 | 单线程直通、失败收敛、稳态零分配、TSAN、带宽饱和点 |
| FP32 Prefill | 已选 driver 内分配 N tile，列块不足试验 M/N 二维分片 | 短/长 prompt 分开验证，比较重复打包成本、workspace 峰值、prepared 耗时、模型延迟 |
| 扩展 consumer 与精度 | 接入 QKV、GateUp，覆盖 W16 数值回退；B-panel 按独立布局验收 | 每新增范围单独作启用决定 |

### 1.3 时间节点

**未给出任何时间节点、里程碑排期或目标机器清单。** 仅各阶段附证据门禁。注：CPU 路线图 §4.1 明确"编号不代表严格串行顺序或工期承诺"，不强制要求工期估算，但目标机器清单与阶段间顺序约束仍属缺失（见 P0-4）。

### 1.4 资源需求

- 线程：`intra_op_threads=T`（默认 1），调用线程计入 T，Runtime 创建 T−1 个常驻 worker；部分启动失败时回收。
- scratch：Decode 走权重直读零新增 scratch；Prefill 需核对每线程 panel/累加缓冲容量，新增 scratch 的路径依赖 CPU-07（sizing、预算查询、绑定）。
- 未给出人力/工作量估算与采集机硬件配置。

### 1.5 预期收益

- 方向性收益：Decode 单 token 延迟（带宽受限，多核并行流式读取权重）、Prefill 吞吐。
- **未量化**：线程数、启用阈值、加速比均明示"待实测"；未给出带宽上限下的理论收益下界或止损线。

---

## 2. 逻辑一致性、可行性与路线图匹配度核查

### 2.1 代码事实核对表

| 提案论断 | 核查结果 | 代码依据 |
|---|---|---|
| "现有 FP32 AVX2 driver 在 M=1..8 与 M>=9 采用不同路径" | ✅ 属实 | `gemm_a32_w32_avx2.cpp`：`m∈[1,8]` 走 rows-per-pair 路径（约 L319），`m>=9` 走 tile-6 blocked 路径（约 L347/L364）；scalar 侧同有 `kScalarSmallMMax=8` 分界 |
| "W16 的 TryRunGemm* 可在写入部分结果后要求 reference 重算" | ✅ 属实 | `linear_a32_optimized.cpp` L36-38/L45-47：TryRun 返回 false 后调用 `RunLinearF32W16Reference` 全量重算覆盖输出。提案"先汇合再覆盖整份输出"的约束正确且必要 |
| "当前 Runtime 未提供生产 worker pool" | ✅ 属实 | `src/runtime/` 无 thread/worker 实现；`KernelContext`（`include/aethermind/backend/kernel_context.h`）无任何并行接口字段——"借用窄的同步并行接口"属新增设计而非现状，提案表述与事实无矛盾 |
| "Decode 从权重直读路径开始" | ✅ 属实 | FP32 identity Decode 为 direct-weight 路径；blocked/panel 仅用于 Prefill（[GEMM 证据索引](../operators/gemm/README.md)） |
| 分片"先按完整形状选 driver，再划分其外层 tile 循环"的必要性 | ✅ 论证成立 | 若先按行拆分再选路，会改变 M=1..8/M>=9 的路径选择、blocked 策略与 panel 复用，确实改变数值路径与性能形态 |

### 2.2 内部逻辑一致性

- **分片与确定性**：N 分片下每个输出元素仍由单一线程按原 K 顺序完成全部归约，未改变单元素累加顺序，因此"保持相同指令和累加顺序的路径做单线程与多线程逐位对照"（§5.1）在技术上可执行；分片边界沿用完整问题的 tile 分解正是保证逐位一致的前提。
- **失败收敛与 KV 发布**：先汇合再覆盖（W16 fallback）、KV 提交晚于最后一次 worker 访问、失败后等全部任务结束再返回——与 CPU 路线图 §5 风险表"同步 join/barrier 与失败收敛；完成前不发布 KV、不释放借用资源"逐条吻合。
- **同步产品边界**："调用线程按原顺序执行计划、单次 Generate 同步返回"与产品"同步、单请求"边界一致；worker 内再入并行接口走串行路径可避免嵌套等待与超额占核。
- **依赖方向**：Phase 2（Decode）零新增 scratch，可先于 CPU-07 实施；Phase 3（Prefill）依赖 CPU-07，提案以"先接通 CPU-07 再启用并行"表述——方向正确但边界未清单化（见 P0-2）。

### 2.3 与整体路线图匹配度

- 与 [CPU 路线图 §3.4 线程与资源规划](../operators/cpu-operator-evolution.md#34-线程与资源规划) 的划分策略（Decode 切 N、Prefill 切 M/N、禁 split-K、单 worker 独占输出 tile、worker 不嵌套）**完全一致**。
- 与 [GEMM 优化方案 §4.7 Threading 与 NUMA](../operators/gemm/cpu-gemm-optimization.md#47-threading-与-numa) 的八条原则一致，仅 NUMA/first-touch 一条未在本提案声明（见 P1-6）。
- CPU-08 的启动前提"对应串行 kernel 稳定；所需 CPU-07 资源链已具备"：FP32 identity 串行核心已由 CPU-03 闭环，Phase 2 满足启动条件；Phase 3 的前置 CPU-07 条目未列清单（见 P0-2）。
- 与当前优先事项（11 号 A16 AVX-512 补证、13 号生产验证）无目标冲突，但存在机器与人力共享，提案未说明并行工作流间的顺序/资源约束（见 P0-4）。
- 验收与启用条件（§5）对齐 [算子工作流 O0–O4 门禁](../guides/operator-development-workflow.md)：正确性对照、独立进程交错 A/B、噪声 floor 判据、启用范围按机器/phase/dtype/布局/形状记录，与工作流规范无冲突。

**结论**：技术论证与代码事实、路线图约束高度一致，无自相矛盾之处；主要缺口集中在契约冻结、规划信息与测量风险控制三个层面。

---

## 3. 风险点、模糊表述、遗漏环节

### P0（阻塞性，须在复审前修订）

| # | 问题 | 说明 |
|---|---|---|
| P0-1 | **并行接口契约完全未定义** | CPU 路线图 §2.3 将"接口签名、错误传播和配置"的冻结责任明确落在 CPU-08——即本提案。但提案仅用"借用窄的同步并行接口"一句话带过，缺失：任务描述结构、派发/等待原语、错误传播签名、嵌套检测机制（worker 内如何识别"再次进入"）、预建对象（任务描述/worker 状态/同步对象）的存储归属与生命周期、`KernelContext` 的扩展点与 Runtime→execution→kernel 的转接链。没有契约草案，Phase 2 无法开工，也无法评审失败收敛设计的可实现性 |
| P0-2 | **CPU-07 依赖边界未清单化** | "需要新增 scratch 的路径先接通 CPU-07 的 sizing、预算查询及绑定"是模糊表述：Phase 3 具体需要哪些 CPU-07 能力（worker scratch 的 per-thread 缩放语义？`WorkspaceRequirement` 的线程维度？溢出/上限检查归属？）未列清单，也未区分哪些可并行推进、哪些是硬前置。存在开工后发现阻塞的返工风险 |
| P0-3 | **T=1 vs T>1 对照的 DVFS/功耗混杂未处理** | 多线程负载改变功耗包络与睿频状态，独立进程交错 A/B 只能消除进程内频率漂移，不能消除"线程数不同导致频率基线不同"的系统偏移。提案未规定固定 P-state、记录每进程频率/功耗或同线程数空白对照。项目已实际经历过 WSL2 频率爬升造成虚假收益的教训（见项目记忆与 [GEMM 优化方案 §6.8.1](../operators/gemm/cpu-gemm-optimization.md#681-噪声-floor-是机器属性必须按机器按组量化)），该风险真实存在且直接影响"T=1 无可辨回退"门禁的判读 |
| P0-4 | **目标机器与阶段顺序约束缺位** | GEMM 证据制度是"一机器一文件、不可跨机引用"，但提案未点名任何采集机（单 socket 规格、核数、是否裸机）；也未说明与 CPU-07/11/13 并行工作流的先后与资源共享约束。不要求工期承诺，但机器清单与顺序约束是评审资源投入与周期规划的前提 |

### P1（重要，建议修订）

| # | 问题 | 说明 |
|---|---|---|
| P1-5 | **异构核的 per-worker ISA 保障只有原则无机制** | "选用的 ISA 对调用线程及各 worker 可运行的 CPU 均须有效"是正确原则，但当前 dispatch 基于进程级 CPU features，P/E 核或 big.LITTLE 下 worker 可能被调度到不支持该 ISA 的核上。提案未给出初选方案（per-CPU capability 枚举或 affinity 约束），也未声明首版单 socket 同构核假设。CPU 路线图 §5 风险表已列"ISA eligibility 与执行位置不一致"，本提案应承接 |
| P1-6 | **NUMA/first-touch 未声明范围与归属** | [GEMM 优化方案 §4.7](../operators/gemm/cpu-gemm-optimization.md#47-threading-与-numa) 明确 packed weight placement 与 first-touch/NUMA 归 runtime/model preparation 管理，但本提案只字未提。多 socket 机器上的采集可能不可解释，需至少声明首版单 socket 范围 |
| P1-7 | **同步对象与 worker 唤醒的设计要点未提** | 预建同步对象是一次性 barrier 还是 per-worker event？对象放哪个缓存行域以避免争用 ping-pong？Decode 每投影一次派发，唤醒延迟是首个实验的核心变量，但这些设计要点缺席 |
| P1-8 | **"建议 Runtime 统一配置"应为决策项** | 线程数配置归属是池所有权合同的一部分，用"建议"表述不构成冻结决策；且未说明 T 取值与 SMT 的关系策略（§5.2 要求"记录 SMT"属测量层，设计层未定） |
| P1-9 | **止损判据未前置** | §5.3 只有启用条件（配对收益超噪声），没有否定条件：若首轮实验证明派发+同步开销超过串行单 token 时长、或收益落在噪声内，该方向是否终止、结果如何归档，应前置定义，避免把"继续调参"变成默认动作 |

### P2（轻微）

| # | 问题 | 说明 |
|---|---|---|
| P2-10 | 静态连续分片尾负载不均未量化 | tile 数不被 T 整除时末段线程负载差异对 M=1 Decode 影响小，可接受，但建议记录负载差 |
| P2-11 | 短 Prefill 切 N 时 A 流量被 T 倍放大 | §2 已提及成本比较，但建议明确"只读 A 的流量放大"是与 dup-A 教训同源的否决风险之一 |
| P2-12 | 文档缺版本/日期/变更记录字段 | 同类文档（GEMM 优化方案）均带版本、日期、变更记录，便于修订追踪 |

---

## 4. 四维评估打分

| 维度 | 得分 | 依据 |
|---|---|---|
| 技术实现 | **8/10** | 三个关键论断与代码事实逐一核对属实；分片"先选完整 driver 再切外层 tile"的推理严密且直接决定逐位一致性与路径稳定性；失败收敛、回退顺序、零分配要求与路线图风险表逐条吻合；验收设计对齐 O0–O4 工作流。扣分：并行接口契约整体缺席（P0-1）、异构核保障无机制（P1-5） |
| 资源投入 | **6/10** | 范围克制是亮点：Phase 2 走权重直读零新增 scratch，正确绕开未完成的 CPU-07；线程池是 Runtime 级新基础设施（生命周期、失败回收、嵌套策略），TSAN 与基准矩阵工作量可观。扣分：无人力/工作量估算、无采集机配置、CPU-07 依赖未清单化（P0-2） |
| 业务价值 | **7/10** | 直击 Decode ms/token 这一产品核心指标；带宽受限 Decode 的多核并行流读是理论正确方向，且与"单请求、交互式本地推理"产品形态匹配（用空闲核换延迟）。扣分：收益未量化、派发开销吞噬收益的风险真实存在且为首要命题；多核利用率提升的业务假设未展开论证 |
| 周期规划 | **5/10** | 四阶段证据门禁定义清晰、可执行，部分替代里程碑功能。扣分：无目标机器清单、无阶段间顺序约束（尤其与 CPU-07 的硬前置关系）、无与 11/13 并行工作流的资源协调说明（P0-4）。尊重项目"不承诺工期"惯例，不因无日期而扣分，但规划信息缺位属实 |

**加权总分**：技术 0.4 / 资源 0.2 / 价值 0.25 / 周期 0.15 → 8×0.4 + 6×0.2 + 7×0.25 + 5×0.15 = **6.9/10**

---

## 5. 审核结论：修改后再审

**结论理由**：提案技术方向正确、与代码事实和路线图优先级（CPU-08 在串行核心稳定后接入）高度匹配，分片、回退与失败收敛设计严谨，Phase 2 的启动前提（FP32 identity 串行闭环）已满足；但作为 CPU-08 的冻结文档，并行接口契约（路线图明确要求 CPU-08 冻结的内容）完全缺席，目标机器与 CPU-07 依赖清单缺位，且存在 DVFS 测量混杂与异构核 ISA 两个未处理的实质性风险。上述问题均可通过文档修订解决，不涉及方向性错误，故不驳回。

### 修改建议清单

| # | 优先级 | 建议 | 对应问题 |
|---|---|---|---|
| M1 | 必须 | 新增"并行接口契约草案"一节：任务描述结构、派发/等待原语、错误传播签名、嵌套检测机制（如 thread-local 深度标记）、预建对象存储归属（任务描述/同步对象归 Runtime 或请求 arena 的明确划分）、`KernelContext` 扩展点与 Runtime→execution→kernel 转接链 | P0-1 |
| M2 | 必须 | 列出 CPU-07 前置清单：Phase 3 所需的具体 CPU-07 条目（worker scratch 的 per-thread 缩放语义、workspace planner 线程维度、溢出/上限检查归属），并区分硬前置与可并行项 | P0-2 |
| M3 | 必须 | 测量协议补充 DVFS 控制：固定 P-state 或记录每进程频率/功耗，T=1 与 T>1 在同一频率基线判读；引用项目内 WSL2 频率漂移教训 | P0-3 |
| M4 | 必须 | 补充目标机器清单（单 socket 规格、核数、裸机/WSL2 身份）与阶段顺序约束（含与 CPU-07/11/13 并行工作流的资源协调说明） | P0-4 |
| M5 | 必须 | 异构核与 ISA：给出 per-worker ISA 有效性保障的初选方案（per-CPU 枚举或 affinity 约束），或显式声明首版单 socket 同构核假设并注明失效条件 | P1-5 |
| S1 | 建议 | 声明首版单 socket 范围与 NUMA/first-touch 策略归属（对接 GEMM 优化方案 §4.7） | P1-6 |
| S2 | 建议 | 明确预建同步对象形态（barrier vs per-worker event）、缓存行域与复用策略 | P1-7 |
| S3 | 建议 | 将"建议 Runtime 统一配置"改为冻结决策，并补充 T 取值与 SMT 关系的设计说明 | P1-8 |
| S4 | 建议 | 前置止损判据：若派发+同步开销超过串行时长或收益低于噪声 floor 时的终止与归档条件 | P1-9 |
| S5 | 建议 | 补版本/日期/变更记录字段；记录静态分片尾负载差；将 A 流量放大列为短 Prefill 切 N 的显式否决风险 | P2-10/11/12 |

---

## 6. 证据追溯表

| 审核点 | 提案出处 | 核查依据 | 结论 |
|---|---|---|---|
| AVX2 driver M 分界 | §2 | `src/backend/cpu/kernels/gemm/gemm_a32_w32_avx2.cpp`（m∈[1,8] 与 m>=9 双路径）、`gemm_a32_w32_scalar.cpp`（kScalarSmallMMax=8） | 属实 |
| W16 部分写入后 reference 重算 | §3 | `src/backend/cpu/kernels/linear/linear_a32_optimized.cpp` L36-38/L45-47 | 属实 |
| 无生产 worker pool | §3 | `src/runtime/` 无线程实现；`include/aethermind/backend/kernel_context.h` 无并行字段 | 属实 |
| Decode 直读权重 | §3 | [GEMM 证据索引](../operators/gemm/README.md) | 属实 |
| 分片划分策略与路线图一致 | §2 | [CPU 路线图 §3.4](../operators/cpu-operator-evolution.md#34-线程与资源规划) | 一致 |
| CPU-08 启动前提 | §4 | [CPU 路线图 §4.1](../operators/cpu-operator-evolution.md)（CPU-08 依赖"对应串行 kernel 稳定；所需 CPU-07 资源链已具备"） | Phase 2 满足，Phase 3 边界未清单化 |
| 验收门禁对齐工作流 | §5 | [算子工作流 O0–O4](../guides/operator-development-workflow.md) | 一致 |
| 提案注册状态 | 状态行 | [路线图索引](README.md)（14 = CPU-08，Draft 候选） | 一致 |
