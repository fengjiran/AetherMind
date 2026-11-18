# CPU 算子覆盖与性能演进路线图

本文编排 CPU 算子演进的实施顺序、跨模块依赖与退出条件。当前活跃计划入口见 [roadmap](../roadmap.md)，产品范围见 [product](../product.md)，模块所有权遵循 [AGENTS.md §2.1](../../AGENTS.md)。

本文记录后续增量及各工作包的实施状态；已实现项仅在对应退出证据列明的机器、模型、精度与 workload 范围成立。初版现状表来自源码核对，未重新运行测试/性能；后续 CPU-02/03 的实际验证与测量已链接于 §4.1，不泛化为全部产品目标达成。后续工作包编号 `CPU-*` 与编译预设 O1/O2、算子工作流 O0–O4 分别独立。

## 1. 现状分析

### 1.1 当前生产路径与缺口

覆盖面需要分别回答四个问题：operator 语义是否成立、CPU kernel 是否存在、生产准备入口是否能选中、指定模型与平台是否完成验收。某一层成功不能替代下一层的执行证据。

| 能力 | 当前可核对事实 | 后续增量与依据 |
|---|---|---|
| FP32 Linear / MatMul | reference 保持独立计算路径；Linear/QKV/GateUp 已接入 scalar/AVX2 GEMM，MatMul 保持 reference；packed-B candidate 与 identity optimized 并存，identity 保持默认 | 接通并验证 optimized production path；见 [Linear entry](../../src/backend/cpu/kernels/linear/linear_entry.cpp)、[MatMul](../../src/backend/cpu/kernels/matmul/matmul_f32_reference.cpp)、[GEMM 状态](gemm/README.md) |
| W16/A32 | Linear、LM head、QKV、GateUp 已接入 AVX2/FMA W16 driver，F16 还需 F16C；O1/plain 与 O2/identity-packed 使用同一 recipe 下的 Decode/Prefill driver，reference 继续独立保留 | SIMD widening、Decode GEMV、Prefill blocking；见 [reference](../../src/backend/cpu/kernels/linear/linear_a32_w16_reference.cpp)、[Linear 证据](linear/benchmarks/desktop-54h5mmi-linear.md) |
| BF16 A16 | 高层 Load 已准入 activation/weight 的 BF16 同 dtype 组合（KV 组合约束在编译期校验）；Linear/RMSNorm 已有 AVX2 widening 和 AVX-512 实现，其他主链算子已有 reference | 扩大目标 shape/硬件证据，随后按 profile 优化；见 [Load](../../src/inference/executable_model.cpp)、[精度专题 §4.5](../improvement-plan/11-fp16-bf16-execution-support.md) |
| FP16 A16 | kernel 层已有实现，高层 Load 因 FP32 residual/Cast 路径未完成而明确返回 Unimplemented | 按精度专题 P4-B 补齐模型路径；不能由单 kernel 可调用推导模型支持 |
| Attention / KV | F32、A32/KV16、A16 reference 已存在；当前 Attention 不物化完整 score matrix，但两遍扫描重复计算 QK | SIMD、GQA KV 复用、Decode/Prefill 分组优化；见 [F32 reference](../../src/backend/cpu/kernels/attention/attention_a32_kv32_reference.cpp)、[KV16 reference](../../src/backend/cpu/kernels/attention/attention_a32_kv16_reference.cpp) |
| 融合与小算子 | FP32 RMSNorm 已有 AVX2；QKV/GateUp/AddRmsNorm 的部分默认路径仍是 reference；A16 编译主动关闭 QKV/GateUp/AddRmsNorm fusion | 检查 O2 fused-vs-unfused，避免优化单算子后让融合路径仍落在标量实现；见 [compiler](../../src/compiler/model_compiler.cpp)、[CPU kernels](../../src/backend/cpu/kernels/) |
| 通用执行覆盖 | Softmax、Reshape、Permute、Reorder、Cast 有语义定义，当前 CPU registry 缺少相应 standalone runtime kernel | 按真实 consumer 补齐；常量折叠成功不代表动态执行可用；见 [OpType](../../include/aethermind/operators/op_type.h)、[operator inference](../../src/operators/operator_inference.cpp) |
| 算子内并行 / 量化 | 当前 Runtime 未提供生产 worker pool；INT8/INT4 尚无完整加载、准备、kernel、Generate 链 | 分别建立 threading 和 quantization 纵向合同；见 [RuntimeOptions](../../include/aethermind/runtime/runtime_options.h)、[HF validator](../../src/model/formats/hf/hf_model_validator.cpp) |
| 资源复用 | workspace requirement 与 activation arena 主要按顺序累计，尚无完整 lifetime/liveness reuse | 在引入 tile/worker scratch 时纳入规划与预算；见 [workspace planner](../../src/runtime/workspace_planner.cpp)、[activation planning](../../src/execution/execution_bindings.cpp) |

精度状态以实际入口和专题证据为准。[product §1.4](../product.md#14-当前能力与验证状态) 分别记录 BF16 A16 的限定模型路径、FP16 A16 的高层拒绝与 native ISA 的待验证状态；限定 fixture 验收不能扩展为任意模型、平台或上下文的承诺。

### 1.2 优先级的机制依据

首批建议以 x86-64 AVX2/FMA/F16C 为优化基线。已有机器记录不足以支持跨机器性能排名，也不能证明 AVX-512/AMX 一定更快。

Decode 在权重读取主导、每个权重近似读取一次时，矩阵计算的 arithmetic intensity 约为 `2 / 每权重字节数`：F32 约 0.5 FLOP/B，W16 约 1 FLOP/B，另有 activation/output 与 metadata 流量。该模型说明权重带宽的重要性；当前 W16 reference 仍有明显 scalar widening 成本，必须先消除转换和归约瓶颈，再判断是否到达带宽上限。

Prefill 的多行输入可以复用权重，重点是 register/cache blocking 与数据复用。Attention 随 context 增长增加 KV 扫描，优化重点还包含 GQA 复用、score 归约与 exp 开销。小算子的优先级由 production profile 决定，不能按名称预先判定全部 memory-bound。

默认顺序为：矩阵计算 → Attention → 按需小算子/精度覆盖 → 算子内并行 → 量化优化 → 目标平台扩展。量化合同和 reference 可以提前开展；顺序不构成所有工作包必须串行的依赖。

### 1.3 覆盖台账与 workload

复用各算子 README 与既有测试/benchmark 元数据，按 `op × input/output/weight/state dtype × phase × plain/recipe × shape/stride × effective ISA` 关联实现与证据；记录实际 kernel 名称、模型/fixture 身份和所用线程配置。分别标记 semantic-only、CPU unsupported、reference、optimized candidate、production default，以及测试 pass/fail/not-run/硬件不适用，避免用一个“支持”掩盖不同成熟度。

| 类别 | 首批 workload | 验证目的 |
|---|---|---|
| Linear / GEMV / GEMM | 从实际 QKV、attention output、Gate/Up/Down、LM head 提取 N/K，覆盖 M=1/2/4/8/32/128/512 | Decode、small-M 与 Prefill；补非整齐维度、SIMD tail、合法 stride/unaligned access |
| Attention | MHA/GQA/MQA，head_dim=64/128，context=128/512/2048/4096 | query/KV group 复用、短长 context、causal/visibility 边界；扩大上下文须独立通过数值门禁 |
| 精度 | FP32、W16/A32、已有 BF16 A16 及其已准入 KV 组合 | 每个组合独立对 oracle，不将 FP16 kernel 存在写成高层可执行 |
| 缓存状态 | hot 与权重工作集超出目标机器 LLC 的 streaming 分开 | 区分 cache 驻留、转换开销与实际带宽；不能仅按固定工作集大小推定已超出 LLC |
| 执行边界 | prepared kernel → ExecutionPlan → 公开 Generate；准备与 packing 单列 | 验证实际 resolve、编排开销、资源成本和用户可见影响 |

优先覆盖真实角色和边界，不穷举上述维度的笛卡尔积。复用 [模型 benchmark](../../tests/benchmark/models/README.md)、[Linear benchmark](../../tests/benchmark/cpu_kernels/benchmark_cpu_linear.cpp)、[KV16 prepared benchmark](../../tests/benchmark/cpu_kernels/benchmark_cpu_kv16_kernels.cpp) 与 [KV16 ExecutionPlan benchmark](../../tests/benchmark/cpu_kernels/benchmark_cpu_kv16_execution.cpp)，按缺口增量扩展。

## 2. 目标架构

### 2.1 范围与不变量

保持 CPU-only、单进程、同步单请求、Token IDs I/O、静态 KV 和 Llama Dense 主线。本路线图不引入服务调度、连续批处理、Paged KV、GPU、MoE 或其他模型家族的默认交付承诺。

1. reference 与 optimized 分离，保留独立 oracle；优化的 FMA、归约顺序与近似行为单独验收。
2. kernel registry 仍在准备期 resolve；concrete shape/stride 特化尽量在 binding 冷路径完成。Attention 的动态 context/visibility 在固定 kernel 内处理。
3. optimized descriptor 覆盖其声明的合法合同；不适合快路径的布局使用兼容 fallback，不能指望 params build 失败后 registry 自动换核。
4. packed fallback 必须理解同一 recipe；不得把 B-panel bytes 当成 plain/identity 权重读取。
5. Decode 稳态零分配区间沿用产品合同；Generate 整体仍包含请求准备与 owning 输出。
6. 每次优化同时检查 Prefill/Decode、O1/O2、fused/unfused；预设编号、fusion 和 packing 本身不构成加速证据。
7. kernel 只借用窄执行资源；Runtime、packed artifacts、请求 arena 和 KV storage 的 owner 链保持明确。

### 2.2 职责、数据与接口

| 所属模块 | 后续职责 | 边界 |
|---|---|---|
| operators / shape_inference | dtype、shape、port、alias 前提与数值语义；按需补 Cast 等合同 | 不包含 ISA、tile 或线程数 |
| graph / compiler | backend-independent rewrite；pipeline composition；显式 dtype transition 与结构化 selector | graph 不依赖执行层；compiler 不查询 registry |
| backend/cpu | 共享 GEMM primitive、算子 adapter、Attention/小算子 kernel、recipe/layout 与 packer | GEMM 保持内部计算 primitive；不新增 semantic Gemm operator，不依赖 Graph IR |
| model | checkpoint scheme 解析、logical quantization metadata、raw weight 解析与准备请求 | 不决定 backend 物理 tile/layout；packing 经 Backend::PackWeights |
| execution | prepared 参数、binding 特化、workspace requirements、运行期窄资源转接与完整 KV transaction | public headers 不引入 compiler artifact；不拥有 Runtime worker pool |
| runtime / kv_cache | Runtime 拥有拟议的长驻 worker pool；kv_cache 保持静态物理状态与 session lifecycle；通用 workspace offset 规划归 runtime | runtime 不反向依赖 execution/compiler/model；并行任务不能修改 KV 发布权 |
| inference | 精度/量化准入、Load/Prepare 编排、资源预算与请求 owner 聚合 | 不复制 kernel、packing 或量化计算算法 |

共享计算数据继续从现有 `GemmArgs`、Linear 各精度 prepared args、`KernelSelector`、`PackingRecipe` 和 `WorkspaceRequirement` 演进，避免新建平行 planner/registry。`GemmArgs` 使用 byte 指针与元素 stride，精度由具体 kernel entry 定义；不同精度共享 driver 结构和调优方法，不强行合并 storage、转换与归约合同。

### 2.3 准备与执行数据流

```text
Load / 显式权重策略 → BuildModelGraph → semantic optimization → LoweredModelArtifact
  → Prepare：recipe 查询 / packing / plan-time resolve
  → 请求准备：concrete binding / driver 特化 / workspace 与 worker scratch 规划
  → Execute：冻结 kernel + 窄资源；动态 KV bounds 每次校验
  → 等待本次全部工作完成 → 完整 plan/layer 校验 → CommitAppend
```

若引入并行资源，拟议关系为 `Runtime 拥有 pool → context 借用窄并行接口 → kernel 同步完成`。跨层接口应放入允许下层消费的窄契约位置，具体 pool 实现不得暴露给 Graph IR 或 OperatorSchema；接口签名、错误传播和配置在 CPU-08 冻结。

## 3. 方案与备选

### 3.1 矩阵计算：完成共享 engine 的生产闭环

在 [GEMM 优化方案](gemm/cpu-gemm-optimization.md) 与 [packed weight 方案](gemm/cpu-gemm-packed-weight.md) 上继续实施，本文不另行冻结 tile 或阈值。

| 路径 | 推荐增量 | 关键取舍 |
|---|---|---|
| FP32 | 为已有 scalar/AVX2 driver 补齐真实 optimized descriptor、adapter 与生产测量；完成 B-panel candidate 的 promotion 证据 | 独立 reference 保持可审计；有内部 optimized primitive 不等于默认模型已使用 |
| W16/A32 | 先支持 plain/identity 的批量 widening；FP16 用 F16C，BF16 用向量 widening，保持 F32 input/accumulation | 先消除 scalar 转换；不能隐式把 A32 窄化到 BF16 来使用 native dot |
| A16 | 继续利用现有 Linear ISA 路径；改进多累加器、多输出与 blocking，保留各 ISA 数值门禁 | 当前 AVX2 路径主要按 row/output 做 dot，仍需验证真实大 M 的权重复用 |
| Decode / small-M | 沿 K SIMD，同时展开多个 N 输出；复用 input、控制 reduction 与寄存器压力 | 小 M 先用直接访问权重的 GEMV/skinny 路径，不强制套大 GEMM blocking |
| Prefill | 联合调优 MR/NR、MC/NC/KC；评估 A packing、B packing、边界块与 cache 工作集 | scratch 只有在收益覆盖 copy/规划成本时引入；栈上临时也要计入容量与 worker 成本 |
| QKV / GateUp | 复用同一计算 engine，逐步实现 combined-N traversal 和 input tile 复用；adapter 保留 output split/alias 校验 | 当前多次 reference projection 不能作为已取得融合复用收益的证据 |

FP32 生产闭环以兼容与正确性基线为目标：完成 CPU-03 的证据后按 profile 决定是否继续投入，不与 W16/A32 主线并行全速优化。

先用同一 recipe 服务 Prefill/Decode 的不同 driver。确需双布局时再扩展 artifact 规划、预算与 owner 去重，核对 tied Embedding/LM head；不默认复制全模型权重。

当前 W16 identity 可承载首批 SIMD。新 B-panel 只有在 profiling 显示布局收益后引入；改变 dtype、tile、panel 顺序或 padding 必须新增具体 `PackingLayout`，同步尺寸公式、packer、validator、artifact key 与 consumer。比较 compute-only、packing-inclusive 和公开请求三个边界，并记录 packing 摊销次数。

### 3.2 Attention：SIMD → Decode 复用 → Prefill blocking

1. **SIMD 基线**：向量化 QK dot、weighted-V、KV16 widening；先保留清晰的 softmax 结构，定位归约、转换、exp 与访存占比。
2. **Decode**：按 KV block 扫描，在 GQA group 内复用 K/V；比较两遍扫描、tile score scratch 和 online softmax，按 head_dim/context/group size 保留合适路径。
3. **Prefill**：query/key blocking，组织 QK、稳定 softmax、PV 的 tile 计算；在现有不物化完整 score matrix 的基础上减少重算、提高复用。

在线归约会改变求和顺序，需覆盖有限值误差、NaN/Inf、极端 score、尾块和 causal mask。KV16/A16 路径读取实际窄化后的 cache，Prefill 不得旁路使用更高精度 K/V。

scratch 由 prepared requirements 与请求 workspace 提供，不保留平行全量 F32 cache。KV 指针、read frontier、query positions 只在本次调用有效；保留 generation、visibility、dtype、stride、alias 和 range 校验。kernel 成功不能替代 whole-plan commit proof。具体状态域合同沿 [KV/Attention 专题](../improvement-plan/08-kv-cache-and-attention-capability-evolution.md) 与 [精度专题](../improvement-plan/11-fp16-bf16-execution-support.md)。

### 3.3 小算子、精度边界与通用执行覆盖

| 项目 | 推荐路径 | 触发条件 / 必须保留的语义 |
|---|---|---|
| AddRmsNorm | 补与独立 RMSNorm 对应的 SIMD 路径，检查 O2 fused-vs-unfused | 首批检查项；同时保留 normalized output 和 residual output、合法 alias 与 rounding placement |
| Add / SiluMul | contiguous SIMD、合理 unroll、tail 与 fallback | production profile 显示值得优化；近似 exp 单独验证，不全局开启 fast-math |
| Embedding | 复用已有 W16 scalar/F16C/AVX2 行级 widening，按 profile 优化 gather/copy 与输出布局 | 不重复建设已有转换路径；tied LM head 的 artifact/owner 策略一致，见 [Embedding](embedding/README.md) |
| RoPE | SIMD 旋转，评估三角值复用与按需缓存 | 冻结 position、算法参数、容量与预算；Dynamic NTK 的有效频率身份不能误用静态表 |
| KVCacheUpdate | 批量窄化或同型 copy，减少逐元素转换 | 保留 logical head_dim、padding、写入范围；A16 同型 copy 不重复窄化 |
| Cast / FP16 A16 | 复用既有 typed Cast 语义与转换合同，补 CPU kernel 和 residual 边界 | 按精度专题 P4-B 先冻结完整 rounding placement；LM head 直接输出 F32 |
| Reshape | 按 consumer 先做明确的 contiguous copy reference，再评估 view alias | copy 路径验证 numel/dtype/bytes；view 需要 activation alias、last-consumer 与 lifetime 证明 |
| Permute / Reorder | 语义视图与物理搬移分别处理 | 优先支持真实 consumer 需要的布局；不把 metadata permutation 自动视为零成本 executable view |
| Standalone Softmax | 有动态消费者时补 CPU kernel | Attention 内部 softmax 已有执行，不因缺 standalone kernel 而判整条 Attention 不可用 |
| Bias / QK norm 等 | 在现有 Linear、RMSNorm、shape ops 上按家族需求扩展 | 由 [模型家族专题](../improvement-plan/09-qwen-and-moe-architecture-support.md) 触发，不扩张当前 Llama 准入 |

A16 当前关闭的 QKV/GateUp/AddRmsNorm fusion 只有在对应 kernel、端口精度、rounding 与完整模型证据齐备后再开启。增加 fused operator 或 epilogue 时检查 consumer、残差流、layer boundary、权重绑定和中间舍入，不能单凭代数表达式改写数值合同。

### 3.4 线程与资源规划

首轮在单线程核心稳定后采用 Runtime-owned persistent pool。同步调用线程负责控制流，kernel 使用窄并行接口；禁止每个 kernel 临时建线程、隐式引入 OpenMP runtime 或进入嵌套并行区。

| 计算 | 首选划分 | 首轮约束 |
|---|---|---|
| Decode GEMV | N/output tiles | 单 worker 独占输出 tile，避免 false sharing |
| Prefill GEMM | M/N tiles | K 归约在 tile owner 内完成，先避免 split-K |
| Attention | query heads / query blocks | 少 head、长 context 的 split-KV 仅在后续有收益证据时增加 |
| Norm / elementwise | 多 row 或足够大的连续区间 | 小任务保留串行，阈值由调度/barrier 成本测得 |

Decode 并行按 latency 驱动评估：权重带宽在 worker 间共享，不按吞吐增益判收益；Prefill 并行才按吞吐评估。

pool 配置需冻结线程数、caller 是否参与、启动/销毁、失败传播、嵌套策略和 worker scratch。队列、任务描述、同步对象与临时存储提前准备；失败必须等待所有已派发任务结束，再返回或释放请求资源。只在全部 layer 写入完成后发布 KV。

scratch 容量按 tile 与 worker 数规划，纳入资源查询、alignment、overflow 和 cap。workspace lifetime reuse、activation liveness 可独立推进；完整复用优化不是所有 kernel 的前置条件，但新增 scratch 必须先有正确的规划与预算链。

先测单 socket scaling、带宽饱和与 barrier，再触发 affinity/first-touch/NUMA。worker 的有效 ISA 必须在其可运行 CPU 上成立；异构核和迁移策略不能仅凭调用线程探测结果推定安全。

### 3.5 量化：独立纵向合同，尽早开始 reference

按产品目标推进 INT8 per-channel、INT4 group-wise（group_size=128），推荐顺序：

```text
scheme/metadata 合同 → INT8 reference → W8/A32 优化
                   → INT4 reference → W4/A32 优化 → 代表性模型验收
```

合同设计与 reference 可在 CPU-01/02/03 期间开展，optimized 路径复用相关 GEMV/GEMM 基础；不以线程池或全部 ISA 完成为前置。

- 冻结 quantization axis、signedness、scale dtype/shape、zero-point policy、group、tail/padding、INT4 nibble order 及已支持 checkpoint scheme；未知或不一致格式在对应边界拒绝。
- 优先扩展已有 [QuantizationSpec](../../include/aethermind/graph/graph_types.h)，保持 logical quantization 与 backend packing 分离；审查 scheme 如何进入 selector、请求、artifact identity 和 consumer 校验，不能只增加 dtype。
- 贯通 checkpoint 解析 → raw metadata/weights → semantic graph → recipe/artifact → kernel → Generate；明确 Linear、LM head、Embedding、Norm 与 tied weights 的策略。
- 首批保持 F32 activation/accumulation，在 tile 内融合 unpack/dequantize 与计算。reference 使用独立、可审计的反量化公式；不以整模型拓宽为 F32 冒充低内存量化执行。
- VNNI/AMX INT8 的整数 dot 需要显式 activation quantization 合同；W8/A32 不自动满足其输入要求。任何 A8 路径独立验收量化与 requantization 边界。
- 同时验收实现误差、模型质量、准备峰值、packed 副本、KV/workspace、latency 和 RSS。7B × 4 bits 仅是逻辑权重估算，不能代替总内存验收。

量化 scheme 的完整专项设计在实施时形成，具体算子方案和证据仍归相应算子目录，不在本路线图猜测物理格式。

### 3.6 ISA 与平台扩展

保持 [CPU ISA 构建隔离](../../src/CMakeLists.txt) 与 runtime effective feature filtering，不给整个库使用 `-march=native`。每种 ISA 分别记录编译、dispatch/fallback、真实硬件数值与性能状态。

1. AVX2/FMA/F16C：首批生产优化与稳定 fallback。
2. AVX-512 BF16/FP16：验证已有 kernel，测量数值保护检查、转换、归约与降频等实际成本。现有 Linear/RMSNorm native descriptor 的 priority 高于 AVX2/reference（数值见 [backend 选核规则](../modules/backend.md#42-准备期解析)）；已编译且 effective features 满足时，native **已会被生产准备入口自动选中**。这是精度专题 §4.5 的既有策略，不代表真实硬件数值/性能已验收，状态文字也不会阻止选核。后续 priority 调整仍须生产证据，不能根据向量宽度推定收益；逐 ISA 状态与硬件补验项见 [精度专题 §6.4](../improvement-plan/11-fp16-bf16-execution-support.md#64-性能资源与发布状态)。
3. AMX：在目标硬件、OS 状态管理与允许的精度合同齐备后，优先研究大 M Prefill；Decode 独立评估。
4. AArch64：目标平台进入验收后建设 NEON，随后按硬件与 workload 决定 BF16/I8MM/SVE；保留 portable fallback。

不穷举所有 dtype × ISA × layout。每个组合都需要明确 consumer、机器、workload 与性能价值。

CPU-01 的 ISA 台账必须分别记录真实检测与合成 capabilities：机器/build、effective features、feature policy、完整 selector（含 phase、output dtype 与 plain/packed）、`ResolvedKernel::name` 和 exact recipe。合成 dispatch 只证明选择逻辑；真实执行另记数值测试、fallback、prepared/ExecutionPlan/模型级结果与性能状态。没有目标硬件时明确记为 not-run，不能把 AVX2 结果归给 AVX-512。部署方需要暂时排除 native 时，可用既有 `CpuFeaturePolicy.disabled_features` 削减对应特征；这不是新增准入门。

已有本机选核记录：[A16 Linear](linear/benchmarks/desktop-54h5mmi-linear.md#a16-实际选核身份与验证边界核对)、[A16 RMSNorm](rmsnorm/benchmarks/desktop-54h5mmi-rmsnorm.md#a16-实际选核身份与验证边界核对)。记录来自真实 capability detection；本机无 AVX-512，不能完成 native 硬件验收，也不据此关闭整个 CPU-01 工作包。跨算子完整 selector 台账（含 F32/W16/A32 矩阵、Attention/KV、小算子与语义-only 算子的真实 `ResolvedKernel::name` 与 exact recipe）见 [CPU 覆盖与生产基线台账 §3](cpu-coverage-baseline.md#3-生产选核身份台账真实检测)。

## 4. 实施步骤

### 4.1 工作包与退出条件

下表是待实施或待补证据的增量，已有基础见 §1。编号表示稳定引用，不代表严格串行顺序或工期承诺；状态列初始为“待实施”，启动或完成时更新并链接退出证据，不因“代码存在”直接标记完成。

| 工作包 | 状态 | 交付内容 / 主要位置 | 依赖 | 退出条件 |
|---|---|---|---|---|
| CPU-01 覆盖与基线 | 进行中（[覆盖台账](cpu-coverage-baseline.md)：参考机器/构建与代表模型 revision 已冻结、跨算子真实选核身份已记、同实现噪声与主路径耗时分布已取得限定观测；M0 基准达成，单算子 README 能力表与状态漂移修订收尾中） | 算子 README 能力表、prepared/ExecutionPlan/model benchmark、真实 kernel/recipe 身份；冻结参考机器规格与代表模型 revision；修订相关状态漂移 | 当前生产入口 | 主要组合可追溯，明确 unsupported/not-run；取得同实现噪声与主路径耗时分布 |
| CPU-02 W16/A32 矩阵计算 | 已实现（[本机限定验收](linear/benchmarks/desktop-54h5mmi-linear.md#cpu-02-w16a32-生产优化与-cpu-03-关联闭环)） | backend Linear SIMD widening、Decode/small-M、QKV/GateUp 复用，随后 Prefill blocking | CPU-01 与既有 W16 合同 | F16/BF16、plain/identity、tail/stride/fallback 数值通过；生产收益稳定，稳态零分配 |
| CPU-03 FP32 GEMM 生产闭环 | 已实现（[identity 生产闭环；B-panel 全局 promotion 拒绝](gemm/benchmarks/54h5mmi-gemm.md#cpu-03-fp32-gemm-生产闭环与布局-promotion-决策)） | 已有 scalar/AVX2 driver 接入；B-panel candidate 与 composite consumers 的 promotion 决策 | CPU-01；可与 CPU-02 独立实施 | registered/prepared、fused/unfused、模型路径与 packing 摊销均有证据；保留无收益组合的 fallback |
| CPU-04 Attention / KV | 待实施 | SIMD 基线 → Decode GQA/KV blocking → Prefill blocking/online softmax | CPU-01；引入 scratch 时补相应规划 | context/geometry/精度矩阵、事务/visibility 与数值门禁通过；短长 context 收益和资源成本分别可解释 |
| CPU-05 小算子与融合 | 待实施 | 首批 AddRmsNorm；其余按 profile 补 Add/SiluMul/RoPE/KVUpdate 等 | CPU-01；按优化后的模型重新 profile | 保留输出、alias、rounding 与 residual；fused/unfused 生产对照通过 |
| CPU-06 精度与执行覆盖 | 待实施 | Cast + FP32 residual 的 FP16 A16 模型路径；consumer 驱动的 shape/layout/Softmax kernel | CPU-01、精度专题 P4-B；家族 consumer 按需触发 | schema/Infer/graph/lowering/prepare/kernel/model 全链覆盖；合法路径可执行，unsupported 在准备期可诊断 |
| CPU-07 资源规划 | 待实施 | binding-dependent scratch、worker scratch、workspace lifetime、activation liveness 与预算 | CPU-01；按实际 scratch 需求拆分 | sizing/query 与执行一致，alignment/overflow/cap/OOM/alias 覆盖；复用不破坏已准备地址与 lifetime |
| CPU-08 算子内并行 | 待实施 | Runtime pool、窄接口、同步完成/失败收敛、tile 并行与阈值 | 对应串行 kernel 稳定；所需 CPU-07 资源链已具备 | 单线程不回退；scaling/barrier 可解释；TSAN、重复性、失败清理与零分配区间通过 |
| CPU-09 INT8 / INT4 | 待实施 | scheme 合同、reference、packing/identity、W8/W4 GEMV/GEMM 与模型执行 | 合同/reference 可从 CPU-01 启动；optimized 复用相关矩阵 primitive，不硬依赖 CPU-08 | 逐 scheme 数值/质量/资源/性能闭环，代表性模型验收，未知格式 fail closed |
| CPU-10 目标平台扩展 | 待实施 | 补齐现有 AVX-512 kernel 的真实硬件验证；需求触发的 AMX/AArch64 及其他 consumer | CPU-01（含参考基准）、对应精度/算子合同和目标硬件 | 分别取得编译、实际执行、数值、fallback、资源与性能证据，不扩大未验收组合 |

### 4.2 第一批实施队列

1. **CPU-01**：补齐矩阵/Attention 主路径台账，复用现有 runner 记录实际 descriptor/recipe、热点和噪声；冻结参考机器规格与代表模型 revision 作为公共基准，并记录 AVX-512 机器上实际 resolve 的 kernel 身份；不新建 capability runtime registry。进展见 [覆盖台账](cpu-coverage-baseline.md)：`desktop-54h5mmi` 机器/构建身份与 tiny fixture 代表模型 revision 已冻结，跨算子真实选核身份与覆盖缺口已记，4 进程同实现噪声 floor 与主路径耗时分布已取得限定观测（M0 基准达成）；native AVX-512 仍 not-run，单算子 README 能力表与状态漂移修订收尾中。
2. **CPU-02 首段（已完成，完整工作包证据见 §4.1）**：实现 W16/A32 plain/identity AVX2 widening 与 `M=1` GEMV，保持数值策略；随后覆盖 QKV/GateUp 与 small-M。
3. **CPU-03（已完成本机闭环）**：为已有 FP32 driver 完成生产 descriptor/adapter；补齐 B-panel 对照、tail、packing 与 Generate 证据，再决定 promotion。
4. **CPU-04 首段**：实现 Attention SIMD 候选，建立 F32/KV16/BF16 A16 的对应 baseline；随后推进 GQA 复用与 blocking。
5. **CPU-05 首段**：补齐 AddRmsNorm SIMD 并验证 O2 路径，随后根据重新 profiling 的结果决定小算子顺序。

首批首段组合的端到端里程碑（M1 已按本机限定范围达成，证据见 §4.1）：M0 = CPU-01 台账与参考基准冻结；M1 = CPU-02 首段 W16/A32 Decode 全链（prepared → ExecutionPlan → Generate）与 CPU-03 FP32 生产闭环；M2 = CPU-04 首段 Attention SIMD 与 CPU-05 首段 AddRmsNorm 后的重新 profile。其余工作包按触发条件推进，不占用上述关键路径。

CPU-09 的合同/reference 可提前推进；CPU-06 按 FP16 或新 consumer 需求启动。CPU-08 在相应串行核心稳定后接入，不等待所有小算子、精度与平台全部完成。没有 profile 支持时，不提前实现完整 autotuner、cost-based optimizer 或通用 Primitive IR。

### 4.3 文档与既有专题的分工

| 入口 | 负责内容 |
|---|---|
| 本路线图 | 跨算子优先级、模块依赖、交付工作包与退出条件 |
| [算子工作流](../guides/operator-development-workflow.md) | O0–O4、benchmark 方法、记录格式、promotion 与停止规则；本路线图不另建流程 |
| [GEMM 专项](gemm/cpu-gemm-optimization.md) | GEMV/GEMM engine、tile、packing、driver 与候选方案 |
| [08 KV/Attention](../improvement-plan/08-kv-cache-and-attention-capability-evolution.md) | KV owner/lifecycle、发布权、物理布局与 Attention 资源边界 |
| [11 精度](../improvement-plan/11-fp16-bf16-execution-support.md) | W/A/KV 组合、转换、rounding、校准和冻结预算 |

实施时同步代码、算子合同/状态与触发的模块文档。新的性能实验按算子记录，已接受的架构决策按现有模块/architecture 规则沉淀；不新建独立验证、日志或计划目录。

## 5. 风险与依赖

| 风险 | 必须检查的约束 | 收敛方式 |
|---|---|---|
| 精度变化被当作等价优化 | FMA、reduction、fast exp、A32→A16、fusion 舍入各有不同数值影响 | 冻结误差与 ranking 规则；同合同实现 A/B 与跨精度质量/资源比较分开 |
| O2 性能回退 | 融合可能选择 reference；identity copy、padding、双布局增加准备和内存成本 | 实际 kernel/recipe 可观测；fused/unfused 和 packing-inclusive 对照 |
| 描述符提升后合法布局失效 | shape/stride/tail 不满足快路径，但仍属于 descriptor 合同 | descriptor 内兼容 fallback；同 recipe 解释，拒绝布局混读 |
| scratch/复用破坏 lifetime | prepared params 可能持有地址；alias 与 last-consumer 影响可复用区间 | 冷路径完成完整证明，执行期间保持地址稳定；独立验证资源查询与执行一致 |
| 并行失败后继续访问资源 | worker 仍在写 output/KV，调用方已返回或释放 arena | 同步 join/barrier 与失败收敛；完成前不发布 KV、不释放借用资源 |
| ISA eligibility 与执行位置不一致 | 构建可编译不代表 OS 可用；worker 迁移、异构核与 policy 削减 | effective features、per-TU 隔离、worker domain/affinity 合同和真实硬件回归 |
| 性能证据失真 | tiny/cache-resident shape、WSL2 噪声、只测 microkernel 或最好一次 | 真实形状与 streaming；独立进程交错 A/B，保留全部 repetitions，按机器/workload 定阈值 |
| 量化内存目标过度推导 | scales/zeros、packed 副本、KV geometry、请求资源与 RSS 均有成本 | 冻结模型 revision/scheme/几何/单位，区分逻辑 payload、分配峰值与驻留内存 |

## 6. 验收标准

### 6.1 数值、语义与错误边界

- [ ] 每个新增组合都有独立 reference/oracle，覆盖真实 shape、tail、stride、alias、zero-size、overflow、NaN/Inf 与 fallback；reference 不复用被测 optimized 算法生成 expected。
- [ ] 同一精度策略的实现满足预先冻结的预算；跨精度质量对照使用对应校准资产。不同 ISA 不默认要求逐位相等，固定执行配置的重复性单独验证。
- [ ] teacher-forcing 固定 token/position 轨迹检查 logits/KV；公开 Generate 覆盖 token/EOS、边界、重复调用、错误恢复与 owner lifecycle。
- [ ] packed 布局、metadata、alignment 与 artifact identity 一致；缺失/非法组合在对应准备边界失败，动态 KV binding 错误在执行写入前按合同检查。
- [ ] fusion 保留 consumer、残差流、layer boundary、权重绑定与逐端口 rounding；相关 A16 fusion 不因 kernel 名称相似自动开启。

### 6.2 内存、并行与平台

- [ ] scratch sizing、binding、alignment、overflow、budget/OOM 和 planner reuse 测试通过；prepared pointer 与借用 lifetime 没有缩短。
- [ ] 指定 prepared Decode 区间的 malloc-family 检查通过；Runtime allocator 计数仅说明该 allocator 的调用次数，不能替代全进程分配检查。
- [ ] 受影响路径通过适用的 ASan/UBSan；并行/同步变更验证 TSAN。环境启动失败、skip、未运行与通过分别记录，LSan 只有实际启用并正常完成时才记为通过。
- [ ] worker 失败完整收敛，资源释放和 KV commit 均晚于所有任务完成；固定线程配置重复性通过。
- [ ] ISA 禁用与 baseline fallback 有实际执行证据；原生 ISA 的数值/性能结果来自对应目标硬件。

### 6.3 性能与 promotion

- [ ] 实验遵循算子工作流 O0–O4，既有专项门禁继续有效；以 registered/resolved + prepared production path 为主，包含 ExecutionPlan 与模型级影响，microkernel 仅作诊断。
- [ ] 同机同配置独立进程交错 A/B，记录 commit/dirty、模型与构建身份、effective features、线程/affinity、recipe、命令、全部 repetitions 和 raw retention 信息。
- [ ] Prefill/Decode、hot/streaming、准备/执行/公开请求分别报告；p90/p99 仅在样本数足够且方法适用时使用。上下文长度与实际生成 token 计数一致。
- [ ] 阈值基于同实现自比的机器/workload 噪声 floor；噪声内差异保留 Needs More Data，不设置全平台统一 5% 门禁。
- [ ] canonical workload 稳定受益，显著回退有经过验证的 shape/driver 策略；packing、workspace、worker scratch、RSS 与准备成本可接受，才调整生产 priority。
- [ ] 同合同优化与跨 W/A/KV 精度比较分别记录；当前模型报告校验器要求相同 workload 身份，不能绕过它把跨策略结果标为等价 A/B。
- [ ] 公共首 token、Decode、Generate、冷启动各自使用明确计时边界；现有 runner 的预读身份采集不能验收冷文件缓存启动。

当瓶颈转移、接近实测带宽/吞吐上限，或收益低于该 workload 噪声且端到端影响不明显时，停止当前调参并重新排序。7B 量化质量、总内存与吞吐等产品目标仍需代表性模型单独验收，tiny fixture 和局部加速不替代这些门禁。
