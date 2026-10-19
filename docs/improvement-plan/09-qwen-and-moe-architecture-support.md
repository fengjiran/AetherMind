# Qwen 与 MoE 模型架构支持方案

- **状态**: Draft
- **版本**: 1.1
- **日期**: 2026-10-19
- **产品边界**: [当前产品 PRD](../products/aethermind_prd.md)；本提案规划扩展模型支持范围，保留 CPU-only、单进程、单请求、同步、Token IDs I/O、静态连续 KV
- **架构边界**: [AGENTS.md](../../AGENTS.md) §2.1
- **关联计划**: [06 系统能力路线图](06-system-capability-evolution-roadmap.md)、[07 ExecutableModel 准备入口](07-executable-model-preparation.md)、[08 KV/Attention 演进](08-kv-cache-and-attention-capability-evolution.md)
- **算子工作流**: [算子开发与优化工作流](../guides/operator-development-workflow.md)
- **关联模块**: model / operators / graph / compiler / execution / backend / runtime / inference

本提案是待实施方案，不代表 Qwen、后续模型家族、MoE 或官方 BF16 checkpoint 已可执行。下文类型、算子与接口名称标为“拟新增”的均为设计名称；实施时须检查当前代码和测试，不能将本文当作已实现 API。

## 1. 现状分析

### 1.1 结论与实施主线

Qwen 是模型家族，MoE 是 FFN 结构；支持 MoE 不以先支持多个 Dense 家族为条件。推荐以 Qwen3 为首个纵向目标：

```text
公共配置、逻辑权重与绑定合同
        ├── Qwen3 Dense：Attention / 独立 head_dim / QK norm ──┐
        ├── MoE 语义与 reference kernel ──────────────────────┤
        └── 专家 bank / 多 packed bindings / workspace ──────┤
                                                           ↓
                                            Qwen3 MoE 端到端集成
                                                           ↓
                                            部署 dtype 与生产性能

其他 Dense 家族与第二个 MoE 家族：按需求并行或后续适配，不是上述主线的前置。
```

先验证 Qwen3 Dense 是建立数值基线、隔离 Attention 与 MoE 错误的交付选择；MoE 算子和准备链可以并行开发。部署不要求依次完成全部家族，也不要求先完成 INT8/INT4 才能证明 F32 架构正确性。

### 1.2 源码基线与证据边界

以下为 2026-10-19 的静态源码核对结果。文档编写没有修改实现，也没有重新构建或运行 C++ 测试；已有工作树改动不作为本提案实施证据。

| 当前事实 | 源码依据 | 必须补齐的能力 |
|---|---|---|
| `ParseModelArchitecture` 只接受 `llama`；Qwen/MoE 枚举预留，构图只分派 Llama | [识别](../../src/model/model_architecture.cpp)、[枚举](../../include/aethermind/model/model_architecture.h)、[dispatch](../../src/model/build_model_graph.cpp) | 明确家族识别、兼容性校验与 per-family builder |
| HF config 没有逐层 attention/FFN、专家或滑窗配置，未知字段跳过 | [配置](../../include/aethermind/model/formats/hf/hf_model_config.h)、[parser](../../src/model/formats/hf/hf_directory_reader.cpp) | 对行为字段显式解析和规范化 |
| validator 强制 `head_dim=hidden/heads`，Q/O 方阵，无条件拒绝 MoE 权重 | [validator](../../src/model/formats/hf/hf_model_validator.cpp) | 正确的投影 shape 与家族权重 schema |
| `ResolvedModelWeights` 只有 Dense FFN，无 bias、QK norm、专家集合 | [结构](../../include/aethermind/model/resolved_model_weights.h)、[resolver](../../src/model/formats/hf/hf_tensor_resolver.cpp) | typed 权重结构和完整角色解析 |
| validator 接受 `gelu/relu`，Llama builder 固定生成 `SiluMul` | [validator](../../src/model/formats/hf/hf_model_validator.cpp)、[builder](../../src/model/llama_dense_graph_builder.cpp) | 先消除 accepted config 与实际计算语义的不一致 |
| `QkvLinear` 明确拒绝 bias；CPU Linear 输入、权重、输出要求 F32 | [QkvLinear](../../src/operators/ops/qkv_linear_op.cpp)、[CPU Linear](../../src/backend/cpu/kernels/linear/linear_entry.cpp) | bias 合同与逐 dtype 执行闭环 |
| Reshape 已有语义推导，compiler/execution/backend 中尚无对应执行处理 | [Reshape inference](../../src/operators/ops/reshape_op.cpp) | 不能仅凭 Infer 成功宣称 QK norm 的 reshape 路径可执行 |
| packed lowered step 必须恰好一个 weight；请求构建在首个 weight 后停止 | [lowered validator](../../src/compiler/lowered_graph.cpp)、[request builder](../../src/compiler/weight_packing_request_builder.cpp) | 按语义端口关联多 packed artifacts |
| `ExecutionStep` 持单个 packed artifact；workspace planner 对各 step 求和 | [plan](../../include/aethermind/execution/execution_plan.h)、[planner](../../src/runtime/workspace_planner.cpp) | 多权重 ownership 与有界 scratch 复用 |

### 1.3 与产品及其他计划的关系

PRD 当前仍限定 Llama Dense，并排除 MoE。本提案记录用户要求的范围扩展；进入实现时同步修订 PRD 的模型矩阵和验收范围，交付时只把已通过相应证据门禁的组合标记为支持。本次文档化不把 PRD 的现状改写为已经交付。

06 保留通用 correctness、physical planning、量化和发布路线，本提案负责多架构/MoE 的具体依赖与集成；07 的 `PrepareExecutableModel` 仍是唯一生产准备入口；08 继续负责 KV 物理状态、提交权威、Attention 优化及资源预算。MoE 不要求 Paged KV、服务化、GPU 或专家并行。

## 2. 目标架构

### 2.1 模块职责

| 模块 | 本次扩展职责 | 保持的边界 |
|---|---|---|
| model | HF 配置规范化、家族识别、权重 schema/resolver、per-family 构图 | Loader 不构图、resolve kernel 或 prepack |
| operators | BiasAdd、RouterTopK、RoutedSwiGLU 的端口、typed params、dtype/shape/数值合同与 serde | 不表达 CPU 分桶、tile、workspace 或 packed layout |
| graph | semantic builder、binding 与合法 rewrite | 不依赖 backend，不把 grouped GEMM 当作前端模型语义 |
| compiler | 组合 semantic pipeline，lowering 和多 weight requests | 不查询 registry 或选择实际 kernel |
| execution | 按端口绑定 artifacts、specialize tensor、校验执行计划与资源 | public header 不 include compiler；不接管专家 packing 算法 |
| backend | reference/optimized kernels、bank layout、`Backend::PackWeights` | 不读取 HF 配置或 Graph IR |
| runtime | 通用 scratch 规划、资源聚合 | 不解析 MoE，也不依赖 graph/compiler/execution |
| inference | 编排 recipe、packing、plan、immutable bindings 与所有权 | 不另设第二套模型准备/执行入口 |
| kv_cache | 保持现有静态物理存储与 session view | 不承担路由或 MoE 状态；提交仍由完整 plan transaction 控制 |

### 2.2 核心不变量

1. `ParseModelArchitecture` 是家族判定唯一权威；`BuildModelGraph` 是 HF 到 semantic graph 的唯一 dispatch 入口。
2. `LoadedModel` 仍只拥有 config、resolved raw weights/backing storage；加载成功不等于可构图或可执行。
3. Qwen2 与 Qwen3 的规范化规则明确区分；不使用一个含大量隐式默认值的通用 Qwen 模板。
4. 每层 FFN 使用 Dense/MoE typed variant；每个专家与投影都有确定的逻辑身份。
5. Graph 权重 shape 表达真实数学维度；bank binding 不携带物理 packing 信息。
6. 路由是运行时数据，plan 拓扑保持固定；每个 token 的 K 个选中专家均参与计算。
7. Decode 稳态不做堆分配、registry lookup、shape inference、repack 或按路由重新编译。
8. MoE kernel 不自行提交 KV；完整 plan 成功后才发布 watermark，失败不发布。
9. 架构、checkpoint 表示、activation/weight/KV dtype、packing 与执行证据分别记录。

### 2.3 支持矩阵与范围

| 目标 | HF model_type | 主要差异 | 推荐位置 |
|---|---|---|---|
| Qwen3 Dense | `qwen3` | 独立 head_dim、head-wise Q/K RMSNorm、GQA、RoPE | 主线首个 Dense 基线 |
| Qwen3 MoE | `qwen3_moe` | Router/TopK、专家 bank、归一化开关、逐层 Dense/MoE | 主线首个 MoE 集成 |
| Qwen2/2.5 Dense | `qwen2` | Q/K/V bias、家族默认值、tied embeddings、可选滑窗 | 可并行适配 |
| Mixtral | `mixtral` | 权重命名、其路由及 attention 合同 | 通用 MoE 闭环后适配 |
| Qwen2 MoE | `qwen2_moe` | routed experts 外的 sigmoid-gated shared expert | shared expert 支路补齐后适配 |

主线 M0–M4 首版仅接受规范化后全层 full causal attention、SwiGLU FFN、已验证 RoPE 算法。启用滑窗的 checkpoint 明确拒绝；滑窗语义由后续独立工作包设计，不将 full attention 当作替代。后续家族需要的 activation、normalization 或 mask 扩展按 §3.8、§4.3 单独验收，不自动扩大主线支持合同。

Qwen3-Next 的 hybrid attention、其他 recurrent state、multimodal、encoder-decoder、LoRA 和分布式专家并行不纳入本提案。MoE routing 的 TopK 与生成阶段的 top-k sampling 是不同语义，本提案保持原有 greedy 生成边界。

### 2.4 Qwen 之外的家族优先级

选择依据是目标 CPU 部署、现有能力复用、可验证的架构增量和真实用户需求。以下是工程规划判断，不是模型能力排名、适配工时估算或性能结论；支持身份始终落实到具体 checkpoint 和部署 dtype。

| 家族/目标版本 | 建议优先级 | 支持价值 | 主要新增能力与准入条件 |
|---|---|---|---|
| Mistral，先选 Mistral-7B-v0.3 | 优先兼容扩展 | 验证 Llama 类 decoder 的跨家族复用 | 官方配置为 `model_type=mistral`、`sliding_window=null`；核对 resolver、RoPE、GQA 与输出；其他版本逐一判断有效窗口 |
| SmolLM3，先选官方 3B 文本模型 | 优先兼容扩展 | 检验逐层 position-encoding policy，并提供较小真实模型目标 | `model_type=smollm3`，full attention 中混合 RoPE/NoPE 层；增加 typed per-layer policy，保持 causal 与 KV 合同 |
| Phi，先选 Phi-3/Phi-4-mini 的明确版本 | 优先 CPU 产品候选 | 覆盖不同 checkpoint 表示与小模型部署 | Phi-4-mini 配置使用 `model_type=phi3`；适配 fused QKV/gate-up、partial RoPE、LongRoPE，按版本检查 attention 与缩放语义 |
| Gemma，先锁定一个文本变体 | 应规划，依赖单独语义工作包 | 扩大 Dense Transformer 的结构覆盖 | normalization 参数化、activation、attention scaling、local/global attention；选定版本逐项冻结，不承诺整个 Gemma 品牌或视觉路径 |
| Mixtral | 通用 MoE 闭环后候选 | 验证专家计算、packing 和绑定能否跨家族复用 | 命名、路由归一化与目标 attention 合同；完整专家权重和内存预算单独验收 |
| OLMoE，先选 OLMoE-1B-7B-0924-SFT | 第二个 MoE 验证候选 | 验证与 Qwen3 不同的路由及 norm 合同 | E=64、K=8、`norm_topk_prob=false`；Q/K RMSNorm 沿投影宽度，而非逐 head；检查目标配置的 QKV clipping 等字段 |

上述版本及语义的一手依据见 §7。Mistral 的其他版本、Ministral 文本骨干按实际 `model_type` 和结构另行核对，不能从一个 Mistral checkpoint 推导整个品牌已兼容。

实施上保留三条可独立推进的分支：低增量兼容线（Llama 验收、Mistral、SmolLM3）、CPU 产品扩展线（Phi、语义门禁具备后的 Gemma）、MoE 通用性线（Qwen3 MoE 后从 Mixtral/OLMoE 中先选一个）。分支之间只按实际能力依赖排序，不要求全部 Dense 家族完成后才实现 MoE。

### 2.5 品牌名称与实际架构的区别

| 兼容性验收目标 | 官方配置中的实际底座 | 接入方式 |
|---|---|---|
| SmolLM2-1.7B-Instruct | `model_type=llama`、`LlamaForCausalLM` | 纳入 Llama checkpoint/形状/dtype 验收矩阵；不新建 SmolLM2 架构枚举 |
| DeepSeek-R1-Distill-Llama-8B | `model_type=llama`、`LlamaForCausalLM` | 验证其配置、RoPE、权重与生成合同，复用 Llama adapter |
| DeepSeek-R1-Distill-Qwen-7B | `model_type=qwen2`、`Qwen2ForCausalLM` | 纳入 Qwen2 适配与验收；不新建 DeepSeek 蒸馏 builder |

这些目标扩大 checkpoint 兼容性，并不证明新的底座架构已经实现。原生 DeepSeek-V3 的 MLA、DeepSeekMoE 和缓存表示需要独立设计，不能由蒸馏 checkpoint 的通过结果推导；GLM 等其他家族也应由具体模型和部署机器需求驱动立项，本次仅记录候选，不冻结其架构方案。

## 3. 方案与备选

### 3.1 配置规范化与家族 dispatch

在 `model` 内拟新增纯配置规范化 helper，返回 typed decoder/layer 描述，供 validator 和 family builder 复用；HF 原始字段仍留在 `HfModelConfig`。优先保持现有 `BuildModelGraph(config, weights)` 入口，不把新描述泄漏到 backend。

规范化描述至少确定：`hidden_size`、Q/KV heads、有效 head_dim、各投影 bias 策略、QK norm、每层 attention/FFN 类型、RoPE、专家数 E、top-k K、专家宽度 I、`norm_topk_prob`。同一规则供配置校验、权重 shape 校验和构图使用，避免三处分别推导。

- 扩展 `ModelArchitecture`，明确区分 Qwen2/Qwen3 Dense 和目标 MoE 家族；已有枚举值不随意重排。
- `model_type` 识别家族；`architectures` 存在时校验其与受支持任务/家族相容，冲突给出字段级错误。复用同一识别结果，尽早在读取大权重前拒绝不支持家族。
- 对可选数值区分缺失、null 和显式非法值；不把显式 0 当成默认值。
- 解析 `use_sliding_window`、`sliding_window`、`max_window_layers`、`layer_types` 等相关字段，按具体家族规则规范化；Qwen2.5 中非空窗口字段与关闭开关可以同时存在，不能只检查原始窗口值。
- Qwen2 的 Q/K/V bias 按其架构规定，不能因 config 未提供 `attention_bias` 就推断为无 bias。
- MoE 解析 `num_experts`、`num_experts_per_tok`、`moe_intermediate_size`、`norm_topk_prob`、`decoder_sparse_step`、`mlp_only_layers`；dense 与 expert intermediate size 独立校验，专家宽度允许小于 hidden。
- 已知未支持的行为或量化格式明确拒绝；无关 metadata 继续跳过。eval 下禁用 dropout，训练 aux loss/jitter 元数据不直接等同于推理能力缺口。
- 先把现有固定 SwiGLU builder 的 accepted activation 收敛到 `silu`；其他激活只有在完成对应图与执行实现后再开放。

当前家族数量使用显式 switch 和 per-family adapter 足够，不引入插件注册系统。共享 block helpers 只复用已相同的计算结构，不通过猜测 tensor 名称判定架构。

### 3.2 逻辑权重、binding 与 ownership

拟将 `DecoderLayerRawWeights` 的单个 Dense MLP 扩展为下列结构，具体命名在实施时确定：

```text
DecoderLayerRawWeights
  norm
  attention
    q/k/v/o weights
    optional q/k/v/o biases
    optional q_norm/k_norm weights
  ffn: variant<DenseFFNWeights, MoEFFNWeights>

MoEFFNWeights
  router weight
  experts[expert_id]: gate/up/down
  optional shared_expert
  optional shared_expert_gate
```

validator/resolver 按实际层类型检查所需 tensor，不能继续要求 MoE 层具备 Dense gate/up/down。检查 expert ID 连续性、缺失/重复/多余权重、投影方向、dtype 和字节数；bias/norm 等角色同步贯通 `ParameterSlot` 映射、诊断、绑定校验与 resolver。

专家 bank 的拟新增 binding 表达“当前 decoder layer 的某个专家投影集合”，expert 按 ID 排序。整个 bank 的 E 和维度以 `TensorSpec` 为权威；如需单专家绑定，在 typed binding 中显式记录 expert index，不通过 debug name 解析身份。

对独立二维专家 tensor 直接保留 `RawWeightView`；对连续三维来源按 expert 切成二维 slices。每个 slice 保留原 `RawStorage` 的 shared ownership，并严格验证 offset/bytes 溢出与范围。

对 `[E, 2I, H]` fused gate/up 来源，逐 expert 切出 `[I,H]` gate 和 up。跨 expert 的 gate 区域不是一个连续 `[E,I,H]` raw tensor；当前 `RawWeightView` 没有 stride，禁止伪造连续视图。规范化结果以 slices 集合表示，backend 可直接从这些 components 打包，无需先 stack 一份完整权重。

### 3.3 Dense 路径：独立 head_dim、QK norm 与 bias

有效 head_dim 为显式合法值，缺失时才按家族规则推导。令 H 为 hidden、Nq/Nkv 为 heads、D 为 head_dim，权重形状为：

```text
Q:   [Nq * D, H]       K/V: [Nkv * D, H]
O:   [H, Nq * D]       Qwen3 Q/K norm scales: [D]
```

当 D 显式提供时，不再以 `H % Nq == 0` 或 `Nq*D == H` 作为通用条件；仍验证 heads、D 为正、GQA 可整除及乘积无溢出。官方 Qwen3-0.6B 的 H=1024、Nq=16、D=128，是必须覆盖的非方阵案例。

Qwen3 的 Q/K norm 使用现有 RMSNorm 语义，图中明确表示：

```text
Q/K Linear → Reshape [T, N, D] → RMSNorm(last dim D) → Reshape [T, N*D] → RoPE
```

Q、K 分别使用自己的 scale，norm 在 RoPE 前计算。首版补齐 contiguous F32 Reshape copy reference 执行路径，验证 numel、dtype、bytes、连续性及独立输出，使用已规划 activation buffer。此方案有额外读写成本，但不引入尚未定义的 activation alias 生命周期。

Qwen2/2.5 的 baseline 拟新增 `BiasAdd(x[...,D], bias[D])`，端口为 activation + weight，输出与 x 同 shape；图先用 `Linear → BiasAdd`。现有 Add 虽有广播推导，其 rhs 角色是 activation，不能直接塞入 bias WeightValue，也不应把 checkpoint 权重伪装成 inline constant。

备选为带 optional bias 的 Linear/QkvLinear schema，或 HeadRmsNorm 复合算子。首版选独立 BiasAdd 与已有 RMSNorm/Reshape，减少对现有语义 ABI 的改写。后续依据 production-path 数据决定 bias epilogue、QK norm fusion 或 view alias；任何 rewrite 必须保留 bias、norm 顺序、consumer 和 residual 语义。

### 3.4 MoE 语义与算子合同

拟新增 `RouterTopK` 与 `RoutedSwiGLU`，router 投影复用普通 Linear。T 表示单请求当前调用的 token 数，E 为专家数，K 为每 token 专家数，I 为 expert intermediate size。

| 算子 | 输入端口顺序 | 输出端口顺序 | 关键 params |
|---|---|---|---|
| RouterTopK | logits `[T,E]` | expert_ids `[T,K]`；routing_weights `[T,K]` | K、norm_topk_prob |
| RoutedSwiGLU | x `[T,H]`；expert_ids；routing_weights；gate `[E,I,H]`；up `[E,I,H]`；down `[E,H,I]` | y `[T,H]` | SwiGLU 语义；维度从 TensorSpec 校验 |

首版 logits、routing_weights、activation 和权重为 F32，expert_ids 为 Int64。RouterTopK 在全部 E 个 logits 上进行数值稳定的 F32 softmax，选取 K 个不同专家；若 `norm_topk_prob=true`，再将选中概率除以其和。若为 false，禁止仅对选中的 logits 做 softmax，否则改变分母。

```text
Expert_e(x) = Wdown_e * (SiLU(Wgate_e * x) ⊙ (Wup_e * x))
y[t] = sum_j routing_weights[t,j] * Expert_expert_ids[t,j](x[t])
```

冻结合同：`1 <= K <= E`；平分时以 expert ID 升序确定次序；RouterTopK 拒绝非有限 logits；RoutedSwiGLU 拒绝越界/重复 expert IDs 或非有限 routing weights。错误不得导致越界写入或 KV watermark 发布。各操作允许 T=0 的空输出，生产请求仍遵守 Session 的非空 prompt 合同。首版输出与输入不 alias，输出累加使用 F32。

正常路由满足每 token K 次执行、无 token drop。HF 对拍使用无平分 case；平分及非有限值另测本地合同。`norm_topk_prob=false` 的概率和不要求为 1；true 时要求误差范围内为 1。

首个目标 Qwen3-30B-A3B 的官方配置为 E=128、K=8、expert I=768、`norm_topk_prob=true`；微型 fixture 可以缩小 E/H/I，但不能只测 K=1/2 就宣称覆盖该架构。

`decoder_sparse_step` 与 `mlp_only_layers` 在前端规范化为逐层 Dense/MoE variant。Qwen2 MoE 的 shared expert 后续表示为普通 FFN 与 `Sigmoid(shared_gate(x))` 的乘法/加法支路，补齐 Sigmoid 执行证据；不把该支路默认加到 Qwen3 MoE。

RoutedSwiGLU 是上述加权和的复合数学语义，分桶、分块、grouped GEMM 属 backend 实现。相比首版就增加动态 Gather/Scatter、变长张量和控制流，这保留了静态输出 shape 与固定 plan。整体 MoEBlock 也是可行备选，但会耦合 router 与专家执行；首版拆为两个算子，便于独立数值对照和复用。

### 3.5 专家 bank packing 与多 packed bindings

保留 gate/up/down 三个真实 WeightValue，logical shape 分别为 `[E,I,H]`、`[E,I,H]`、`[E,H,I]`。新增专家 bank request 语义，由 `Backend::PackWeights` 从顺序明确的 raw components 直接构造 bank；参考布局先使用明确的 F32 expert-major 行主序格式，优化布局再由 backend 定义 tile/panel/padding。

每个非空 `PackingLayout` 必须对应确定、不可变的字节格式；维度、expert offset、alignment 与大小公式写入 CPU 布局合同。不能复用旧二维 layout ID 表示新 bank，也不能把三维 bank 放宽为普通 Linear 的权重。

当前单 packed weight 限制需作为一个完整工作包解除：

```text
LoweredGraph 结构验证
  → BuildWeightPackingRequests 枚举全部 weight 端口
  → inference 逐端口解析 recipe
  → backend descriptor / ResolvedKernel 的端口 recipe 合同
  → ExecutionPlanBuilder / ExecutionStep 的按端口 artifact bindings
  → KernelParamsBuildContext / KernelContext / prepared params
```

关联键使用语义 input-port index，避免端口裁剪后的序号歧义；artifact identity 保留 source/value/binding/selector/exact recipe。每个 bank 独立验证 dtype、shape 和 recipe；plan 对 artifacts 持有稳定只读所有权，kernel 只借用其视图。prepared metadata 中的 span/pointer 必须借用稳定 owner，不得指向构建阶段临时 vector。

首版多个 bank 使用同一受支持 dtype，并验证该限制，避免单个 KernelSelector 的 weight_dtype 与实际端口冲突。逐端口不同 dtype/recipe eligibility 只有在显式合同具备后再开放。既有单权重算子作为单元素 binding 集合继续覆盖，不另设平行模型准备入口。

不采用把所有专家、全部投影拼成伪二维 weight 的规避方案；不以 runtime 逐专家调用 `PrepareKernel`/PackWeights 代替模型准备。后续 gate/up 物理组合仍须有明确复合语义与对应布局，不得绕过 logical shape 校验。

### 3.6 静态执行、workspace 与资源预算

plan 步骤固定，expert_ids/counts/offsets 是执行数据。准备阶段计算与校验 T 上界、E、K 及全部字节乘积，运行时不因数据分布重新 specialize 图。

- **Decode**：T=1，直接读取 K 个选中专家并顺序加权累加；不扫描所有专家矩阵，不构造 one-hot mask，不动态打包。
- **Prefill**：构建总长 T*K 的索引桶与 E 组 counts/offsets，逐专家按固定块大小 C gather、FFN、scatter-add；空专家跳过。所有 token 都路由到同一组 K 个专家也是合法 case，必须分块完成。
- **scratch**：包括 `O(E+T*K)` 索引、router softmax 临时空间及 `O(C*(H+I))` 分块 activation；不隐式生成 `[E,T,H]` padding 或 `[T,K,H]` 全量专家输出。Router logits `[T,E]` 与其图内 activation 另计，不能从总内存中遗漏。
- **执行数据与参数分离**：prepared params 缓存 geometry 和有效指针，counts/路由内容在调用时读取，不能在 cold path 对尚未计算的 logits 预先路由。
- **容量与失败**：以所有 T*K 路由为精确容量，无 capacity-factor/first-K 丢弃策略；内存不足在准备阶段可恢复失败，运行时非法输入报告错误，不截断结果。

当前 planner 对所有 step scratch 求和。推荐最小扩展为：仅 `lifetime=kPerOperator && reusable=true` 的同步串行 step 共用一块按最大 bytes/最大 alignment 规划的区域；其余 scope 独立。合同要求 kernel 返回前完成全部访问，禁止保留 scratch 指针。没有该改动前必须按逐 step 求和计预算，不能宣称已跨层复用；通用 activation liveness 仍属 06 的独立工作。

资源预算覆盖 raw mmap 驻留/packed artifacts 重叠、所有 activation、共享与非共享 scratch、静态 KV、元数据和准备期临时内存。A3B 是激活参数规模，不能作为完整 checkpoint 或运行内存规模。先用 tiny MoE 证明正确性，再按真实机器预算决定完整模型验收；不把逐 token 按需 packing 作为内存不足的补救。

### 3.7 dtype 与量化的独立纵向闭环

首版采用 F32 activation/weights/KV 和固定版本 HF F32 oracle。小型真实 Qwen 可以使用明确记录来源的 F32 导出 checkpoint；这只能标为相应 F32 组合支持，不能宣称官方 BF16 文件已可直接运行，也不能只修改 config dtype 而不转换数据。

BF16/FP16 权重、activation、KV 分别列入 capability matrix。若引入转换，明确其时机、所有权、计算精度与峰值内存。INT8/INT4 选择具体 scheme 后完整贯通 metadata、scale/zero-point/group size、bank slices、binding、packing、kernel 与数值验收；不把 `allow_quantized_tensors` 当作执行支持。

### 3.8 后续家族的具体适配边界

#### 3.8.1 Mistral 与 SmolLM3：复用通用 decoder

Mistral-7B-v0.3 作为首个无滑窗目标，仍需完成家族识别、配置默认值、tensor schema 与完整推理对照。结构接近 Llama 仅说明可以复用 block helpers，不构成直接把 `mistral` 字符串映射为 Llama 且跳过家族校验的理由。

SmolLM3 在前端将 `no_rope_layers`/interval 等字段规范化为逐层 RoPE/NoPE policy，依照所固定官方实现解释数组值，避免根据字段名称猜测布尔方向。NoPE 层跳过 Q/K 旋转，仍执行 causal attention 和 KV append，缓存该层未旋转的 K；绝对位置、layer 顺序和完整 plan 提交规则保持一致。图中明确保留这种层差异，不让 kernel 读取品牌名称决定是否旋转。

#### 3.8.2 Phi：checkpoint 融合表示与 RoPE

按明确 Phi 版本适配 fused QKV 和 gate-up：model resolver 将来源 tensor 解析成具有正确 Q/K/V、Gate/Up 边界和顺序的逻辑 slices，保留 backing owner；这些来源布局不等于 backend packed layout。后续优化图仍可按现有语义复合 binding 重新组合，由 backend 完成物理打包，不为 HF 融合文件建立第二套 Linear 执行体系。

partial RoPE 验证 rotary dimension 与 head_dim 的关系、配对方式和未旋转维度；LongRoPE 复用已存在的算法基础，但单独对照目标版本的 short/long factors、原始上下文长度、位置与长度选择规则。不能只因解析了 LongRoPE 枚举就开放长上下文支持；通过配置、位置边界、Prefill/Decode 数值和 KV 对照后再标记对应组合。

#### 3.8.3 Gemma：文本路径与局部注意力

先锁定一个文本变体及 revision，逐项确定 norm 公式及 scale 表示、activation/GELU 变体、embedding/logit 缩放、attention scaling、QK norm 与逐层 local/global mask。只有目标版本实际启用的语义进入其图；不把 Gemma 各代差异折叠成一个品牌开关，也不以 Qwen SwiGLU 模板替代不同激活。

滑窗正确性首版可保留完整静态连续 KV，在 operators 中定义 typed local causal attention 合同，由 kernel 按每个 query 的绝对位置应用读取范围/mask。若规范化窗口长度 W 包含当前 token，则 query 位置 p 的可见 key 范围为 `[max(0,p-W+1), p]`；从原始模型字段到 W 的转换必须按固定 oracle 确认。Prefill 的每个 query 分别限窗，Decode 依据当前绝对位置限窗，global 层继续使用全部已提交历史及当前合法 append。

该方案不要求 ring cache、Paged KV 或淘汰历史 token，也不减少静态 KV 预留字节；窗口语义与物理存储优化分别验收。position 不能按窗口取模后传给 RoPE。相关 KV owner、水位发布与资源合同继续由 08 负责，本工作包只扩展算子及读取语义。启用这些能力前，主线的 unsupported-window 拒绝行为保持有效。

#### 3.8.4 Mixtral/OLMoE：第二个 MoE 家族

Qwen3 MoE 闭环后先从 Mixtral、OLMoE 中选一个，检验 RouterTopK、RoutedSwiGLU、bank packing、多 packed bindings 和静态 workspace 的跨家族复用，避免同时新增多个未经验证的 adapter。Mixtral 优先验证其命名、选中概率归一化与 attention 合同；OLMoE 优先验证 `norm_topk_prob=false` 和完整投影宽度的 Q/K norm。

OLMoE 的 Q scale 对应 Q 投影宽度，K scale 对应 KV 投影宽度；前端应在 reshape 拆 head 前沿投影最后一维做 RMSNorm，不能复用 Qwen3 的逐 head `[D]` scale 路径。归一化轴与 scale shape 由 typed 家族合同确定，底层可复用 RMSNorm。目标配置中非空 `clip_qkv` 等额外行为必须实现并验证或明确拒绝。

第二个 MoE 家族的通过证据应同时包含至少一个真实 checkpoint 与可再生成 tiny fixture；激活参数少不代表完整专家权重少，不能以“1B activated”替代 OLMoE 全部专家的内存预算。

## 4. 实施步骤

### 4.1 工作包与依赖

| 工作包 | 范围与主要落点 | 前置 | 退出证据 | 状态 |
|---|---|---|---|---|
| M0 公共前端合同 | model architecture/config/validator/resolver、typed layer/FFN 与 binding 草图；修正 activation、缺失/非法值、有效滑窗判断 | 无 | 家族冲突和拒绝测试；Llama 原支持范围回归；PRD 拟支持矩阵明确 | 待实施 |
| M1 Qwen3 Dense 基线 | per-family builder、独立 head_dim、Q/K norm roles、Reshape reference、图/融合检查 | M0 | 真实 CpuBackend 经生产入口完成 tiny Qwen3 Prefill→多步 Decode 的逐层/logits/KV 对照 | 待实施 |
| M2 MoE semantic/reference | RouterTopK、RoutedSwiGLU 的 schema/params/Infer/serde 与 F32 reference | M0 合同冻结 | O0/O1；prepared operator 对照；K=1/2/8 与不均匀路由覆盖 | 待实施，可与 M1/M3 并行 |
| M3 专家准备与资源 | bank binding/slices、request/backend packing、多 packed bindings、scratch 预算/窄复用 | M0 合同冻结 | 多端口 artifacts 正反向校验、ownership、布局、溢出/OOM、单权重回归 | 待实施，可与 M1/M2 并行 |
| M4 Qwen3 MoE 集成 | MoE family builder、Dense/MoE 混合层、生产准备与 Session 集成 | M1/M2/M3 | 完整 Load→Compile→Prepare→Prefill→多步 Decode；router/experts/hidden/logits/KV 数值及零分配证据 | 待实施 |
| M5 部署与性能 | 按需求补 BF16/FP16/量化；优化 expert GEMV/GEMM、packing、路由与内存 | 对应 F32 集成基线 | O2–O4、真实模型资源与端到端基准；逐组合标记支持 | 待实施 |
| D2 Qwen2/2.5 Dense | QKV bias、BiasAdd、家族默认值、builder 与 fixtures | M0；不依赖 MoE | 自身生产链端到端对照；带 bias 的优化图不丢语义 | 可并行选做 |
| X 后续家族 | §4.3 的其他 Dense 家族、第二个 MoE 家族及 shared expert 分支 | 仅依赖所复用的具体能力 | 每个家族独立 fixture、实际支持矩阵和端到端对照 | 按需求进入 |

M1 是推荐集成顺序，不阻塞 M2/M3 的独立实现。M0–M4 不以完成 D2、Mixtral、滑窗、通用动态 shape 或量化为门禁。M5 不是先做性能再验证正确性；每项 dtype/优化都从已验证基线开始。

### 4.2 改动落点与审查单元

1. `model`：修改 HF config/parser/validator、architecture 与现有 weights/resolver；拟新增 Qwen3 Dense/MoE builder 和必要的共享 normalization helper。
2. `operators/graph`：新增 RouterTopK/RoutedSwiGLU，按需新增 BiasAdd；同步 schema、typed OpParams、Infer、serde、graph builder、binding 及相关 validator。
3. `compiler/execution/backend/inference`：将多 packed bindings 作为一条完整合同变更审查，覆盖所有 producer/consumer，保留单权重路径证据。
4. `runtime/backend`：scratch 大小与执行 lifetime 合同先冻结，再实现窄复用；不借机引入线程池或通用异步 allocator。
5. `tests/tools/docs`：扩展现有 HF fixture 与生产入口测试，新增固定版本 oracle 生成工具；算子设计、状态与机器级证据按 O0–O4 工作流归档，更新模型能力矩阵。

工作包可按合同、reference、integration 拆成可构建的小提交，不能在仍无法绑定/执行时先开放 loader 的支持开关。新增/扩展算子遵守 O0–O4；优化记录不堆积在本系统提案中。

### 4.3 后续家族工作包

以下分支不构成 M0–M4 前置，也不扩大其首版 activation/mask 合同。除 fixture 选型外，进入实现前冻结具体 checkpoint/revision、部署 dtype、目标机器预算及数值 oracle。

| 工作包 | 交付范围 | 实际依赖 | 家族退出证据 | 状态 |
|---|---|---|---|---|
| C1 底座兼容性 | Llama 实际 checkpoint、SmolLM2、R1-Distill-Llama；Qwen2 具备后加入 Distill-Qwen | 相应底座、RoPE 与 dtype 的生产链 | 不新增品牌枚举；实际权重、tied/untied head、logits/KV/生成验收 | 按需求进入 |
| D3 Mistral | v0.3 无滑窗目标的 adapter/validator/builder | M0 与通用 Dense 执行基线，不依赖 MoE | 规范化窗口确为关闭、真实配置与完整推理对照 | 可并行选做 |
| D4 SmolLM3 | 逐层 position-encoding policy 与 builder | 通用 Dense 基线、RoPE 与 KV 合同，不依赖 MoE | RoPE/NoPE 混合层、未旋转 K、Prefill/多步 Decode 对照 | 可并行选做 |
| D5 Phi | fused 来源解析、partial/LongRoPE、具体版本策略 | 通用 Dense 基线与目标 RoPE 能力，不依赖 MoE | fused/分离来源等价，旋转维度及原始上下文边界，完整推理 | 可并行选做 |
| A1 local causal attention | typed window/mask、静态完整 KV 上的读取语义 | Attention/KV reference 与 08 的状态合同；不依赖 ring cache | 每 query mask、窗口边界、global/local 混合、失败不发布 | 独立能力包 |
| D6 Gemma 文本 | 一个明确版本的 norm/activation/scaling 与构图 | 目标版本需要的 A1 及其他语义；不要求完成 D3–D5 | 逐层数值、local/global KV 及真实文本 checkpoint，视觉范围不扩展 | 条件式进入 |
| X1 第二 MoE 家族 | Mixtral 或 OLMoE 先选一个 | M4 通用 MoE 集成基线；目标独有 norm/attention 行为 | 不同命名和 router 合同；OLMoE 另验 flat Q/K norm 与不重归一化概率 | 后续优先候选 |
| X2 Qwen2 MoE | 对应 attention adapter、shared expert、Sigmoid 支路 | 通用 MoE 能力、该家族的 Dense/attention 能力与 Sigmoid | routed/shared 输出分别对照后合并，完整生成与资源证据 | 按需求进入 |

“优先”表示候选选择次序，不要求把表格变成串行大项目。D3/D4 可用于快速扩大兼容性，D5/D6 由 CPU 产品需求选择；X1 用来检验 MoE 通用性。Ministral、原生 DeepSeek、GLM 等只有在独立目标与额外架构合同明确后再增加对应工作包。

## 5. 风险与依赖

| 风险 | 影响与处理 |
|---|---|
| 只开放配置/枚举 | 可能加载成功但语义错误；支持声明绑定到 checkpoint/图/准备/执行证据 |
| 公共合同或 ABI 变化 | OpType、OpParams、WeightBinding、kernel context 与 plan 同步修改；已有值不任意重排，序列化/公开 ABI 按仓库版本规则处理 |
| bank slice 或 span 悬空 | 保留 RawStorage 与 artifacts ownership；metadata 堆稳定；测试构建临时对象销毁后的执行与 move 后有效性 |
| 路由数值敏感 | 精度或 tie 改变 expert 集合；比较 logits/probabilities/IDs/专家输出，平分单独验合同 |
| 多层 scratch 与 activation 累积 | 窄 scratch 复用不等于 activation liveness；准备预算与实测峰值分别记录 |
| BF16/FP16/量化被过度承诺 | loader 接受 dtype 与 kernel 可执行分别登记；每种部署组合独立闭环 |
| 提前引入并发 | scatter-add、scratch 和 KV transaction 出现 race；首版串行，后续线程化单独设计并通过 TSAN |
| 拷贝与专家访存成本 | Reshape baseline、全专家 packing 可能增加流量/峰值；测 prepared path、端到端、RSS/page faults 后优化，不预设加速倍数 |
| checkpoint/库版本演进 | oracle 固定 Transformers 版本与 checkpoint revision；独立二维和 fused 三维专家表示分别记录与测试 |
| 品牌与底座混淆 | 基于实际 config、权重及语义识别；SmolLM2/蒸馏模型通过不等于新家族或原生 DeepSeek 已支持 |
| 将家族差异误作参数默认值 | RoPE/NoPE、norm axis、GELU 变体和 local mask 明确表达；新能力仅对已验收目标开放 |
| 将滑窗语义等同于存储压缩 | 完整静态 KV 可先保证正确读取，但仍按完整 context 分配；ring/eviction 另立资源方案 |

资源预算、workspace reuse 与 06/08 共享基础合同；其完整重构不是本提案的默认前置。若窄实现无法满足目标机器预算，应停止扩大模型支持声明，先完成有证据的资源工作包。

## 6. 验收标准

### 6.1 配置、权重与图

- [ ] model_type/architectures 相容性、缺失/null/显式非法值、未知家族和已知未支持行为有正反例；关闭滑窗与启用滑窗分别验证。
- [ ] Qwen3 覆盖 `Nq*D != H`，Q/K norm scale 为 `[D]`，norm 在 RoPE 前；Qwen2 覆盖 config 缺省仍要求 QKV bias。
- [ ] expert 数量/ID、router、各投影 shape/方向/dtype/bytes、shared expert、tied embeddings 缺失和冲突可诊断。
- [ ] fused/独立专家来源经 slices 解析得到同一逻辑权重，跨 expert 非连续区域未伪装为 contiguous。
- [ ] 图优化前后计算一致；consumer、residual、层边界、bias/norm 和权重 binding 不丢失。

### 6.2 算子、packing 与资源

- [ ] BiasAdd 与 contiguous Reshape 的合法/非法 shape、dtype、zero-size、overflow、连续性和 alias 合同覆盖。
- [ ] RouterTopK 覆盖 K=1/2/8/E、K 非法、平分、极端有限 logits、非有限值拒绝、两类 norm_topk_prob 及概率和。
- [ ] RoutedSwiGLU 覆盖空专家、集中路由、全部合法 T*K 路由、非法/重复 IDs、权重校验、F32 reference 对照。
- [ ] 多 packed ports 的缺失/重复/错端口/错 source/错 shape/dtype/recipe 均被拒绝；已有单 packed weight 和 plain 路径回归。
- [ ] bank 布局按精确公式检查 expert offset、padding、alignment、tail 与越界；packed/reference 输出一致。
- [ ] scratch planner 验证可复用区的最大尺寸/对齐、非复用区不重叠、失败不部分提交；kernel 不跨返回保留 scratch。
- [ ] 准备资源上界、OOM、乘积溢出与实际分配一致；Decode 测量边界内无 allocation/lookup/repack。

### 6.3 生产入口与数值 oracle

每个拟标记支持的组合通过 `ModelLoader → ModelCompiler → PrepareExecutableModel → InferenceSession` 和真实 CpuBackend，不用手工拼接 bindings 的测试代替集成门禁。

先采用可再生成 tiny HF checkpoints：Qwen3 Dense；Qwen3 MoE 的纯 MoE 与混合 Dense/MoE 层；按需追加 Qwen2/Mixtral/shared expert。固定权重、token IDs、Transformers 版本、checkpoint revision/hash、eval/attention 实现与 dtype。

对比 embedding、Q/K norm、RoPE、router logits/probabilities/IDs、单专家结果、加权合并、逐层 hidden、最终 logits 与 KV。覆盖 Prefill 和至少两个 Decode step，多个 prompt 长度、非零 position、context 边界、失败后 KV 不发布及后续恢复。

数值阈值按 dtype/算子/模型规模显式记录 `atol/rtol` 与 max abs/rel error，不以最终 token 碰巧相同替代数值验收。非退化 case 比较路由 ID 与 greedy token；临近平分的差异保留定位信息，不依赖其他库的平分排序作为本地合同。

tiny fixture 证明架构闭环，真实小模型验证实际 checkpoint 与词表/资源行为；完整 30B 模型还需目标机器内存和执行证据。三类证据分别标注，不相互替代。

### 6.4 验证与交付规则

按最窄相关 target/filter 开始，构建与测试命令遵循 [AGENTS.md](../../AGENTS.md) §3–4；受影响配置和 sanitizer 按变更风险运行。并发/同步变更至少验证 TSAN；性能结论必须具有 prepared operator 和生产端到端证据。

性能报告区分 Decode/Prefill、专家分布、T/E/K/H/I、hot/streaming、weight/activation/KV dtype，记录 CPU/ISA、build、commit/dirty、latency、tok/s、峰值内存与原始数据。microbenchmark 仅用于定位，不能代替模型生产路径。

每个里程碑交付时更新 capability matrix 与对应设计，标明未完成的家族/dtype/资源限制。只有已完成的工作包更新实施状态；本提案全部状态目前为待实施。

### 6.5 后续家族的附加门禁

- [ ] 兼容性目标按实际底座路由；SmolLM2/R1-Distill 通过不新增品牌枚举、不扩大原生架构声明。
- [ ] Mistral 无滑窗目标与启用窗口配置分别测试；未经 A1 验收不得将后者当作 full attention 执行。
- [ ] SmolLM3 逐层 RoPE/NoPE 解释与固定官方实现一致，绕过旋转后仍更新并正确读取 KV；混合层的 position/watermark 一致。
- [ ] Phi fused/分离权重得到同一逻辑 Q/K/V 和 Gate/Up；slice 偏移、顺序、owner、partial rotary 尾部、LongRoPE factor/长度选择与位置边界分别对照。
- [ ] Gemma norm 参数化、activation、embedding/logit/attention 缩放逐项与选定版本一致，不被已有 fusion 当作 Qwen/Llama 语义改写。
- [ ] local/global attention 在 Prefill 和多步 Decode 中覆盖窗口前/边界/边界后、非零绝对位置；不读取 causal/window 之外 token，不将 position 取模，完整静态 KV 预算不因 mask 自动缩小。
- [ ] OLMoE 的 Q/K norm 沿投影宽度，与逐 head 归一化的错误实现可被测试区分；`norm_topk_prob=false` 的权重对齐全 E softmax 的选中概率，不强制其和为 1。
- [ ] 新增家族均通过 §6.3 的生产入口与 tiny/真实 checkpoint 两层证据，继续分别记录 dtype、资源和性能；单个版本通过不自动覆盖全家族。

## 7. 一手资料与版本约束

以下资料用于核对模型语义与 checkpoint 表示；正式 oracle 必须在 fixture manifest 中固定版本/revision，而不是跟随 main。

- [Qwen3-0.6B 官方配置](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)：独立 head_dim、Q/KV heads、tied embeddings。
- [Qwen3-30B-A3B 官方配置](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)：E=128、K=8、expert I=768、norm_topk_prob 与层类型字段。
- [Qwen2.5-0.5B-Instruct 官方配置](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json)：非空 sliding_window 与关闭开关并存。
- [Transformers v4.57.1 Qwen3](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py)：Q/K norm 与 RoPE 顺序。
- [Transformers v4.57.1 Qwen2](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2/modeling_qwen2.py)：Q/K/V bias 与 O projection。
- [Transformers v4.57.1 Qwen3 MoE](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)：路由、归一化及 Dense/MoE 层选择。
- [Transformers main Qwen3 MoE](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)：fused 三维专家表示的适配参考；不作为浮动 oracle。
- [Transformers v4.57.1 Mixtral](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/mixtral/modeling_mixtral.py)、[Qwen2 MoE](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py)：后续家族路由和 shared expert 差异。
- [Qwen3-Next 官方文档](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_next)：hybrid attention 需要单独评估，本提案不承诺覆盖。
- [Mistral-7B-v0.3 官方配置](https://huggingface.co/mistralai/Mistral-7B-v0.3/blob/main/config.json)：`model_type=mistral`、`sliding_window=null` 的首个兼容目标。
- [SmolLM3-3B 官方配置](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/config.json)、[官方实现](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/smollm3/modeling_smollm3.py)：full attention 与逐层 RoPE/NoPE policy。
- [Phi-4-mini 官方配置](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/config.json)、[Transformers v4.57.1 Phi3](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/phi3/modeling_phi3.py)：phi3 底座、fused 投影、partial RoPE 与 LongRoPE。
- [Gemma3 官方模型卡](https://ai.google.dev/gemma/docs/core/model_card_3)、[Transformers v4.57.1 Gemma3](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/gemma3/modeling_gemma3.py)：文本/视觉范围及 norm、attention scaling、local/global 差异。
- [OLMoE-1B-7B-0924-SFT 官方配置](https://huggingface.co/allenai/OLMoE-1B-7B-0924-SFT/blob/main/config.json)、[Transformers v4.57.1 OLMoE](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/olmoe/modeling_olmoe.py)：E/K、未重归一化路由及投影宽度 Q/K norm。
- [SmolLM2 官方配置](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/main/config.json)、[Distill-Llama 官方配置](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/blob/main/config.json)、[Distill-Qwen 官方配置](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/blob/main/config.json)：品牌不同但复用 Llama/Qwen 底座的兼容性目标。
- [DeepSeek-V3 官方架构说明](https://huggingface.co/docs/transformers/main/en/model_doc/deepseek_v3)：原生 MLA/DeepSeekMoE 另行设计，蒸馏模型不构成其支持证据。

## 8. 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-10-19 | 1.0 | 建立 Qwen3 Dense→Qwen3 MoE 主线，明确并行工作、家族边界、多 packed bindings、资源预算和端到端门禁；未修改实现 |
| 2026-10-19 | 1.1 | 补充 Mistral/Phi/SmolLM3/Gemma 与 Mixtral/OLMoE 的优先级、底座兼容目标、分支工作包和验收；保留 Qwen3 主线与当前执行边界 |
