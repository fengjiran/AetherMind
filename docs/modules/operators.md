# operators：算子语义契约层

`operators` 定义 backend-independent 的算子类型、端口 ABI、typed 参数、形状与约束推导，以及 `OpParams` 的文本序列化。它回答“一个算子在图语义上需要哪些输入输出、参数是否合法、输出 spec 与哪些运行期约束成立”，不回答“由哪个 kernel、哪块内存、哪个设备执行”。系统分层见 [architecture.md](../architecture.md)，产品边界见 [product.md](../product.md)。

## 1. 职责与边界

### 负责

| 职责 | 具体内容 |
|---|---|
| 算子类型标识 | `OpType` 枚举、`ToString(OpType)`、`std::hash<OpType>` |
| 端口 ABI | 19 个已注册算子的 `OperatorSchema`：输入/输出端口名与顺序、端口种类、是否参与 spec 推导、traits、state alias 声明 |
| typed 参数 | `OpParams`（typed variant）与每个算子的参数结构；不使用 `std::any`、无类型 `attrs` 或字符串参数袋 |
| 语义推导 | `InferOperator` 分发到 `detail::Infer*`：参数校验、输入数量校验、dtype/rank/静态维度校验、输出 `TensorSpec` 推导、deferred `ShapeConstraint` 生成 |
| 公共派生规则 | `MakeCompactInputSpecs`、`DeriveSelectorDTypes`、`ValidateInferenceInputCount` |
| 参数序列化 | `SerializeOpParams` / `ParseOpParams` / `OpParamsKindName`，以及 Reshape shape、Permute permutation 的规范文本形式 |
| dtype 契约 | 各算子的 `kXxxSupportedDTypes` / `IsXxxSupportedDType` / `MakeXxxUnsupportedDTypeMessage` 纯数据头文件，供语义校验、backend kernel dispatch 与常量折叠共用 |
| RoPE 频率契约 | backend-independent 的 `ValidateRoPEFreqParams`、`ResolveStaticRoPERotationCoefficients`、`ResolveDynamicRoPERotationCoefficients`、`ComputeDynamicNtkBase` |

当前 `OpType` 有 19 个已注册算子（`kUnknown` 是非法哨兵，无 schema）：`kEmbedding`、`kRmsNorm`、`kLinear`、`kQkvLinear`、`kMatMul`、`kRoPE`、`kAttention`、`kSilu`、`kSiluMul`、`kElementwiseMul`、`kKVCacheUpdate`、`kAdd`、`kAddRmsNorm`、`kSoftmax`、`kArgmax`、`kReshape`、`kPermute`、`kReorder`、`kGateUpLinear`。算子契约、数值语义与证据统一放在 [../operators/](../operators/)，本文件不复制契约全文。

### 不得负责什么

| 不属于 operators | 归属 |
|---|---|
| `ModelGraph` / nodes / values / liveness、rewrite session、pass pipeline | `graph`（[graph.md](graph.md)） |
| 编译期常量求值的 `ConstEvaluator::Plan` / `Evaluate`、输出 buffer 分配、`AddSessionConstant` / `ReplaceValue` | `graph` 的 constant folding（见下文） |
| 数值 reference / optimized kernel、kernel registry、kernel resolve、packing、workspace、线程与 `KernelContext` | `backend` / `execution` |
| `ExecutionPlan`、`PreparedExecutionBindings`、运行期 TensorView 绑定与 concrete shape 校验 | `execution` |
| lowering、`LoweredGraph`、`KernelSelector` 的最终 dtype/weight_format | `compiler`（[compiler.md](compiler.md)） |
| HF 配置解析、模型家族判定、HF RoPE 字段归一化、图构建 | `model` |
| 张量/buffer/权重/KV 存储的所有权与生命周期 | `base` / `memory` / `kv_cache` / `runtime` |
| Session、Generate、Token IDs、tokenizer、采样 | `inference` / API 层 |
| 整图序列化或跨进程 IR 格式 | 当前无；本模块只序列化 `OpParams` |
| 通用 shape 符号、约束求值、broadcast 基础设施 | `shape_inference`（[shape-inference.md](shape-inference.md)） |

### 常量求值与 reference evaluation 的归属

- 早期常量求值设计中的 evaluator 概念，在当前代码中实现为 `graph` 模块的 `ConstEvaluator`（[const_evaluator.h](../../include/aethermind/graph/optimization/const_evaluator.h)）。当前注册的 evaluator 只有 `Add`、`ElementwiseMul`、`Silu`、`SiluMul`（[const_evaluator.cpp](../../src/graph/optimization/const_evaluator.cpp)）。
- `operators` 只提供 evaluator 所需的前提：`OperatorTraits::compile_time_evaluable`、`IsCompileTimeEvaluable`、`OpParams`、`Infer*` 推导出的 spec 与约束。`Plan`/`Evaluate`、预算检查、输出分配与图重写全部在 `graph`，详见 [graph.md](graph.md) §3.5 / §4.2。
- 数值 reference kernel 与 optimized kernel 属于 backend；其契约、容差与证据记录在 [../operators/](../operators/) 对应算子目录，不在本模块实现。
- 因此本模块没有运行时求值入口，也不持有 TensorView、buffer 或设备上下文。

依赖方向固定为 `operators → shape_inference + dtypes + base`（实现另使用 `include/utils/` 的 variant/overflow/parse 辅助）。`operators` 不得 include `graph`、`compiler`、`execution`、`runtime`、`model`、`backend` 或 `inference`。backend/kernels 反向只允许 include 本模块的 `OpParams` / `OpType`、纯数据 dtype 契约头和 `rope_frequency_resolver.h`，这与 [AGENTS.md](../../AGENTS.md) §2.1 一致。

## 2. 关键数据结构与所有权

### 2.1 `OpType`

| 属性 | 语义 |
|---|---|
| 表示 | `enum class OpType : uint8_t`，`kUnknown = 0` 是非法哨兵 |
| 名称 | `ToString(OpType)` 对 `kUnknown` 与任意越界 `uint8_t` 转型都返回 `"Unknown"`，不抛异常 |
| 哈希 | `std::hash<OpType>` 基于底层 `uint8_t`，可用于 `unordered_map` |
| 角色 | graph 节点标识、schema 查找键、lowered step 与 kernel 注册/解析的语义类型 |

### 2.2 `OperatorSchema` 与端口 ABI

| 类型 / 字段 | 语义 |
|---|---|
| `OperatorSchema::op_type` | 关联的 `OpType` |
| `OperatorSchema::input_ports` / `output_ports` | 端口声明；**向量下标就是端口 index**，不另存 index 字段 |
| `OperatorPortKind` | `kModelInput`、`kActivation`、`kWeight`、`kConstant`、`kState` |
| `OperatorInputPort::contributes_tensor_spec` | `false` 时该输入不参与输出 spec 推导；当前用于 state 输入（KV cache） |
| `OperatorInputPort::name` / `OperatorOutputPort::name` | 端口名，借用静态 schema 定义中的字符串字面量；`FindInputPortIndex` / `FindOutputPortIndex` 按名查找 |
| `OperatorTraits` | `has_side_effects`、`deterministic`、`compile_time_evaluable`；默认分别为 `true` / `false` / `false` |
| `StateAliasPortPair` | state 输入端口名与 state 输出端口名；lowering 按名解析为端口 index 并生成 must-alias |

端口顺序是 graph 与 kernel 之间的**语义 ABI**：`ModelGraph::AddNode`、`LoweredStepSpec`、`ExecutionStep`、`ExecutionPlanNodeSpec` 与 backend kernel 都按同一 schema 顺序解释输入输出。声明顺序变化等同于 ABI 变化，必须同步 graph 构建、lowering、execution 与 kernel 测试。

traits 的判定入口：

| 函数 | 规则 |
|---|---|
| `IsPureOperator(schema)` | `!has_side_effects && deterministic` |
| `IsCompileTimeEvaluable(schema)` | `IsPureOperator(schema) && compile_time_evaluable` |

schema 注册表是 [operator_schema.cpp](../../src/operators/operator_schema.cpp) 中的静态只读 `std::array`，没有运行期注册接口，也没有可变全局状态。`GetOperatorSchemas()` 返回指向静态存储的 `std::span<const OperatorSchema>`；`GetOperatorSchema(OpType)` 返回 `StatusOr<OperatorSchema>` 的**值拷贝**，未注册类型返回 `NotFound`。

### 2.3 `OpParams`

`OpParams` 是 `std::variant<std::monostate, ...>`，包含 19 个算子的参数结构：

`EmbeddingParams`、`RmsNormParams`、`LinearParams`、`RoPEParams`、`MatMulParams`、`SoftmaxParams`、`AddParams`、`SiluParams`、`SiluMulParams`、`ElementwiseMulParams`、`KVCacheUpdateParams`、`AttentionParams`、`ArgmaxParams`、`ReshapeParams`、`PermuteParams`、`ReorderParams`、`QkvLinearParams`、`AddRmsNormParams`、`GateUpLinearParams`。

| 特性 | 语义 |
|---|---|
| `std::monostate` | 表示无参数；graph 校验要求已注册算子使用匹配的 typed alternative |
| 空参数结构 | `LinearParams`、`AddParams`、`SiluParams`、`SiluMulParams`、`ElementwiseMulParams`、`KVCacheUpdateParams`、`ReorderParams` 等只用于区分算子语义 |
| 嵌套 typed variant | `RoPEAlgorithmParams`（`StandardRoPE` / `LinearRoPE` / `DynamicNtkRoPE` / `YarnRoPE` / `Llama3RoPE` / `LongRoPE`）与 `ReshapeDim`（`ReshapeLiteralDim` / `ReshapeInputDim` / `ReshapeInferDim`） |
| 自持容器 | `LongRoPE::short_factors` / `long_factors`、`ReshapeParams::target_shape`、`PermuteParams::permutation` 按值拥有 |
| 归一化辅助 | `EffectiveRoPERotaryDim` 把 legacy `rotary_dim == 0` 解析为 `head_dim`；新语义生产者应写入规范化正值 |

`OpParams` 是值类型，随 `ModelGraph` 节点、`LoweredStepSpec` 与执行计划复制/移动；它不拥有张量、权重或 KV 存储。

### 2.4 `InferenceResult` 与 deferred `ShapeConstraint`

```cpp
struct InferenceResult {
    std::vector<TensorSpec> outputs{};
    std::vector<ShapeConstraint> runtime_checks{};
};
```

| 类型 | 语义 |
|---|---|
| `outputs` | 按 schema 输出端口顺序排列的 `TensorSpec`；只含 dtype 与 `SymbolicShape`，不含 stride、address 或物理 layout |
| `runtime_checks` | 算子语义产生、构图期无法证明、需要执行期 concrete shape 验证的约束 |
| `ShapeConstraint` | `shape_inference` 的纯数据对象；`error_context` 借用发出约束的 `Infer*` 中的 static-duration 字符串字面量 |

约束的生成原则是“局部已证明就不生成”：同一 `ShapeSymbol` identity 或两侧静态相等时不生成；两侧静态冲突时立即返回 `InvalidArgument`；其余情况生成 `DimEqualConstraint`、`DimBroadcastableConstraint`、`VolumeEqualConstraint` 或 `DimPositiveConstraint`。约束的值语义、三态求值与执行期校验见 [shape-inference.md](shape-inference.md)。

### 2.5 `SelectorDTypes`

`DeriveSelectorDTypes` 从端口 spec 推导 selector 所需的 dtype：

| 字段 | 规则 |
|---|---|
| `act_dtype` | 所有参与 spec 推导的 activation 输入必须一致；无 activation 输入时回退到 activation 输出（例如 Embedding） |
| `weight_dtype` | 所有参与 spec 推导的 weight 输入必须一致；无 weight 端口时回退到 `act_dtype` |
| `state_dtype` | 扫描完整 schema 的所有 `kState` 输入/输出端口，必须一致；无 state 端口时为 `Undefined` |

该函数是 lowering 与 execution 的共享唯一规则，避免两处各自推导导致 selector 漂移。

### 2.6 `RoPERotationCoefficients`

| 字段 | 语义 |
|---|---|
| `inv_freqs` | 每个 rotary pair 一个逆频率，长度为 `rotary_dim / 2` |
| `position_divisor` | 旋转前对 position 施加的除数；`LinearRoPE` 把 factor 放在这里 |
| `rotary_output_scale` | 旋转输出的幅度缩放；默认 `1.0`，YaRN / LongRoPE 可带前端归一化值 |

该结构由调用方按值拥有，供 backend kernel 消费；它只承载公式结果，不持有 tensor。

### 2.7 `OperatorName`

`OperatorName` 保存 `name` 与 `overload_name` 两个字符串，用于 qualified name 的兼容 dispatch 路径与诊断。它**不是**当前语义分发的权威：主路径以 `OpType` + `OperatorSchema` 为准。保留它是为了兼容既有名称查询，不应用于新增算子语义。

### 2.8 所有权与线程安全

| 对象 | 所有权 | 生命周期 / 并发 |
|---|---|---|
| schema 注册表 | 静态只读 `std::array`；`GetOperatorSchemas` 借用，`GetOperatorSchema` 返回值拷贝 | 进程级；并发只读安全 |
| 端口名 / `error_context` | 借用 static-duration 字符串字面量 | 必须长于所有复制品；不拥有、不分配 |
| `OpParams` | 值类型，按值复制/移动 | 自持容器随对象生命周期 |
| `InferenceResult` | 值类型，由 `InferOperator` 返回 | 不含 solver、view 或图指针 |
| dtype 数组 | `inline const` 头文件常量 | 只读、无全局初始化顺序问题 |
| `RoPERotationCoefficients` | 调用方按值拥有 | 不借用 params 或 tensor |
| `OperatorName` | 值类型，拥有两个 `std::string` | 可复制；比较按 `(name, overload_name)` 全序 |

`Infer*` 是纯函数：不修改全局状态、不缓存结果、不持有输入 spec 之外的引用。schema 与 dtype 常量只读，可被多个线程并发访问。

## 3. 接口与代码位置

### 3.1 文件

| 头文件 | 实现 | 内容 |
|---|---|---|
| [op_type.h](../../include/aethermind/operators/op_type.h) | [op_type.cpp](../../src/operators/op_type.cpp) | `OpType`、`ToString`、hash |
| [operator_schema.h](../../include/aethermind/operators/operator_schema.h) | [operator_schema.cpp](../../src/operators/operator_schema.cpp) | schema 类型、静态注册表、端口查找、traits 判定 |
| [op_params.h](../../include/aethermind/operators/op_params.h) | 无 | `OpParams` 与全部参数结构 |
| [operator_inference.h](../../include/aethermind/operators/operator_inference.h) | [operator_inference.cpp](../../src/operators/operator_inference.cpp) | `InferOperator`、compact specs、selector dtype 推导 |
| [inference_result.h](../../include/aethermind/operators/inference_result.h) | 无 | `InferenceResult` |
| [op_params_serde.h](../../include/aethermind/operators/op_params_serde.h) | [op_params_serde.cpp](../../src/operators/op_params_serde.cpp) | 规范文本序列化/解析、legacy RoPE 迁移 |
| [rope_frequency_resolver.h](../../include/aethermind/operators/rope_frequency_resolver.h) | [rope_frequency_resolver.cpp](../../src/operators/rope_frequency_resolver.cpp) | RoPE 频率公式、静态/动态系数解析 |
| [operator_name.h](../../include/aethermind/operators/operator_name.h) | [operator_name.cpp](../../src/operators/operator_name.cpp) | qualified name 兼容类型 |
| [ops/*.h](../../include/aethermind/operators/ops/) | [src/operators/ops/](../../src/operators/ops/) | 各算子 dtype 集合与 `detail::Infer*` 实现 |

### 3.2 公共入口

| 接口 | 语义 |
|---|---|
| `InferOperator(OpType, const OpParams&, std::span<const TensorSpec>)` | 语义推导唯一公共入口；输入按 schema 端口顺序排列 |
| `ValidateInferenceInputCount(OpType, span<const TensorSpec>)` | 校验输入数量与 schema 一致 |
| `MakeCompactInputSpecs(schema, span<const TensorSpec>)` | 过滤 `contributes_tensor_spec == false` 的端口，得到推理输入 |
| `DeriveSelectorDTypes(schema, input_specs, output_specs)` | 推导 act/weight/state dtype；spec 数量必须与完整 schema 端口数一致 |
| `GetOperatorSchema(OpType)` / `GetOperatorSchemas()` | schema 查找 / 只读全量视图 |
| `FindInputPortIndex` / `FindOutputPortIndex` | 按端口名解析 index；未知名称返回 `InvalidArgument` |
| `HasStatefulOutput` / `IsPureOperator` / `IsCompileTimeEvaluable` | traits 查询 |
| `SerializeOpParams` / `ParseOpParams` / `OpParamsKindName` | `OpParams` 规范文本序列化、解析与 kind 名 |
| `SerializeReshapeShape` / `SerializePermutation` | Reshape / Permute 的共享规范拼写 |
| `ValidateRoPEFreqParams` | 校验算法 payload（因子、context、LongRoPE 表长等） |
| `ResolveStaticRoPERotationCoefficients` | 与执行期序列长度无关的算法（Standard / Linear / YaRN / Llama3） |
| `ResolveDynamicRoPERotationCoefficients` | 依赖执行期序列长度的算法（Dynamic NTK / LongRoPE） |
| `IsDynamicRoPE` / `ComputeDynamicNtkBase` | 动态性查询 / Dynamic NTK base 计算 |

### 3.3 算子清单（端口与 traits）

输入输出名称来自 schema，顺序即 ABI；`(state)` 表示 `kState` 端口。

| OpType | 输入 → 输出 | traits |
|---|---|---|
| `kEmbedding` | `tokens`(model input), `weight`(weight) → `output` | 纯函数，非编译期可求值 |
| `kRmsNorm` | `input`, `weight` → `output` | 纯函数，非编译期可求值 |
| `kLinear` | `input`, `weight` → `output` | 纯函数，非编译期可求值 |
| `kQkvLinear` | `input`, `qkv_weight` → `q`, `k`, `v` | 纯函数，非编译期可求值 |
| `kMatMul` | `lhs`, `rhs` → `output` | 纯函数，非编译期可求值 |
| `kRoPE` | `q`, `k`, `position_ids`(model input) → `q_rope`, `k_rope` | 纯函数，非编译期可求值 |
| `kAttention` | `q`, `k_cache`(state), `v_cache`(state) → `output` | 纯函数，无 state 输出，非编译期可求值 |
| `kSilu` | `input` → `output` | 编译期可求值 |
| `kSiluMul` | `gate`, `up` → `output` | 编译期可求值 |
| `kElementwiseMul` | `lhs`, `rhs` → `output` | 编译期可求值 |
| `kAdd` | `lhs`, `rhs` → `output` | 编译期可求值 |
| `kAddRmsNorm` | `input`, `residual`, `weight` → `output`, `new_residual` | 纯函数，非编译期可求值 |
| `kSoftmax` | `input` → `output` | 纯函数，非编译期可求值 |
| `kArgmax` | `logits` → `output` | 纯函数，非编译期可求值 |
| `kKVCacheUpdate` | `k`, `v`, `k_cache_in`(state), `v_cache_in`(state) → `k_cache_out`(state), `v_cache_out`(state) | 有状态，非确定性，非编译期可求值 |
| `kReshape` | `input` → `output` | 纯函数，非编译期可求值 |
| `kPermute` | `input` → `output` | 纯函数，非编译期可求值 |
| `kReorder` | `input` → `output` | 纯函数，非编译期可求值 |
| `kGateUpLinear` | `input`, `gate_up_weight` → `gate`, `up` | 纯函数，非编译期可求值 |

关键语义不变量（实现见 [src/operators/ops/](../../src/operators/ops/)）：

| 类别 | 已验证规则 |
|---|---|
| 参数优先级 | 每个 `Infer*` 先校验 `OpParams` alternative，再校验输入数量；错误消息形如 `"<Op> node requires <Op>Params"` |
| dtype | 有 dtype 契约的算子（Reshape / Permute / Reorder 只做形状或透传）要求输入/weight 落在对应 `kXxxSupportedDTypes`；`RmsNorm` 的 input 与 weight dtype 可独立；`Linear` 的 activation 与 weight dtype 可独立；`RoPE` 要求 q/k 同 dtype；`SiluMul` 要求 gate/up 同 dtype |
| mixed dtype | 只有显式声明的 KV16 边界：`Attention` 允许 q=F32 + K/V cache=F16/BF16；`KVCacheUpdate` 允许 k/v=F32 + cache=F16/BF16；其余混合组合不做隐式转换 |
| broadcast | `Add`、`SiluMul`、`ElementwiseMul`、`MatMul` 的 batch 维使用 `InferBroadcastShape`；歧义轴生成 deferred `DimBroadcastableConstraint` |
| 静态冲突 | 静态维度不等、静态非法值、静态 volume 不等、参数越界等立即 `InvalidArgument`；`Reshape` 的 `VolumeEqualConstraint` 先做符号求值，已证伪立即失败 |
| 符号未知 | 无法证明的等式/正数/volume/broadcast 生成 `runtime_checks`；`Unknown(-1)` 不建立 identity |
| 零维 | `Linear` / `MatMul` 允许零 batch、零 in/out features 与零 K；`RmsNorm` 只要求最后一维正；`Argmax` 拒绝静态为空的 reduction 轴 |
| state | `KVCacheUpdate` 的 cache 输入不参与 compact spec 推导，输出逐字复制 cache 输入；`Attention` 只读 cache，无 state 输出；capacity 的 `current_pos` 部分属于运行期 `KVCacheSessionView`，不在图中表达 |

具体契约与数值语义见 [../operators/operator_contract_design.md](../operators/operator_contract_design.md)、[../operators/算子系统设计.md](../operators/算子系统设计.md) 及每算子目录（[RoPE](../operators/rope/RoPE算子契约.md)、[RMSNorm](../operators/rmsnorm/RMSNorm算子契约.md)、[Linear](../operators/linear/LinearOp算子契约.md)、[Attention](../operators/attention/README.md)、[KVCacheUpdate](../operators/kv_cache_update/README.md)、[Embedding](../operators/embedding/README.md)、[GEMM](../operators/gemm/README.md)）。

### 3.4 序列化边界

`op_params_serde` 只处理 `OpParams` 的规范文本形式：

| 方面 | 行为 |
|---|---|
| 格式 | `<Kind>` 后接空格分隔的 `field=value`；`RoPE` 写 `version=2`；空参数算子只写 kind |
| 特殊拼写 | Reshape `shape=[32,@0,*]`（`@N` 引用输入轴，`*` 为 infer，`[]` 为 rank zero）；Permute `permutation=[2,0,1]`（`[]` 为 rank zero） |
| 语法校验 | 拒绝未知 kind、缺失/多余/重复字段、非法数字、未加括号的列表、非规范前导零等 |
| 语义归属 | serde **不做**算子语义校验：例如重复 Permute 轴可解析成功，由 `InferPermute` 拒绝；多个 Reshape infer marker 可解析，由 `InferReshape` 拒绝 |
| legacy RoPE | 无 `version` 的旧格式（`head_dim` / `num_q_heads` / `num_kv_heads` / `max_pos_embeddings` / `theta` / `scaling_factor` / `scaling_type`）迁移为 `StandardRoPE` 或 `LinearRoPE`；未知 version / algorithm 拒绝 |
| 不做的事 | 不序列化 `ModelGraph`、`TensorSpec`、`ShapeConstraint` 或执行计划；整图 dump 属于 graph 的诊断路径 |

### 3.5 RoPE 频率解析

`rope_frequency_resolver` 是 backend-independent 的公式契约，不是 kernel：

| 算法 | 解析入口 | 关键结果 |
|---|---|---|
| `StandardRoPE` | `ResolveStaticRoPERotationCoefficients` | 基础逆频率 |
| `LinearRoPE` | 同上 | 基础逆频率 + `position_divisor = factor` |
| `YarnRoPE` | 同上 | 修正区间插值后的逆频率 + `rotary_output_scale` |
| `Llama3RoPE` | 同上 | 按波长分段缩放的逆频率 |
| `DynamicNtkRoPE` | `ResolveDynamicRoPERotationCoefficients` | 由 `effective_seq_len` 计算 Dynamic NTK base 后再生成逆频率 |
| `LongRoPE` | 同上 | 按 `effective_seq_len` 选择 short/long factor 表；表长必须等于 `rotary_dim / 2` |

`ValidateRoPEFreqParams` 校验各算法的因子、上下文长度、`beta_fast > beta_slow`、表长等；非有限或溢出的频率返回 `Overflow`。kernel 只消费 `RoPERotationCoefficients` 并应用 pairing 与布局。

### 3.6 错误语义

| 场景 | 返回值 |
|---|---|
| `OpParams` alternative 与 `OpType` 不匹配 | `InvalidArgument`，且先于输入数量检查 |
| 输入数量与 schema 不一致 | `InvalidArgument` |
| dtype / rank / 静态维度 / 参数范围非法 | `InvalidArgument` |
| 静态冲突（等式、volume、broadcast 不可兼容） | `InvalidArgument` |
| 无法静态证明的等式/正数/volume/broadcast | 成功，写入 `runtime_checks` |
| 溢出（Reshape volume、RoPE 频率等） | `Overflow` |
| schema 未注册 / `kUnknown` 查找 | `NotFound` |
| `InferOperator(kUnknown, ...)` | `InvalidArgument` |

## 4. 主流程 / 时序

### 4.1 构图期

```text
ModelGraph::AddNode(op_type, inputs, outputs, op_params)
  → GetOperatorSchema(op_type)
  → 按 schema 顺序收集所有输入 value 的 TensorSpec
  → InferOperator(op_type, op_params, all_input_specs)
      → 校验 OpParams alternative
      → ValidateInferenceInputCount
      → 逐算子参数 / dtype / rank / 静态维度校验
      → 输出 TensorSpec + deferred ShapeConstraint
  → 输出 spec 写入 GraphValue，runtime_checks 写入 GraphNode
  → ValidateAndTopologicalOrder 逐节点重放 InferOperator 并比对
```

`ModelGraph::AddNode` 不接受调用方提供的输出 `TensorSpec`；输出 spec 只能由 `InferOperator` 派生。graph 的 `Commit` 与 rewrite 校验也会重放 `InferOperator`，因此 operators 是图语义的唯一推导源。graph 侧细节见 [graph.md](graph.md)。

### 4.2 lowering 与执行计划

```text
LowerModelGraph
  → GetOperatorSchema(node.op_type)
  → 按完整 schema 端口序收集 input_specs / output_specs
  → DeriveSelectorDTypes(schema, input_specs, output_specs)
  → LoweredStepSpec{op_type, selector, op_params, input_specs, output_specs, runtime_checks}
  → 按 schema.state_alias_ports 生成 LoweredStateAlias

ExecutionPlanBuilder::Build（untrusted ExecutionPlanNodeSpec）
  → GetOperatorSchema
  → MakeCompactInputSpecs(schema, semantic_inputs)
  → InferOperator(op_type, params, compact_inputs)
  → 比对 caller 的 output_specs / runtime_checks
  → DeriveSelectorDTypes 并校验 caller selector
```

lowering 保留完整端口 spec（含不参与 compact 推理的 state 端口），execution plan 只把输出端口投影到 compact kernel 端口；具体见 [compiler.md](compiler.md) 与 [execution.md](execution.md)。

### 4.3 常量折叠与 reference evaluation

```text
ConstantFoldingPass（graph）
  → GetOperatorSchema + IsCompileTimeEvaluable(schema)
  → FindConstEvaluator(op_type)
  → 检查所有输入均为带内联字节的 ConstantValue
  → ConstEvaluator::Plan（归 graph 所有；输出布局 + 成本上界）
  → CheckFoldingBudget
  → ConstEvaluator::Evaluate（纯 C++ 数值实现）
  → AddSessionConstant + ReplaceValue
```

`operators` 只决定“该算子是否声明为 compile-time evaluable”，不参与 buffer 分配或图重写。当前可折叠算子为 `Add`、`ElementwiseMul`、`Silu`、`SiluMul`；其他算子即使纯函数也不可折叠。

evaluator 契约的已验证形态：`Plan` 接收带 spec 的 `GraphValueDesc`，返回输出布局与 `FoldingCost`，自身不分配输出内存；`Evaluate` 只读写 `TensorView` / `MutableTensorView`，返回后不保留 view 或数据指针；`ConstEvalPolicy{max_output_bytes, max_compute_ops}` 的预算由 graph 的 folding pass 统一检查，evaluator 只报告成本上界。完整契约与实现见 [const_evaluator.h](../../include/aethermind/graph/optimization/const_evaluator.h) 与 [graph.md](graph.md)。

### 4.4 RoPE 参数从模型到 kernel

```text
HF config
  → model::MakeRoPEParams（HF 字段归一化为 typed RoPEAlgorithmParams）
  → InferRoPE（标量几何、dtype、rank、deferred 形状约束）
  → backend CPU kernel 在准备阶段（元数据构建器）调用
      ResolveStaticRoPERotationCoefficients 或
      ResolveDynamicRoPERotationCoefficients(effective_seq_len)
  → RoPERotationCoefficients 交给 kernel（Dynamic NTK 在调用时计算频率）
```

`InferRoPE` 不检查 position tensor 内容；非负 position、有效 position 上界与符号宽度由执行期路径校验。

### 4.5 参数序列化流程

```text
DumpGraph / 诊断
  → DumpOpParams（graph）→ SerializeOpParams（operators）
文本输入
  → ParseOpParams（operators）→ OpParams
  → InferOperator / graph 校验负责语义判定
```

序列化只保证文本语法与规范拼写稳定，不构成图或执行计划的跨进程格式。

## 5. 关键取舍

| 选择 | 原因 | 被否方案 |
|---|---|---|
| `OpType` 枚举 + 静态 schema 表 | 查找无哈希、无注册顺序、无可变全局状态；schema 静态只读可共享 | 运行期可变 registry：初始化顺序与测试隔离复杂；字符串 key：每次查找有哈希/比较成本 |
| 端口顺序即语义 ABI，不存独立 index | 端口名、顺序、index 只有一个权威来源，避免 schema 与硬编码下标漂移 | 额外 `index` 字段或分散常量：容易出现两份不一致 |
| `OpParams` typed variant | 编译期类型检查；参数校验、dump、serde、常量折叠都有确定类型 | `std::any` / 无类型 `attrs`：错误推迟到执行期，无法静态审查 |
| `Infer*` 自由函数 + `InferOperator` 分发 | 无语义层虚调用；所有算子共享同一入口与错误优先级 | 每算子 virtual schema 对象：增加对象模型与生命周期管理，收益有限 |
| 只做局部证明，未证明的约束 deferred | 当前没有图级 shape environment；局部证明能消掉大部分 obvious check，又不会伪造证明 | 在 `Infer*` 内引入全局 solver：需要图边与统一 solver 生命周期，当前未落地 |
| 静态冲突立即失败，符号冲突 deferred | 构图期尽早暴露不可恢复错误；符号图仍可表达动态 batch/seq | 全部延迟到执行期：错误发现过晚，图可表达非法组合 |
| `ShapeConstraint` 作为纯数据 + borrowed `error_context` | 可复制进 graph/lowered/execution artifact，不引入虚函数或额外所有权；诊断文本仍可读 | 约束上放虚函数或 `std::string`：复制与生命周期成本更高 |
| state 端口用 `contributes_tensor_spec=false` 排除 | KV cache 的 layout/容量归 kv_cache，不由算子 spec 推导；避免 graph 传播物理 cache spec | 让 cache 参与普通 shape 推导：会把物理存储语义泄漏到语义图 |
| state alias 由 schema 按端口名声明 | lowering 对所有 stateful 算子通用，无需硬编码 `KVCacheUpdate` | 在 lowering 中按 `OpType` 特判：新增 stateful 算子需改 compiler |
| dtype 支持集放在算子纯数据头文件 | 语义校验、backend kernel dispatch 与常量折叠共享同一集合，错误消息一致 | 各层各维护一份：注册 kernel 与语义校验容易漂移 |
| 只声明显式 mixed-dtype 家族（A32→KV16） | 不引入隐藏 dtype promotion；精度边界由调用方显式选择 | 通用自动提升：语义层无法表达转换策略，且会掩盖精度风险 |
| serde 只校验语法，语义归 `Infer*` | 保持单一语义权威；避免 serde 与 `Infer*` 对同一不变量给出不同答案 | serde 内复制算子规则：两处规则会漂移 |
| RoPE 频率公式放在 operators | backend-independent，所有 ISA kernel 复用同一公式与校验，避免逐 kernel 复制 | 每个 kernel 自带公式：数值行为与边界检查容易分叉 |
| `compile_time_evaluable` 只是 schema trait，evaluator 在 graph | operators 不依赖 graph/session，也不持有 buffer；graph 负责重写与常量生命周期 | 把 evaluator 放进 operators：必须引入 graph 类型，破坏模块边界 |
| 保留 `OperatorName` 兼容路径，但主分发用 `OpType` | 兼容既有 qualified name 查询与诊断，同时保持新路径的强类型 | 删除名称类型：会破坏既有调用方；或以其替代 `OpType`：失去 schema 强约束 |
| 不实现整图序列化 | 当前没有跨进程/round-trip 需求；文本 dump 只用于诊断 | 把 `DumpGraph` 当稳定格式：格式与调试信息耦合 |

## 6. 测试与验证入口

构建与运行：

```bash
cmake --build build --target aethermind_unit_tests -j
./build/tests/unit/aethermind_unit_tests \
  --gtest_filter='OperatorSchema.*:OperatorSemantics*:OperatorSelectorDTypes.*:OpParamsSerde.*:RoPEFrequencyResolver.*:Operators_OpType.*'
./build/tests/unit/aethermind_unit_tests --gtest_filter='*Inference.*'
```

主要套件：

| 测试文件 | 覆盖 |
|---|---|
| [test_operator_schema.cpp](../../tests/unit/operators/test_operator_schema.cpp) | 19 条 schema 齐全、`kUnknown` 返回 `NotFound`、各算子端口名/种类/arity、`contributes_tensor_spec`、state alias 引用有效、traits 分类（compile-time evaluable 与 runtime-only 的边界） |
| [test_operator_inference.cpp](../../tests/unit/operators/test_operator_inference.cpp) | 每个算子“错误 variant 先于输入数量检查”、`kUnknown` 拒绝、`MakeCompactInputSpecs` 过滤 state、`DeriveSelectorDTypes` 的正常/回退/冲突/数量不匹配路径 |
| [test_op_params_serde.cpp](../../tests/unit/operators/test_op_params_serde.cpp) | 全部 `OpParams` variant round-trip、Embedding 显式 output dtype、legacy RoPE 迁移与未知 version 拒绝、Reshape/Permute 规范形式与畸形输入拒绝 |
| [test_op_type.cpp](../../tests/unit/operators/test_op_type.cpp) | 所有已注册 `OpType` 有字符串名、越界值返回 `"Unknown"`、结果非空 |
| [test_operator_name.cpp](../../tests/unit/operators/test_operator_name.cpp) | `OperatorName` 构造、命名空间解析、比较/排序/hash、容器使用；仅兼容路径，不覆盖 `OpType` 语义 |
| [test_rope_frequency_resolver.cpp](../../tests/unit/operators/test_rope_frequency_resolver.cpp) | Standard/Linear/Dynamic NTK/YaRN/Llama3/LongRoPE 的 golden 逆频率、非法 typed variant、静态/动态入口错配拒绝 |
| [ops/](../../tests/unit/operators/ops/) | 19 个算子各自的 `Infer*`：端口 arity、参数范围、dtype 集合、rank/静态维度冲突、deferred constraint 的存在与缺失、输出 spec/dtype |
| [test_kernel_context.cpp](../../tests/unit/operators/test_kernel_context.cpp) | 位于该目录，但覆盖 backend 的 `KernelContext` 窄字段默认值；该类型不属于 operators 模块 |

验证重点：

- 每个 `Infer*` 至少覆盖“合法推导、错误 variant、错误 arity、错误 dtype/rank、静态冲突、符号 deferred”六类路径中的适用项。
- 新增算子必须同时新增 schema、`OpParams` alternative、`Infer*`、serde、dtype 契约头与对应 `ops/test_*_inference.cpp`；若声明 `compile_time_evaluable`，还需 graph 侧 `ConstEvaluator` 与折叠测试。
- 端口顺序或名称变更必须同步 graph 构建、lowering、execution plan 与 backend kernel 测试；这类变更属于 ABI 变更。
- 契约与证据更新写入 [../operators/](../operators/) 对应算子目录，工作流见 [operator-development-workflow.md](../guides/operator-development-workflow.md)。
