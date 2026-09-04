# RMSNorm 算子契约

版本：v0.2
状态：当前实现契约
适用范围：AetherMind CPU FP32 RMSNorm
更新时间：2026-09-02

本文件记录当前已实现的语义与 CPU kernel 边界。它可以随实现验证结果演进，不以早期 Phase 1 草案为不可变限制。

***

## 1. 算子语义

RMSNorm 对输入的最后一维独立归一化，并按该维的权重 `gamma` 缩放：

$$
\operatorname{mean\_square}_r = \frac{1}{H}\sum_{h=0}^{H-1} X\_{r,h}^2
$$

$$
Y\_{r,h} = X\_{r,h} \cdot \frac{1}{\sqrt{\operatorname{mean\_square}\_r + \varepsilon}} \cdot \gamma\_h
$$

其中 `r` 是将所有 leading dimensions 按 row-major 顺序展平后的逻辑 row。RMSNorm 不包含 bias / beta、residual 融合、量化或 layout transformation。

## 2. Semantic operator 契约

`InferRmsNorm` 定义 backend-independent 的 last-dimension 语义：

| 输入/输出    | 约束                                       |
| -------- | ---------------------------------------- |
| `input`  | 支持的浮点 dtype，rank `>= 1`，shape `[..., H]` |
| `weight` | 支持的浮点 dtype，rank `1`，shape `[H]`         |
| `output` | 与 `input` dtype 和 shape 相同               |

- `H` 必须为正数；静态不匹配立即拒绝，symbolic 不匹配由 runtime constraint 检查。

- leading dimensions 可以为零。此时输出是空 tensor，RMSNorm 成功完成且不进行 reduction。

- CPU FP32 kernel 只实现 FP32；其他 semantic dtype 是否可执行由具体 backend kernel 决定。

## 3. CPU FP32 kernel 契约

### 3.1 Row model

CPU entry 将 `input.shape[0 : rank - 1]` 的乘积转换为 `row_count`，并将最后一维作为 `hidden_size`：

| input shape        |     `row_count` | `hidden_size` |
| ------------------ | --------------: | ------------: |
| `[H]`              |             `1` |           `H` |
| `[S, H]`           |             `S` |           `H` |
| `[B, S, H]`        |         `B * S` |           `H` |
| `[D0, ..., Dn, H]` | `D0 * ... * Dn` |           `H` |

`row_count` 使用 checked multiplication 计算；无法由 `int64_t` 表示时，entry 返回 `InvalidArgument`。`row_count == 0` 是成功 no-op，允许 activation input/output 的 data pointer 为 null。

### 3.2 Layout / stride

- `input`、`weight`、`output` 都必须是有效 FP32 `TensorView`，且 `output.shape == input.shape`。

- 非空 tensor 的所有 stride 必须为正数，且最大 row/column address offset 必须可由 `int64_t` 表示。

- rank-1 将其唯一维度视为列维；rank-2 使用 `stride(rank - 2)` 作为 row stride。

- rank 大于 2 时，leading dimensions 必须可折叠。对每个 `d` 满足 `0 <= d < rank - 2`：

  ```text
  stride[d] == shape[d + 1] * stride[d + 1]
  ```

  因而允许每一 row 带 padding，例如 `[B, S, H]` 的 stride 可为 `[S * padded_H, padded_H, 1]`。

- 不可折叠的 higher-rank view 在语义上仍合法，但当前 CPU kernel 返回 `Unimplemented`；不应把它误报为 operator shape error。

- Reference kernel 支持任意正的 input / weight / output inner stride。

- AVX2+FMA kernel 要求三者的 inner stride 都为 `1`。

- Mutable output 的逻辑 rows 必须物理不重叠。对折叠后的 row 模型，该约束保守地要求：

  ```text
  row_count <= 1 ||
  row_stride >= (hidden_size - 1) * column_stride + 1
  ```

  其中 `row_stride = output.stride(rank - 2)`（rank-1 只有一行，天然满足），`column_stride = output.stride(rank - 1)`。row overlap 的 output view 返回 `InvalidArgument`。input 是只读 view，不做该要求。

### 3.3 Aliasing

- CPU FP32 kernel 支持 exact input/output in-place。

- Exact 要求 data pointer、dtype、shape 和 strides 完全一致；same-base 但 mapping 不同（如同 data pointer 但 strides 不同）返回 `InvalidArgument`。

- Mutable output 的逻辑 rows 必须物理不重叠，见 §3.2 的 row span 条件。

- `output` 不得与 `weight` alias；相同 data pointer 直接返回 `InvalidArgument`。指向同一 allocation 不同 offset 的部分重叠当前无法完整检测，仍由调用方保证。

- 不同 base pointer 之间的 partial overlap 由调用方禁止；当前 `TensorView` 不包含足够的 backing-storage 信息进行完整 alias 检测。

- 支持 in-place 不表示 ExecutionPlan 默认选择 in-place；kernel 只是声明能力，buffer 复用由执行层 planner 决定。

- row micro-kernel 不使用 `restrict`，避免与 exact in-place 契约冲突。

## 4. Entry 与 compute 边界

生产路径为：

```text
PrepareExecutionBindings（PreparedExecutionBindings 构建，cold path）
    -> step.kernel.params_builder（每 step 每 PreparedExecutionBindings 恰好一次）
    -> BuildRmsNormF32ReferenceArgs / BuildRmsNormF32Avx2FmaArgs
    -> ValidateAndBuildCommonRmsNormF32Args
    -> PreparedExecutionBindings 持有 compute-ready RmsNormF32KernelArgs

每次 Execute（hot path）
    -> RmsNormF32ReferenceEntry / RmsNormF32Avx2FmaEntry
    -> RunRmsNormF32Reference / RunRmsNormF32Avx2Fma
```

- prepared params（`RmsNormF32KernelArgs`）由 `PreparedExecutionBindings` 的 params arena 持有，生命周期等于 PreparedExecutionBindings；PreparedExecutionBindings 内存续期内 data pointer / shape / stride / dtype 不变，任何变化都必须重建 PreparedExecutionBindings。

- `ValidateAndBuildCommonRmsNormF32Args` 是 FP32 的唯一 binding-time 验证路径，负责 epsilon、dtype、rank、shape、row count、layout、pointer 和 stride 检查。它在 `PrepareExecutionBindings` 阶段执行，非法 binding 在该阶段返回错误，而不是在 Execute 时失败。

- AVX2+FMA 的 unit inner-stride 要求在它的 `KernelParamsBuilder` 中检查（zero-row 豁免）。

- entry 是 thin wrapper：只做一次 pointer 转换并调用 compute，不再解析 attrs、读取 TensorView 或重复 shape/stride/alias 校验。

- `RmsNormF32KernelArgs` 是只供已验证 compute primitive 使用的内部类型；compute 不重复验证，也不拥有内存。其 alignment 不超过 `max_align_t` 且 trivially destructible，满足 PreparedExecutionBindings params arena 的类型约束。

- 不提供 public `LaunchRmsNorm` / `RmsNormArgs` SDK。benchmark 和测试均通过 prepared `ResolvedKernel` 调用 production entry。

- kernel 内不使用 OpenMP 或其他隐式线程创建。当前实现保持单线程；未来 row parallelism 应由 runtime thread-pool 统一调度。

## 5. 数值与 workspace

- `epsilon` 必须 finite 且大于零，默认值为 `1.0e-5F`。

- Reference kernel 使用 double 累加。AVX2+FMA 路径允许 FP32 reduction 和不同 reduction order，因此只要求 tolerance equal，不要求 bitwise equal。

- kernel 不分配 heap memory，workspace requirement 为空。

- 不在 kernel 内修改 floating-point environment；NaN / Inf 按 IEEE-754 默认行为传播。

## 6. ISA 与可移植性

- Reference descriptor 始终注册，是当前 default-lowering 的执行路径。

- AVX2+FMA translation unit 仅在编译器同时支持 `-mavx2` 和 `-mfma` 时编译并注册；否则 binary 中只有 reference descriptor。

- AVX2+FMA descriptor 通过 `cpu_requirements = CpuFeatureSet::From({kAvx2, kFma})` 声明完整指令集要求；运行期由 backend 按 `CpuCapabilities.effective_features` 过滤，因此"支持 AVX2 但缺少 FMA"的机器会自动回退 reference descriptor。能力模型见 `docs/designs/cpu_capability_design.md`。

## 7. 验证要求

- inference：rank-0 拒绝，rank-1 及 higher-rank `[..., H]` 保持语义，zero leading dimension 成功，hidden / weight / epsilon 约束正确。

- binding / entry：rank-1、rank-2、可折叠 rank-3 / rank-4、zero-row、不可折叠 layout、row-count / address-offset / row-span overflow、reference strided layout、AVX2 inner-stride 拒绝、output row overlap 拒绝、same-pointer 异 mapping 拒绝、output/weight alias 拒绝以及 exact in-place（reference 与 AVX2+FMA 路径）。上述非法 binding 现在都在 `PrepareExecutionBindings` / params builder 阶段拒绝；binding 成功后 entry 可重复执行。

- correctness：使用 double reference 覆盖 SIMD tail、非 2 的幂 / 质数 hidden 和常见 Llama hidden size。

- benchmark：prepared benchmark 在计时循环外 prepare kernel 并构造 compute-ready params，循环内只执行 entry；legacy build-and-invoke 与 binding specialization 作为对照。
