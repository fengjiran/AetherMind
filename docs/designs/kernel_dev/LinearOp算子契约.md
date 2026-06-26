# LinearOp 算子契约

版本：v0.1
状态：Phase 1 契约草案
适用范围：AetherMind Phase 1 CPU-first / Llama-family dense / FP32 Linear（无 bias）
更新时间：2026-06-27
前置文档：`LinearOp算子设计与实现方案_v1.0.md`、`Operator语义层接口实施步骤_v1.0.md` Section 18
样板契约：`RMSNorm算子契约.md`

---

## 1. 算子语义

Linear 对输入张量 `X` 做线性变换 `Y = X @ W.T`，其中 `W` 为权重张量。Phase 1 不包含 bias、不融合激活、不做量化反量化，也不改变输入张量的 layout。

记号约定：

- `X` 形状 `[M, K]`（rank-2，Phase 1 仅支持 ≤2；rank-1 视作 `[1, K]`）
- `W` 形状 `[N, K]`（PyTorch/HF row-major 约定：`[out_features, in_features]`）
- `Y` 形状 `[M, N]`

逐元素语义：

$$
Y_{m,n} = \sum_{k=0}^{K-1} X_{m,k} \cdot W_{n,k}
$$

注意：因为 `W` 按 `[N, K]` 行主序存放，`W_{n,k} = W[n \cdot K + k]`，所以 kernel 不需要对 `W` 做显式转置，直接按行访问即可。这与 `X @ W.T` 等价。

在 Llama 推理中的角色：每层 DecoderLayer 包含 7 个 Linear 投影（q/k/v/o_proj + gate/up/down_proj）+ 顶层 `lm_head`，是推理最大热点。

---

## 2. 输入输出契约

### 2.1 输入

| 输入 | 语义 | Phase 1 约束 |
|---|---|---|
| `input` | 激活张量 `X` | `float32`，rank ∈ {1, 2}，shape `[M, K]` 或 `[K]`（视作 `M=1`） |
| `weight` | 权重张量 `W` | `float32`，rank-2，shape `[N, K]`（`[out_features, in_features]`） |

### 2.2 输出

| 输出 | 语义 | Phase 1 约束 |
|---|---|---|
| `output` | 线性变换结果 `Y` | 预分配 `float32`，rank 与 `input` 相同，shape 为 `input.shape[:-1] + [N]` |

### 2.3 Shape 约束

- `M > 0`、`N > 0`、`K > 0`（空 tensor 见 2.6）。
- `input.shape[-1] == weight.shape[1]`（即 `K` 一致）。
- `output.shape = input.shape[:-1] + [weight.shape[0]]`。
  - rank-1 input：`output.shape = [N]`。
  - rank-2 input：`output.shape = [M, N]`。
- `M`、`N`、`K` 在运行时动态确定；kernel 不能依赖编译期固定 shape。
- correctness 与 benchmark 至少覆盖：
  - Decode GEMV：`[1, 4096] @ [4096, 4096]`、`[1, 4096] @ [11008, 4096]`、`[1, 4096] @ [32000, 4096]`（lm_head）。
  - Prefill GEMM：`[16, 4096] @ [4096, 4096]`、`[128, 4096] @ [4096, 4096]`、`[128, 4096] @ [11008, 4096]`。
  - SIMD tail：`K` 不是 8 / 16 / 32 的倍数。
  - 非 2 的幂和质数 `K`、`N`。
  - rank-1 input（`M=1` 退化路径）。

### 2.4 Layout / stride / alignment

- Phase 1 只支持 contiguous row-major layout。
- `input` stride 必须等价于 `[K, 1]`（rank-2）或 `[1]`（rank-1）。
- `weight` stride 必须等价于 `[K, 1]`（即 `[N, K]` row-major）。
- `output` stride 必须等价于 `[N, 1]`（rank-2）或 `[1]`（rank-1）。
- 不支持 arbitrary stride、gather/scatter、blocked layout、transposed weight（如 `[K, N]`）、packed weight layout（`kPacked` 由 `WeightPrepackPlanner` 处理，见 8.2）。
- 不要求调用方提供 32B / 64B 对齐地址；CPU kernel 必须能处理 unaligned load/store。
- 后续如果引入 aligned fast path，必须保留 unaligned fallback。

### 2.5 Aliasing / in-place

- **不允许** `output` 与 `input` 或 `weight` 重叠。Linear 输出 shape 与 input shape 不同（最后一维从 `K` 变为 `N`），不存在合法 in-place 场景。
- `input` 和 `weight` 在 kernel 执行期间只读，不得被修改。
- 不允许 `output` 与 `input` 部分重叠；部分 overlap 行为未定义，后续应在 Tensor alias 检查能力完善后显式拒绝。

### 2.6 空 tensor

Phase 1 不支持空 tensor。`M <= 0`、`N <= 0` 或 `K <= 0` 必须返回 `InvalidArgument`。
（实现说明：scalar reference kernel 的三重循环在 `M=0`/`N=0` 时自然不写入，但 `K=0` 会让输出全 0 而非报错；Phase 1 在 Operator 层 `CheckInputSpecs` 拒绝 `IsPositiveIfStatic` 为假的维度，避免依赖 kernel 的隐式行为。）

---

## 3. 参数契约

| 参数 | 类型 | 默认值 | 约束 | 语义 |
|---|---|---:|---|---|
| —（无） | — | — | — | Phase 1 `LinearParams` 为空结构体，无 tunable 参数 |

Operator 层参数为 `LinearOp::Params`（`LinearParams` 别名），当前无字段。CPU kernel 不依赖 `ctx.attrs`（与 `EmbeddingOp` 一致，不覆写 attrs）。

后续可扩展方向（不在 Phase 1 范围内）：

- `alpha`：标量缩放因子，`Y = alpha * X @ W.T`。
- `bias`：偏置向量，通过 schema 扩展为 3 输入或通过 attrs 传递。
- `transpose_weight`：是否允许 weight 以 `[K, N]` layout 直接传入。

任何新增参数必须在引入时同时定义 ABI 语义、attrs 编码方式与回归测试。

---

## 4. 数值契约

### 4.1 Reference 语义

Reference 实现用于 correctness oracle，应使用更高精度累加：

$$
\operatorname{acc}_{m,n} = \sum_{k=0}^{K-1}\operatorname{double}(X_{m,k}) \cdot \operatorname{double}(W_{n,k})
$$

$$
Y_{m,n} = \operatorname{float}(\operatorname{acc}_{m,n})
$$

Reference kernel 必须按 `k` 升序单线程累加，不重排 reduction order，不使用 FMA，不使用 approximate 指令。

### 4.2 Optimized kernel 允许的差异

- Optimized kernel 不要求 bitwise equal。
- 允许使用 FP32 累加、FMA、多累加器、blocked reduction、不同 reduction order、向量化 load/store。
- 允许对 `K` 做分块累加（例如 Kahan、pairwise summation）以降低误差；若引入则需在文档中记录分块策略。
- 默认不启用全局 fast-math；不得依赖破坏 NaN/Inf 传播或舍入语义的编译选项作为 correctness 前提。
- 默认不允许 approximate 指令（如 `_mm_rsqrt_ps`）；Linear 不涉及 rsqrt，本条仅作风格约束。

### 4.3 误差指标

Phase 1 correctness 以 double reference 为基准。Linear 的累加误差随 `K` 增长，阈值需要按 `K` 缩放：

- 常规输入（`K <= 4096`）：`max_abs_diff <= 1e-3F`。
- 大 `K` 输入（`K = 8192`、`K = 11008`）：`max_abs_diff <= 1e-2F`。
- 同时记录 `max_rel_diff`，建议目标 `<= 1e-4F`，但最终阈值应由真实模型 logits/token 回归确认。
- benchmark 输出应记录 `max_abs_diff`、`max_rel_diff`、`K`，不能只报告 latency。
- 上述阈值为初始建议值，第一次端到端回归后必须重新校准。

### 4.4 特殊数值

- NaN / Inf 不做额外清洗，遵循 IEEE-754 默认传播行为。
- 若 `X` 或 `W` 中存在 NaN，对应 `Y_{m,n}` 必须为 NaN。
- 若 `X` 中存在 Inf 且对应 `W_{n,k} != 0`，结果可以是 Inf 或 NaN（Inf * 0 = NaN 的传播行为允许）。
- denormal/subnormal 不在 kernel 内显式设置 flush-to-zero；实际行为继承进程和平台的浮点环境。
- rounding 依赖默认舍入模式；kernel 不应修改全局 rounding mode。
- 特殊数值测试必须按该契约补齐，再决定是否收紧或放宽行为。

---

## 5. 性能与执行契约

### 5.1 目标 workload

| 场景 | 典型 shape（M, N, K） | 优先级 | 目标 |
|---|---|---|---|
| Decode q/k/v/o_proj | `(1, 4096, 4096)` | 最高 | 低 latency，GEMV 路径，避免线程调度开销 |
| Decode gate/up_proj | `(1, 11008, 4096)` | 最高 | GEMV，`K` 维 reduction 热点 |
| Decode lm_head | `(1, 32000, 4096)` | 高 | GEMV，输出大但 `K` 中等 |
| Prefill q_proj | `(128, 4096, 4096)` | 高 | GEMM，row 维吞吐，稳定多核扩展 |
| Prefill gate_proj | `(128, 11008, 4096)` | 高 | GEMM，`N` 较大 |
| 边界测试 | 非 2 的幂、质数 `K`/`N`、SIMD tail | 高 | correctness 优先，验证 tail 不越界 |

### 5.2 内存与 workspace

- Steady-state zero allocation。
- `ComputeWorkspaceRequirement()` 返回空 workspace（Phase 1 Scalar kernel 原地计算）。
- kernel 不得分配临时 heap buffer。
- 主访问模式：
  - `input`：连续读取 `M` 行，每行 `K` 元素。
  - `weight`：连续读取 `N` 行，每行 `K` 元素；GEMV 场景下 `weight` 可能被多次扫描。
  - `output`：连续写入 `M` 行，每行 `N` 元素。
- 后续如果引入 blocked/tiled kernel，需要 workspace 用于 tile buffer；届时覆写 `ComputeWorkspaceRequirement()` 并在文档中记录 tile 尺寸与 workspace 大小关系。
- 默认不使用 non-temporal store，除非 profiling 证明收益且不会破坏后续 cache locality。

### 5.3 多线程

- Linear 的自然并行维度是 `M`（seq_len）和 `N`（out_features）。
- Phase 1 不做多线程；scalar reference kernel 单线程执行。
- Decode 小 shape（`M=1`）不应启动多线程；当前策略应保持 `M=1` 单线程执行。
- Prefill 阶段（`M >> 1`）后续可按 `M` 维切分；线程数阈值必须由 benchmark / profiling 驱动，不能凭经验固定后长期不回归。
- 多线程实现必须保证 reduction order 不影响 correctness（即不改变误差档位），否则需要单独定义多线程精度档位。

### 5.4 ISA 与 fallback

- CPU backend 必须提供可运行 fallback 路径；高级 ISA 路径不能成为唯一 correctness 路径。
- Phase 1 第一版只注册 1 个 kernel：`kPlain + kBoth + kScalar`（reference naive triple-loop）。
- 后续扩展优先级：
  1. `kScalar + kDecode`：GEMV 优化（复用 `DotProductAvx2Unroll` 风格的内积 kernel，但注册为 `kScalar` 不强制 AVX2）。
  2. `kAVX2 + kPrefill`：blocked GEMM。
  3. `kAVX512 + kPrefill` / `kAMX + kPrefill`：高级向量化路径。
  4. `kPacked` selector 系列：消费 `WeightPrepacker` 输出。
- AVX2/FMA、AVX-512、AMX 等路径必须共享同一 reference 与测试体系。
- 不同 ISA 路径只要求 tolerance equal，不要求 bitwise equal。
- ISA dispatch 必须基于 runtime capability 和 `KernelSelector`，不能在不支持目标 ISA 的机器上执行对应指令。

---

## 6. Operator / Kernel 边界契约

### 6.1 Operator 层职责

`LinearOp` 负责：

- 校验 `LinearParams`（Phase 1 为空，直接返回 Ok）。
- 校验输入数量（必须为 2）、dtype（必须 float32）、input rank（≥ 1）、weight rank（必须为 2）。
- 校验 `input.shape[-1] == weight.shape[1]`（`K` 一致）。
- 推导输出 shape：`output.shape = input.shape[:-1] + [weight.shape[0]]`。
- 声明空 workspace。
- 根据 `OperatorContext.backend` 和 `KernelSelector` 解析 kernel。
- **不**将任何 attrs 写入 `ResolvedKernel`（`LinearParams` 为空，与 `EmbeddingOp` 一致）。
- Run 时构造 `CpuLinearParams` 并写入 `KernelContext.kernel_params`，调用 `resolved_kernel_.fn(ctx)`。

### 6.2 Kernel 层职责

执行期参数绑定：

- `ExecutionPlanBuilder::Build` 产生 `LinearOp` 并完成 `Prepare`（kernel 解析，不写 attrs），但不绑定 tensor view。
- 调用方通过 `RuntimeBindingContext::SetStepTensorBinding` 注册 per-step 的输入/输出 tensor views。
- `LayerRunner::RunStep` 检查 `OpType::kLinear`，从 `RuntimeBindingContext` 取出 tensor binding，构造 `CpuLinearParams` 并写入 `KernelContext.kernel_params`。
- `CpuLinearParams` 的生命周期由 `RunStep` 栈帧保证，覆盖同步 `Operator::Run` 的完整调用。

`CpuLinearKernelEntry` 负责：

- 校验 `ctx.kernel_params` 非空。
- 校验 `TensorView` / `MutableTensorView` 有效、dtype 正确、rank 正确、contiguous。
- 校验 `M`、`N`、`K`、shape、data pointer 和 stride 满足 CPU kernel 的低层参数前置条件。
- 构造 `LinearFp32KernelArgs` 并调用 `CpuLinearKernel`。

`CpuLinearKernel` 是已验证参数上的 typed compute primitive：

- 调用方必须保证 `LinearFp32KernelArgs` 中的指针非空、维度为正、stride 为正，并且 backing storage 覆盖所有访问元素。
- 执行数值计算并写入预分配 output。
- 不拥有输入、权重、输出内存；不延长任何指针生命周期。

---

## 7. 验证清单

### 7.1 Correctness

- [ ] 固定小 shape：`(M=1, N=4, K=4)`、`(M=3, N=4, K=8)`。
- [ ] rank-1 input：`[8] @ [4, 8]` → `[4]`（`M=1` 退化）。
- [ ] Decode q_proj：`(1, 4096, 4096)`。
- [ ] Decode gate_proj：`(1, 11008, 4096)`。
- [ ] Decode lm_head：`(1, 32000, 4096)`。
- [ ] Prefill q_proj：`(16, 4096, 4096)`、`(128, 4096, 4096)`。
- [ ] Prefill gate_proj：`(128, 11008, 4096)`。
- [ ] SIMD tail：`K` 不是 8 / 16 / 32 的倍数。
- [ ] 非 2 的幂和质数 `K`、`N`。
- [ ] 随机输入：uniform、normal、mixed sign、small magnitude、large magnitude。
- [ ] NaN / Inf / denormal 行为按契约记录。
- [ ] 与 double reference 比较 `max_abs_diff` 和 `max_rel_diff`，按 4.3 阈值缩放 `K`。
- [ ] `output` 与 `input` / `weight` 重叠应被拒绝（Phase 1 假定不重叠，alias 检查完善后补测试）。

### 7.2 Benchmark

- [ ] benchmark 不在计时循环内 malloc/free。
- [ ] 记录 `M`、`N`、`K`、dtype、ISA、线程数。
- [ ] 同时报告 optimized kernel 与 reference/baseline。
- [ ] 记录 latency、items/s（= 2*M*N*K）、GB/s、GFLOPS、speedup。
- [ ] 对 decode 小 shape（`M=1`）单独观察线程调度开销（Phase 1 单线程基线）。
- [ ] 对 prefill 大 shape 观察 row 维吞吐与 cache behavior。
- [ ] 阈值调整必须附带 before/after benchmark 数据。

---

## 8. 当前开放问题

1. **rank > 2 input 支持**：Phase 1 `ExtractArgs` 仅处理 rank ≤ 2；rank > 2（如 `[B, S, K]`）需要展平 leading dims 为 `M`。Llama 推理当前仅用 rank-2，是否在 Phase 1 扩展取决于 graph builder 的输出。
2. **`kPacked` selector 与 `WeightPrepackPlanner` 的衔接**：prepack planner 已为每个 Linear 权重创建 `kPacked` 请求；第一版只注册 `kPlain` kernel，`kPacked` 请求会被 prepacker 做 memcpy fallback。需确认 fallback 路径不引入静默性能回归。
3. **bias 支持**：Llama 部分投影（如 QKV bias）有 bias；Phase 1 不实现。后续需要决定：扩展 schema 为 3 输入，还是通过 attrs 传递 bias 指针。
4. **累加精度策略**：大 `K`（`K=11008`）下 FP32 累加误差可能超出阈值；需要决定是否在 optimized kernel 中使用 Kahan / pairwise summation，还是接受放宽阈值。
5. **多线程阈值**：Prefill 阶段按 `M` 维切分的线程数阈值需要按目标硬件和 workload 通过 benchmark 固化，而不是写死为永久策略。
6. **`test_cpu_resolve_kernel.cpp` 回归**：现有测试 `MissingKeyReturnsNullptr` 断言 `kLinear` 返回 nullptr；LinearOp 实现后必须更新该断言为 `EXPECT_NE`。
7. **in-place / alias 显式拒绝**：当前 Phase 1 假定 `output` 不与 `input`/`weight` 重叠；Tensor alias 检查能力完善后应在 Operator 层显式拒绝部分 overlap。
