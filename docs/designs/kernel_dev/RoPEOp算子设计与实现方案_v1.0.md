# RoPEOp 算子设计与实现方案 v1.0

> **文档定位**：Operator 语义层第三个算子（RoPEOp）的详细设计与实现方案
> **前置文档**：`Operator语义层接口实施步骤_v1.0.md` Section 19、`RMSNorm算子契约.md`、`LinearOp算子设计与实现方案_v1.0.md`
> **样板参照**：`RmsNormOp` / `LinearOp`（已完成审核与测试覆盖）
> **状态**：设计方案（未实现）

---

## 1. 功能目标

**语义**：对 Query 和 Key 张量施加旋转位置编码（Rotary Position Embedding），将绝对位置信息以旋转矩阵的形式融入 Q/K，使注意力分数自然具备相对位置依赖。

**数学定义**（Llama HF 约定，split-half 风格）：

对每个 head 的 `head_dim` 维向量 `x`，在位置 `p` 处：

```
inv_freq[i] = 1.0 / (theta ^ (2i / head_dim))   for i in [0, head_dim/2)
freqs[p]    = p * inv_freq                       shape [head_dim/2]
emb[p]      = concat(freqs[p], freqs[p])          shape [head_dim]
cos[p]      = cos(emb[p])                         shape [head_dim]
sin[p]      = sin(emb[p])                         shape [head_dim]

rotate_half(x):
    x1 = x[:head_dim/2],  x2 = x[head_dim/2:]
    return concat(-x2, x1)

x_rope = x * cos + rotate_half(x) * sin
```

**在 Llama 推理中的角色**：每个 DecoderLayer 的 Q/K 投影之后、Attention 之前各执行一次 RoPE（共 2 次/层），是 attention 的前置步骤。

**第一版范围**：

| 维度 | 第一版 | 后续扩展 |
|------|--------|---------|
| dtype (q/k) | float32 | BF16/FP16 |
| dtype (position_ids) | int64 | int32 |
| layout | contiguous | 非 contiguous |
| cos/sin 来源 | kernel 内即时计算 | 预计算 cache 传入 |
| scaling | 校验但不应用（kNone） | Linear/DynamicNtk/Yarn |
| packing | 不启用（仅 `kPlain`） | `kPacked` selector |
| phase | `kBoth`（reference） | `kPrefill`/`kDecode` 分离 |
| isa | `kScalar`（reference） | `kAVX2`/`kAVX512` |

**cos/sin 策略说明**：

现有 operator schema（`operator_schema.cpp`）定义 RoPE 为 3 输入（q, k, position_ids）→ 2 输出（q_rope, k_rope），无 cos/sin cache 端口。第一版 kernel 从 `RoPEParams.theta` 和 `position_ids` 即时计算 cos/sin，与现有 schema 一致。

`Operator语义层接口实施步骤_v1.0.md` Section 19.3 约定"RoPEOp 不应负责生成 cos/sin cache"——本方案将其理解为：RoPEOp 不应维护**持久化**的 cos/sin 查找表；每次调用的即时计算不在此限制范围内。预计算 cache 作为后续优化项（通过扩展 schema 增加 cos/sin 输入端口，或通过 workspace 传递），由 Model/Runtime 初始化阶段准备。

---

## 2. 文件结构规划

参照 LinearOp 样板的目录结构，新增 7 个文件：

```
include/aethermind/operators/
  └── rope_op.h                      # RoPEOp 类声明

src/operators/
  └── rope_op.cpp                    # RoPEOp 实现 + AM_REGISTER_OPERATOR

include/aethermind/backend/cpu/kernels/
  └── cpu_rope_kernel.h             # 公有 Launch 接口 + CpuRoPEParams

src/backend/cpu/kernels/rope/
  ├── rope_internal.h               # RoPEFp32KernelArgs + 内部声明
  ├── rope_entry.cpp                # KernelContext entry + AM_REGISTER_KERNEL
  └── rope_fp32_scalar.cpp          # Reference 旋转实现

tests/unit/operators/
  └── test_rope_op.cpp              # 单元测试（Validate/Infer/Prepare/Run）
```

**与 LinearOp 的差异**：RoPEOp 有 3 输入 2 输出（Linear 为 2 输入 1 输出），`CpuRoPEParams` 携带配置字段（head_dim/theta 等），且 `ValidateParams` 有实质性校验逻辑。

---

## 3. 类结构设计

### 3.1 rope_op.h

```cpp
#ifndef AETHERMIND_OPERATORS_ROPE_OP_H
#define AETHERMIND_OPERATORS_ROPE_OP_H

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator.h"

namespace aethermind {

/// Semantic operator for Rotary Position Embedding (RoPE).
///
/// Applies rotary position encoding to Query and Key tensors per-head.
/// Input convention (Llama HF):
///   q:            [seq_len, num_attention_heads * head_dim]      float32
///   k:            [seq_len, num_key_value_heads * head_dim]      float32
///   position_ids: [seq_len]                                       int64
/// Output:
///   q_rope: same shape as q
///   k_rope: same shape as k
///
/// RoPEParams (already defined in op_params.h) carries head_dim, head counts,
/// theta, and optional rope scaling config.
class RoPEOp final : public Operator {
public:
    using Params = RoPEParams;

    explicit RoPEOp(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kRoPE;
    }

    AM_NODISCARD const char* Name() const noexcept override {
        return "RoPE";
    }

    AM_NODISCARD Status ValidateParams() const override;
    AM_NODISCARD Status CheckInputSpecs(std::span<const TensorSpec> inputs) const override;
    AM_NODISCARD StatusOr<InferenceResult> InferOutputShapes(
            std::span<const TensorSpec> inputs) const override;

    AM_NODISCARD WorkspaceRequirement ComputeWorkspaceRequirement(
            std::span<const TensorSpec> inputs) const noexcept override {
        UNUSED(inputs);
        return {};
    }

    AM_NODISCARD Status Prepare(OperatorContext& ctx) override;

    AM_NODISCARD Status Run(KernelContext& ctx,
                            const RuntimeBindingContext& bindings,
                            size_t step_index) const noexcept override;

    AM_NODISCARD const ResolvedKernel& GetResolvedKernel() const noexcept override {
        return resolved_kernel_;
    }

private:
    Params params_{};
    ResolvedKernel resolved_kernel_{};
};

}// namespace aethermind

#endif
```

**设计要点**：

- 与 RmsNormOp/LinearOp 结构完全对称，降低认知负担
- `RoPEParams` 已在 `op_params.h` 中定义（含 head_dim、head counts、theta、scaling），无需新增
- `ValidateParams()` 有实质校验逻辑（与 LinearOp 空校验不同）
- `ComputeWorkspaceRequirement` 返回空（Scalar kernel 即时计算 cos/sin，不需要 scratch）
- 后续 AVX2 版本若预计算 cos/sin 到 workspace，再覆写此方法

---

## 4. 核心方法定义与实现逻辑

### 4.1 ValidateParams

镜像 `model_graph.cpp::ValidateRoPEParams` 的图层级校验，并增加 `head_dim` 偶数约束（设计文档 Section 19.2 第 3 项）：

```cpp
Status RoPEOp::ValidateParams() const {
    // Check 1: all dimensions positive
    if (params_.head_dim <= 0 || params_.num_attention_heads <= 0 ||
        params_.num_key_value_heads <= 0 || params_.max_position_embeddings <= 0) {
        return Status::InvalidArgument("RoPEParams dimensions must be positive");
    }

    // Check 2: head_dim must be even (rotation operates on pairs)
    if (params_.head_dim % 2 != 0) {
        return Status::InvalidArgument("RoPEParams head_dim must be even");
    }

    // Check 3: theta finite positive
    if (!IsFinitePositive(params_.theta)) {
        return Status::InvalidArgument("RoPEParams theta must be finite and positive");
    }

    // Check 4: scaling type must be known
    if (params_.scaling_type == HfRopeScalingType::kUnknown) {
        return Status::InvalidArgument("RoPEParams scaling type must be known");
    }

    // Check 5: default scaling must not set scaling_factor
    if (params_.scaling_type == HfRopeScalingType::kNone) {
        if (params_.scaling_factor.has_value()) {
            return Status::InvalidArgument(
                    "RoPEParams default scaling must not set a scaling factor");
        }
        return Status::Ok();
    }

    // Check 6: scaled modes require finite positive scaling_factor
    if (!params_.scaling_factor.has_value() ||
        !IsFinitePositive(*params_.scaling_factor)) {
        return Status::InvalidArgument(
                "RoPEParams scaled modes require a finite positive scaling factor");
    }
    return Status::Ok();
}
```

**验证项映射**：

| 校验项 | 来源 | 实现位置 |
|--------|------|---------|
| dimensions > 0 | ValidateRoPEParams | ValidateParams |
| head_dim 偶数 | Section 19.2 第 3 项 | ValidateParams（新增） |
| theta finite positive | ValidateRoPEParams | ValidateParams |
| scaling_type != kUnknown | ValidateRoPEParams | ValidateParams |
| scaling_factor 规则 | ValidateRoPEParams | ValidateParams |

**与 RmsNormOp/LinearOp 的差异**：

- RmsNormOp：仅校验 `eps > 0`
- LinearOp：空校验（LinearParams 为空）
- RoPEOp：6 项校验，逻辑最复杂

---

### 4.2 CheckInputSpecs

实现设计文档 Section 19.2 的 6 项检查（适配 3 输入 schema）：

```cpp
Status RoPEOp::CheckInputSpecs(std::span<const TensorSpec> inputs) const {
    if (inputs.size() != 3) {
        return Status::InvalidArgument(
                "RoPE expects exactly 3 inputs, got " + std::to_string(inputs.size()));
    }

    const auto& q_spec = inputs[0];
    const auto& k_spec = inputs[1];
    const auto& pos_spec = inputs[2];

    // Check 1: q/k dtype == float32, position_ids dtype == int64
    if (q_spec.dtype != DataType::Float32() || k_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("RoPE q/k only supports float32 in Phase 1");
    }
    if (pos_spec.dtype != DataType::Int(64)) {
        return Status::InvalidArgument("RoPE position_ids must be int64");
    }

    // Check 2: q/k rank == 2, position_ids rank == 1
    if (!HasRank(q_spec.shape, 2)) {
        return Status::InvalidArgument("RoPE q must be rank-2 [seq_len, num_heads * head_dim]");
    }
    if (!HasRank(k_spec.shape, 2)) {
        return Status::InvalidArgument("RoPE k must be rank-2 [seq_len, num_kv_heads * head_dim]");
    }
    if (!HasRank(pos_spec.shape, 1)) {
        return Status::InvalidArgument("RoPE position_ids must be rank-1 [seq_len]");
    }

    // Check 3: q.shape[1] == num_attention_heads * head_dim (if static)
    const int64_t q_hidden = params_.num_attention_heads * params_.head_dim;
    if (!IsPositiveIfStatic(q_spec.shape[1])) {
        return Status::InvalidArgument("RoPE q last dimension must be positive");
    }
    if (q_spec.shape[1].IsStatic() &&
        q_spec.shape[1].GetStaticValue() != q_hidden) {
        return Status::InvalidArgument(
                "RoPE q last dimension must equal num_attention_heads * head_dim");
    }

    // Check 4: k.shape[1] == num_key_value_heads * head_dim (if static)
    const int64_t k_hidden = params_.num_key_value_heads * params_.head_dim;
    if (!IsPositiveIfStatic(k_spec.shape[1])) {
        return Status::InvalidArgument("RoPE k last dimension must be positive");
    }
    if (k_spec.shape[1].IsStatic() &&
        k_spec.shape[1].GetStaticValue() != k_hidden) {
        return Status::InvalidArgument(
                "RoPE k last dimension must equal num_key_value_heads * head_dim");
    }

    // Check 5: q.shape[0] == k.shape[0] (seq_len match)
    if (!UnifyShapeSymbol(q_spec.shape[0], k_spec.shape[0]).ok()) {
        return Status::InvalidArgument(
                "RoPE q and k must share the same seq_len dimension");
    }

    // Check 6: position_ids.shape[0] == q.shape[0] (seq_len match, if static)
    if (!UnifyShapeSymbol(pos_spec.shape[0], q_spec.shape[0]).ok()) {
        return Status::InvalidArgument(
                "RoPE position_ids length must match q seq_len");
    }

    // Check (deferred): contiguity is a runtime check (deferred to kernel)
    return Status::Ok();
}
```

**验证项映射**（设计文档 Section 19.2 → 实现）：

| 设计文档检查项 | 实现位置 | 说明 |
|---------------|---------|------|
| 1. q/k dtype == float32 | CheckInputSpecs | 直接比较 |
| 2. q/k shape 合法 | CheckInputSpecs | rank + dim 校验 |
| 3. head_dim 可被 2 整除 | ValidateParams | 在参数校验阶段 |
| 4. cos/sin cache shape | 不适用 | 第一版无 cache，即时计算 |
| 5. output shape 与 input 一致 | InferOutputShapes | 推导而非校验 |
| 6. layout contiguous | Run/kernel | 运行时检查 |

**新增检查项**（schema 约束）：

| 检查项 | 说明 |
|--------|------|
| position_ids dtype == int64 | 匹配 Llama/HF 约定 |
| q.shape[1] == num_q_heads * head_dim | 静态可验证时校验 |
| k.shape[1] == num_kv_heads * head_dim | 静态可验证时校验 |
| q.seq_len == k.seq_len | `UnifyShapeSymbol` |
| position_ids.len == q.seq_len | `UnifyShapeSymbol` |

---

### 4.3 InferOutputShapes

```cpp
StatusOr<InferenceResult> RoPEOp::InferOutputShapes(
        std::span<const TensorSpec> inputs) const {
    if (inputs.size() != 3) {
        return Status::InvalidArgument(
                "RoPE expects exactly 3 shape inputs, got " + std::to_string(inputs.size()));
    }

    const auto& q_spec = inputs[0];
    const auto& k_spec = inputs[1];
    const auto& pos_spec = inputs[2];

    if (!HasRank(q_spec.shape, 2) || !HasRank(k_spec.shape, 2) || !HasRank(pos_spec.shape, 1)) {
        return Status::InvalidArgument("RoPE input shapes have invalid rank");
    }

    // output shapes mirror input shapes (RoPE is shape-preserving)
    TensorSpec q_rope_spec{
            .dtype = DataType::Float32(),
            .shape = SymbolicShape(std::vector<ShapeSymbol>{q_spec.shape[0], q_spec.shape[1]}),
    };
    TensorSpec k_rope_spec{
            .dtype = DataType::Float32(),
            .shape = SymbolicShape(std::vector<ShapeSymbol>{k_spec.shape[0], k_spec.shape[1]}),
    };

    InferenceResult result;
    result.outputs.push_back(std::move(q_rope_spec));
    result.outputs.push_back(std::move(k_rope_spec));

    // Emit runtime check: q.seq_len == k.seq_len if symbolic
    if (!IsStaticAndEqual(q_spec.shape[0], k_spec.shape[0])) {
        result.runtime_checks.push_back(ShapeConstraint{
                .condition = DimEqualConstraint{
                        .lhs = DimLocator{.tensor_port = TensorPort{.direction = TensorPortType::kInput, .tensor_idx = 0}, .dim_index = 0},
                        .rhs = DimLocator{.tensor_port = TensorPort{.direction = TensorPortType::kInput, .tensor_idx = 1}, .dim_index = 0}},
                .error_context = "RoPE q and k must share the same seq_len dimension",
        });
    }

    // Emit runtime check: position_ids.len == q.seq_len if symbolic
    if (!IsStaticAndEqual(pos_spec.shape[0], q_spec.shape[0])) {
        result.runtime_checks.push_back(ShapeConstraint{
                .condition = DimEqualConstraint{
                        .lhs = DimLocator{.tensor_port = TensorPort{.direction = TensorPortType::kInput, .tensor_idx = 2}, .dim_index = 0},
                        .rhs = DimLocator{.tensor_port = TensorPort{.direction = TensorPortType::kInput, .tensor_idx = 0}, .dim_index = 0}},
                .error_context = "RoPE position_ids length must match q seq_len",
        });
    }

    return result;
}
```

**shape 推导逻辑**：

- RoPE 是 shape-preserving：`q_rope.shape == q.shape`，`k_rope.shape == k.shape`
- 输出 dtype 与输入一致（float32）
- 与 RmsNormOp（单输出 pass-through）的区别：RoPE 有**两个** pass-through 输出

**runtime_checks**：

- 当 `q.seq_len` 与 `k.seq_len` 中有 symbolic dim 时，生成 `DimEqualConstraint`
- 当 `position_ids.len` 与 `q.seq_len` 中有 symbolic dim 时，生成 `DimEqualConstraint`
- 参考 LinearOp 的 `EmitsRuntimeCheckForDistinctSymbolicK` 模式

**辅助函数说明**：

`IsStaticAndEqual(a, b)` 不存在于代码库中，需手动实现为：
```cpp
// inline helper (or inline in InferOutputShapes)
auto is_static_and_equal = [](const ShapeSymbol& a, const ShapeSymbol& b) {
    return a.IsStatic() && b.IsStatic() && a == b;
};
```

---

### 4.4 Prepare

```cpp
Status RoPEOp::Prepare(OperatorContext& ctx) {
    if (ctx.backend == nullptr) {
        return Status::InvalidArgument("RoPE Prepare requires OperatorContext.backend");
    }

    const auto resolved = ctx.backend->ResolveKernelInfo(
            OpType::kRoPE,
            ctx.selector);

    if (!resolved.ok()) {
        return resolved.status();
    }

    resolved_kernel_ = resolved.value();
    if (resolved_kernel_.fn == nullptr) {
        return Status::Internal("RoPE Prepare resolved a kernel with null fn");
    }
    // RoPE config (theta, head_dim, etc.) is passed per-call via CpuRoPEParams,
    // not via attrs. This differs from RmsNormOp (eps in attrs) because RoPE
    // has multiple config fields.
    return Status::Ok();
}
```

**与 RmsNormOp/LinearOp 的差异**：

| 方面 | RmsNormOp | LinearOp | RoPEOp |
|------|-----------|----------|--------|
| attrs 覆写 | 是（4 字节 eps） | 否（空 params） | 否（config 走 CpuRoPEParams） |
| config 传递路径 | attrs → ctx.attrs | 无 config | CpuRoPEParams 字段 |

**设计决策：config 走 CpuRoPEParams 而非 attrs 的理由**：

1. RoPE config 包含 head_dim、num_heads、theta、scaling 等多字段，序列化到 `attrs`（`std::vector<std::byte>`）需 reinterpret_cast，类型安全性差
2. RmsNorm 的 eps 是单一 4 字节标量，attrs 方式简洁；RoPE config 为 40+ 字节结构，attrs 方式收益低
3. `CpuRoPEParams` 在 `Run` 中栈上构造，config 字段直接赋值，kernel 通过 `ctx.kernel_params` 获取类型安全访问
4. attrs 仍可用于后续 kernel 需要的额外元数据（如 debug 信息），第一版不使用

---

### 4.5 Run

```cpp
Status RoPEOp::Run(KernelContext& ctx,
                   const RuntimeBindingContext& bindings,
                   size_t step_index) const noexcept {
    if (resolved_kernel_.fn == nullptr) {
        return Status::FailedPrecondition("RoPE Run called before Prepare");
    }

    const auto binding = bindings.GetStepTensorBinding(step_index);
    if (!binding.ok()) {
        return binding.status();
    }

    const auto* b = binding.value();
    if (b->inputs.size() != 3) {
        return Status::InvalidArgument(
                "RoPE requires 3 input tensor bindings, got " +
                std::to_string(b->inputs.size()));
    }
    if (b->outputs.size() != 2) {
        return Status::InvalidArgument(
                "RoPE requires 2 output tensor bindings, got " +
                std::to_string(b->outputs.size()));
    }

    cpu::CpuRoPEParams params{
            .q = b->inputs[0],
            .k = b->inputs[1],
            .position_ids = b->inputs[2],
            .q_rope = b->outputs[0],
            .k_rope = b->outputs[1],
            .head_dim = params_.head_dim,
            .num_q_heads = params_.num_attention_heads,
            .num_kv_heads = params_.num_key_value_heads,
            .theta = params_.theta,
    };
    ctx.kernel_params = &params;
    return resolved_kernel_.fn(ctx);
}
```

**与 RmsNormOp/LinearOp 的差异**：

- 3 输入 2 输出（RmsNorm/Linear 为 2 输入 1 输出）
- `CpuRoPEParams` 携带 config 字段（head_dim、num_heads、theta）
- 第一版不传递 scaling 配置（scaling_type == kNone 时不需要）

---

## 5. CPU Kernel 实现方案

### 5.1 cpu_rope_kernel.h（公有接口）

```cpp
#ifndef AETHERMIND_BACKEND_CPU_KERNELS_CPU_ROPE_KERNEL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_CPU_ROPE_KERNEL_H

#include "aethermind/base/tensor_view.h"

#include <cstdint>

namespace aethermind::cpu {

/// Per-call kernel params for CPU RoPE kernel.
/// Lifetime: stack-bound during RoPEOp::Run, valid for the duration of fn(ctx).
struct CpuRoPEParams {
    TensorView q{};
    TensorView k{};
    TensorView position_ids{};
    MutableTensorView q_rope{};
    MutableTensorView k_rope{};
    int64_t head_dim{};
    int64_t num_q_heads{};
    int64_t num_kv_heads{};
    double theta{};
};

}// namespace aethermind::cpu

#endif
```

---

### 5.2 rope_internal.h（内部结构）

```cpp
#ifndef AETHERMIND_BACKEND_CPU_KERNELS_ROPE_ROPE_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_ROPE_ROPE_INTERNAL_H

#include "aethermind/base/status.h"

#include <cstdint>

namespace aethermind::cpu {

/// Stripped-down kernel args extracted from TensorView for raw compute loops.
struct RoPEFp32KernelArgs {
    const float* q{};
    const float* k{};
    const int64_t* position_ids{};
    float* q_rope{};
    float* k_rope{};
    int64_t seq_len{};
    int64_t head_dim{};
    int64_t num_q_heads{};
    int64_t num_kv_heads{};
    double theta{};
};

/// Reference scalar RoPE: applies rotary embedding per-head per-position.
/// Implements Llama HF split-half convention with on-the-fly cos/sin computation.
Status RoPEKernel_CPU_FP32_Scalar(const RoPEFp32KernelArgs& args) noexcept;

}// namespace aethermind::cpu

#endif
```

**关键设计**：

- `position_ids` 为 `const int64_t*`（匹配 schema int64 约定）
- `seq_len` 从 q.shape[0] 提取
- cos/sin 在 kernel 内即时计算，不预存 cache

---

### 5.3 rope_fp32_scalar.cpp（Reference 实现）

```cpp
#include "rope_internal.h"

#include <cmath>

namespace aethermind::cpu {

namespace {

/// Computes cos/sin for a single position, writing into head_dim-sized buffers.
/// Llama HF convention: emb = concat(freqs, freqs), cos = cos(emb), sin = sin(emb).
void ComputeCosSin(int64_t position, int64_t head_dim, double theta,
                   float* cos_buf, float* sin_buf) {
    const int64_t half = head_dim / 2;
    for (int64_t i = 0; i < half; ++i) {
        // inv_freq[i] = 1.0 / (theta ^ (2i / head_dim))
        const double exponent = 2.0 * static_cast<double>(i) / static_cast<double>(head_dim);
        const double inv_freq = 1.0 / std::pow(theta, exponent);
        const double angle = static_cast<double>(position) * inv_freq;
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        // split-half: first half and second half share the same cos/sin
        cos_buf[i] = static_cast<float>(c);
        cos_buf[i + half] = static_cast<float>(c);
        sin_buf[i] = static_cast<float>(s);
        sin_buf[i + half] = static_cast<float>(s);
    }
}

/// Applies rotary embedding to a single head's vector (split-half convention).
/// rotate_half(x): x1 = x[:half], x2 = x[half:], return concat(-x2, x1)
/// x_rope = x * cos + rotate_half(x) * sin
void ApplyRoPE(const float* x, const float* cos, const float* sin,
               int64_t head_dim, float* out) {
    const int64_t half = head_dim / 2;
    for (int64_t i = 0; i < half; ++i) {
        const float x1 = x[i];
        const float x2 = x[i + half];
        // rotate_half(x)[i] = -x2, rotate_half(x)[i+half] = x1
        out[i] = x1 * cos[i] - x2 * sin[i];
        out[i + half] = x2 * cos[i + half] + x1 * sin[i + half];
    }
}

}// namespace

Status RoPEKernel_CPU_FP32_Scalar(const RoPEFp32KernelArgs& a) noexcept {
    // Per-position, per-head rotation.
    // q layout: [seq_len, num_q_heads * head_dim] row-major.
    // k layout: [seq_len, num_kv_heads * head_dim] row-major.
    const int64_t q_stride = a.num_q_heads * a.head_dim;
    const int64_t k_stride = a.num_kv_heads * a.head_dim;

    // Reusable cos/sin buffer (stack-local, head_dim floats each).
    std::vector<float> cos_buf(a.head_dim);
    std::vector<float> sin_buf(a.head_dim);

    for (int64_t pos_idx = 0; pos_idx < a.seq_len; ++pos_idx) {
        const int64_t position = a.position_ids[pos_idx];
        ComputeCosSin(position, a.head_dim, a.theta, cos_buf.data(), sin_buf.data());

        const float* q_row = a.q + pos_idx * q_stride;
        float* q_out_row = a.q_rope + pos_idx * q_stride;
        for (int64_t h = 0; h < a.num_q_heads; ++h) {
            ApplyRoPE(q_row + h * a.head_dim, cos_buf.data(), sin_buf.data(),
                      a.head_dim, q_out_row + h * a.head_dim);
        }

        const float* k_row = a.k + pos_idx * k_stride;
        float* k_out_row = a.k_rope + pos_idx * k_stride;
        for (int64_t h = 0; h < a.num_kv_heads; ++h) {
            ApplyRoPE(k_row + h * a.head_dim, cos_buf.data(), sin_buf.data(),
                      a.head_dim, k_out_row + h * a.head_dim);
        }
    }
    return Status::Ok();
}

}// namespace aethermind::cpu
```

**性能特征**：

- 朴素实现：O(seq_len × (num_q_heads + num_kv_heads) × head_dim)
- 每个位置计算一次 cos/sin（head_dim 次三角函数调用），复用于所有 head
- 无向量化、无线程并行
- Decode 阶段（seq_len=1）开销极低
- Prefill 阶段（seq_len>>1）cos/sin 计算可复用（相同 position 的 cos/sin 跨 head 相同）

**与 RmsNormOp kernel 的差异**：

| 方面 | RmsNormOp | RoPEOp |
|------|-----------|--------|
| 计算模式 | elementwise | per-head per-position 旋转 |
| 三角函数 | 无 | cos/sin（即时计算） |
| 访存模式 | 顺序读写 | per-head stride 访问 |
| AVX2 优化 | 已实现 | 待实现 |

---

### 5.4 rope_entry.cpp（Kernel 注册入口）

```cpp
#include "aethermind/backend/cpu/kernels/cpu_rope_kernel.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "rope_internal.h"

#include <vector>

namespace aethermind {

namespace {

RoPEFp32KernelArgs ExtractArgs(const KernelContext& ctx) noexcept {
    const auto& p = *static_cast<const cpu::CpuRoPEParams*>(ctx.kernel_params);
    const auto& q = p.q;
    const auto& k = p.k;
    const auto& pos = p.position_ids;

    const int64_t seq_len = q.shape()[0];
    return RoPEFp32KernelArgs{
            .q = static_cast<const float*>(q.data()),
            .k = static_cast<const float*>(k.data()),
            .position_ids = static_cast<const int64_t*>(pos.data()),
            .q_rope = static_cast<float*>(p.q_rope.data()),
            .k_rope = static_cast<float*>(p.k_rope.data()),
            .seq_len = seq_len,
            .head_dim = p.head_dim,
            .num_q_heads = p.num_q_heads,
            .num_kv_heads = p.num_kv_heads,
            .theta = p.theta,
    };
}

Status CpuRoPEKernelEntry_FP32_Scalar(const KernelContext& ctx) noexcept {
    return cpu::RoPEKernel_CPU_FP32_Scalar(ExtractArgs(ctx));
}

}// namespace

// Phase 1: register one reference kernel for kPlain + kBoth + Scalar.
// Future: add AVX2/Prefill/Decode variants and precomputed cos/sin cache.
AM_REGISTER_KERNEL(OpType::kRoPE,
                   KernelSelector{
                           .device_type = DeviceType::kCPU,
                           .act_dtype = DataType::Float32(),
                           .weight_dtype = DataType::Float32(),
                           .weight_format = WeightFormat::kPlain,
                           .isa = IsaLevel::kScalar,
                           .phase = ExecPhase::kBoth,
                   },
                   &CpuRoPEKernelEntry_FP32_Scalar);

}// namespace aethermind
```

**注册策略**：

- 第一版只注册 1 个 kernel：`kPlain + kBoth + kScalar`
- `kBoth` 意味着 prefill 和 decode 都用同一个 reference kernel
- 后续扩展优先级：
  1. `kScalar + kDecode`（seq_len=1 快速路径，省略 cos/sin buffer 分配）
  2. `kAVX2 + kBoth`（SIMD 相邻 pair 处理，减少 shuffle）
  3. 预计算 cos/sin cache（扩展 schema 或 workspace 传递）

---

## 6. 参数说明

### 6.1 Operator 层参数（RoPEParams，已存在）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `head_dim` | int64_t | 0 | 每个 head 的维度，必须为正偶数 |
| `num_attention_heads` | int64_t | 0 | Q 的 head 数量，必须为正 |
| `num_key_value_heads` | int64_t | 0 | K 的 head 数量（GQA 时 < num_attention_heads），必须为正 |
| `max_position_embeddings` | int64_t | 0 | 支持的最大位置数，必须为正 |
| `theta` | double | 10000.0 | RoPE base frequency，必须 finite positive |
| `scaling_factor` | optional<double> | nullopt | RoPE scaling 因子，kNone 时不应设置 |
| `scaling_type` | HfRopeScalingType | kNone | RoPE scaling 类型，不能为 kUnknown |

### 6.2 Kernel 层参数（CpuRoPEParams）

| 参数 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `q` | TensorView | binding.inputs[0] | Query，shape `[seq_len, num_q_heads * head_dim]` |
| `k` | TensorView | binding.inputs[1] | Key，shape `[seq_len, num_kv_heads * head_dim]` |
| `position_ids` | TensorView | binding.inputs[2] | 位置 ID，shape `[seq_len]`，int64 |
| `q_rope` | MutableTensorView | binding.outputs[0] | 旋转后的 Q，shape 同 q |
| `k_rope` | MutableTensorView | binding.outputs[1] | 旋转后的 K，shape 同 k |
| `head_dim` | int64_t | params_.head_dim | 每个 head 维度 |
| `num_q_heads` | int64_t | params_.num_attention_heads | Q head 数量 |
| `num_kv_heads` | int64_t | params_.num_key_value_heads | K head 数量 |
| `theta` | double | params_.theta | base frequency |

### 6.3 KernelSelector 配置

| 字段 | 第一版取值 | 后续扩展 |
|------|-----------|---------|
| `device_type` | `kCPU` | GPU |
| `act_dtype` | `Float32` | BF16/FP16 |
| `weight_dtype` | `Float32` | （RoPE 无 weight，占位） |
| `weight_format` | `kPlain` | `kPacked` |
| `isa` | `kScalar` | `kAVX2`/`kAVX512` |
| `phase` | `kBoth` | `kPrefill`/`kDecode` 分离 |

---

## 7. 数据处理流程

### 7.1 构建期（Graph Construction → Plan Building）

```
ModelGraphBuilder::AppendDenseLlamaLayerNodes
    ↓
operator_schema.cpp: kRoPE schema = {q(Activation), k(Activation), position_ids(ModelInput)}
                                       → {q_rope, k_rope}
    ↓
ValidateRoPEParams(RoPEParams)         → 6 项参数校验
    ↓
ExecutionPlanBuilder::Build()
    ├── op.ValidateParams()             → 镜像 ValidateRoPEParams + head_dim 偶数
    ├── op.CheckInputSpecs(inputs)      → 6 项 spec 校验
    ├── op.InferOutputShapes(inputs)    → 2 个 pass-through output + runtime_checks
    ├── op.ComputeWorkspaceRequirement()→ 空（Phase 1）
    └── op.Prepare(op_ctx)             → ResolveKernelInfo(kRoPE, selector)
```

### 7.2 执行期（Runtime Execution）

```
Executor::Execute(plan, bindings)
    ↓
for each step:
    RoPEOp::Run(kernel_ctx, bindings, step_index)
        ├── 检查 resolved_kernel_.fn != nullptr
        ├── bindings.GetStepTensorBinding(step_index)
        ├── 校验 inputs.size()==3, outputs.size()==2
        ├── 构造 CpuRoPEParams{q, k, pos, q_rope, k_rope, head_dim, ...}
        ├── ctx.kernel_params = &params
        └── resolved_kernel_.fn(ctx)
                ↓
            CpuRoPEKernelEntry_FP32_Scalar(ctx)
                ├── ExtractArgs(ctx) → RoPEFp32KernelArgs{seq_len, head_dim, ...}
                └── RoPEKernel_CPU_FP32_Scalar(args)
                        ├── for each position:
                        │     ├── ComputeCosSin(position, head_dim, theta)
                        │     ├── for each q_head: ApplyRoPE(q_row, cos, sin)
                        │     └── for each k_head: ApplyRoPE(k_row, cos, sin)
                        └── return Ok
```

### 7.3 Shape 推导示例

| 场景 | q shape | k shape | position_ids shape | q_rope shape | k_rope shape |
|------|---------|---------|---------------------|-------------|-------------|
| Prefill (Llama-7B) | `[128, 4096]` | `[128, 1024]` | `[128]` | `[128, 4096]` | `[128, 1024]` |
| Decode (Llama-7B) | `[1, 4096]` | `[1, 1024]` | `[1]` | `[1, 4096]` | `[1, 1024]` |
| Prefill (Llama-70B GQA) | `[256, 8192]` | `[256, 1024]` | `[256]` | `[256, 8192]` | `[256, 1024]` |
| 符号化 seq_len | `[S, 4096]` | `[S, 1024]` | `[S]` | `[S, 4096]` | `[S, 1024]` |

注：Llama-7B head_dim=128, num_heads=32, num_kv_heads=32（MHA）；Llama-70B head_dim=128, num_heads=64, num_kv_heads=8（GQA）。

---

## 8. 边界情况处理策略

| 边界情况 | 处理策略 | 实现位置 |
|---------|---------|---------|
| head_dim 为奇数 | ValidateParams 返回 InvalidArgument | ValidateParams |
| head_dim=0 | ValidateParams 返回 InvalidArgument | ValidateParams |
| num_kv_heads != num_attention_heads（GQA） | 支持：q/k 独立处理 | kernel 双循环 |
| position_ids 超出 max_position_embeddings | Phase 1 不校验（kernel 直接计算） | 后续可加 runtime 检查 |
| seq_len=0 | kernel 循环不执行，输出为空 | triple-loop 自然处理 |
| seq_len=1（Decode） | 正常处理，cos/sin 只算 1 个位置 | kernel 自然优化 |
| position_ids 含负数 | Phase 1 不校验 | 后续可加 runtime 检查 |
| 动态 shape（symbolic seq_len） | InferOutputShapes 生成 DimEqualConstraint | runtime_checks |
| q/k seq_len 不匹配 | CheckInputSpecs 返回 InvalidArgument（静态时）；runtime_checks（符号化时） | UnifyShapeSymbol + DimEqualConstraint |
| position_ids 与 q seq_len 不匹配 | 同上 | 同上 |
| fn 为 null（backend 返回空） | Prepare 返回 Internal | Prepare 前置检查 |
| 未 Prepare 直接 Run | Run 返回 FailedPrecondition | Run 前置检查 |
| 输入/输出数量错误 | Run 返回 InvalidArgument | Run 前置检查 |
| 非 contiguous 数据 | Phase 1 假定 contiguous，不校验 | 后续 kernel 可加校验 |
| scaling_type != kNone | ValidateParams 校验通过，但 kernel 不应用 scaling | 第一版仅支持 kNone 路径 |

---

## 9. 使用示例

### 9.1 图构建（ModelGraphBuilder 已有，无需新增代码）

```cpp
// src/model/graph/model_graph_builder.cpp 中已存在：
// RoPEParams 由 MakeRoPEParams(config, head_dim) 从 HfModelConfig 构造
graph.AddNode(GraphNode{
        .op_type = OpType::kRoPE,
        .inputs = {q_value, k_value, position_ids_value},
        .outputs = {q_rope_value, k_rope_value},
        .op_params = RoPEParams{
                .head_dim = 128,
                .num_attention_heads = 32,
                .num_key_value_heads = 32,
                .max_position_embeddings = 2048,
                .theta = 10000.0,
                .scaling_type = HfRopeScalingType::kNone,
        },
        .debug_name = "rope",
});
```

### 9.2 单元测试（参照 LinearOp 测试模式）

```cpp
// tests/unit/operators/test_rope_op.cpp

// 1. ValidateParams
TEST(RoPEOp, RejectsOddHeadDim) {
    RoPEOp op{RoPEParams{.head_dim = 127, .num_attention_heads = 32,
                         .num_key_value_heads = 32, .max_position_embeddings = 2048,
                         .theta = 10000.0, .scaling_type = HfRopeScalingType::kNone}};
    EXPECT_FALSE(op.ValidateParams().ok());
}

TEST(RoPEOp, AcceptsValidParams) {
    RoPEOp op{RoPEParams{.head_dim = 128, .num_attention_heads = 32,
                         .num_key_value_heads = 32, .max_position_embeddings = 2048,
                         .theta = 10000.0, .scaling_type = HfRopeScalingType::kNone}};
    EXPECT_TRUE(op.ValidateParams().ok());
}

// 2. CheckInputSpecs
TEST(RoPEOp, ValidatesInputContract) {
    RoPEOp op{MakeValidRoPEParams(/*head_dim=*/128, /*num_q=*/32, /*num_kv=*/32)};
    const TensorSpec inputs[3] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 4096})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 4096})},
            TensorSpec{.dtype = DataType::Int(64),   .shape = StaticShape({4})},
    };
    EXPECT_TRUE(op.CheckInputSpecs(inputs).ok());
}

TEST(RoPEOp, RejectsPositionIdsDtypeFloat) {
    RoPEOp op{MakeValidRoPEParams(128, 32, 32)};
    const TensorSpec inputs[3] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 4096})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 4096})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4})},  // wrong dtype
    };
    EXPECT_FALSE(op.CheckInputSpecs(inputs).ok());
}

// 3. InferOutputShapes
TEST(RoPEOp, InfersPassThroughOutputShapes) {
    RoPEOp op{MakeValidRoPEParams(128, 32, 8)};
    const TensorSpec inputs[3] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 4096})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 1024})},
            TensorSpec{.dtype = DataType::Int(64),   .shape = StaticShape({4})},
    };
    auto inference = op.InferOutputShapes(inputs);
    ASSERT_TRUE(inference.ok());
    ASSERT_EQ(inference->outputs.size(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 4);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 4096);
    EXPECT_EQ(inference->outputs[1].shape[0].GetStaticValue(), 4);
    EXPECT_EQ(inference->outputs[1].shape[1].GetStaticValue(), 1024);
}

// 4. Prepare + Run（FakeBackend + StubKernel，参照 test_linear_op.cpp 模式）
TEST(RoPEOp, RunInvokesKernelAndReturnsOk) {
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel(OpType::kRoPE, &StubRoPEKernel);
    RoPEOp op{MakeValidRoPEParams(128, 32, 8)};
    OperatorContext op_ctx{.backend = &backend};
    ASSERT_TRUE(op.Prepare(op_ctx).ok());

    RoPEBindingBuilder builder;  // q[4,4096] + k[4,1024] + pos[4] → q_rope[4,4096], k_rope[4,1024]
    RuntimeBindingContext bindings;
    bindings.SetStepTensorBinding(0, builder.Build());

    KernelContext kernel_ctx;
    ASSERT_TRUE(op.Run(kernel_ctx, bindings, 0).ok());
    EXPECT_TRUE(g_stub_state.called);
}
```

### 9.3 端到端推理（未来 Executor 集成）

```cpp
// Executor 自动调度，无需手动调用 RoPEOp
RuntimeBindingContext bindings;
bindings.SetStepTensorBinding(step_idx, StepTensorBinding{
        .inputs = {q_view, k_view, position_ids_view},
        .outputs = {q_rope_view, k_rope_view},
});
Executor::Execute(plan, bindings);
// RoPEOp::Run 被自动调用
```

---

## 10. 实施步骤

| 步骤 | 文件 | 内容 | 依赖 |
|------|------|------|------|
| 1 | `include/aethermind/backend/cpu/kernels/cpu_rope_kernel.h` | `CpuRoPEParams` 结构 | 无 |
| 2 | `src/backend/cpu/kernels/rope/rope_internal.h` | `RoPEFp32KernelArgs` + 声明 | 步骤 1 |
| 3 | `src/backend/cpu/kernels/rope/rope_fp32_scalar.cpp` | Reference 旋转实现 | 步骤 2 |
| 4 | `src/backend/cpu/kernels/rope/rope_entry.cpp` | Entry + `AM_REGISTER_KERNEL` | 步骤 1,3 |
| 5 | `include/aethermind/operators/rope_op.h` | `RoPEOp` 类声明 | 无 |
| 6 | `src/operators/rope_op.cpp` | 实现 + `AM_REGISTER_OPERATOR` | 步骤 1,5 |
| 7 | `tests/unit/operators/test_rope_op.cpp` | 单元测试（Validate/Infer/Prepare/Run） | 步骤 5,6 |

**注**：步骤 1-4 为 kernel 实现，步骤 5-6 为语义层实现。若先实现语义层（暂不实现 kernel，参照 LinearOp 当前状态），可仅执行步骤 1,5,6,7，kernel 部分后续补全。

---

## 11. 风险与注意事项

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| **cos/sin 即时计算性能** | 每次 Run 重新计算 cos/sin，重复位置无复用 | 第一版仅验证正确性；后续预计算 cache 并通过 schema 扩展或 workspace 传递 |
| **scaling 未实现** | scaling_type != kNone 时 kernel 不应用 scaling，结果错误 | ValidateParams 校验通过但 kernel 忽略；第一版仅支持 kNone（Llama 默认）；文档明确标注 |
| **position_ids 负数** | 负 position 产生无意义旋转 | Phase 1 不校验；后续可加 runtime 检查 |
| **GQA num_kv_heads < num_q_heads** | q/k hidden_dim 不同，需分别处理 | kernel 双循环独立处理 q/k，已支持 |
| **cos/sin 数值精度** | float32 cos/sin 在大 position 时精度下降 | 第一版 float32 足够；后续可内部用 double 计算再转 float（当前 ComputeCosSin 已用 double） |
| **ExtractArgs 假定 rank-2** | q/k rank != 2 时 shape()[0] 访问越界 | CheckInputSpecs 已校验 rank==2；kernel 不重复校验 |
| **非 contiguous 数据** | kernel 直接按 row-major 访问 | Phase 1 假定 contiguous；后续可在 Run 中加 `is_contiguous()` 校验 |
| **回归测试** | 注册 kRoPE 后，使用 kRoPE 做 fallback 测试的用例可能受影响 | 检查 `test_execution_plan.cpp`、`test_executor_backend_path.cpp` 中 kRoPE 用法，必要时替换为未注册 OpType |
| **std::to_string in noexcept Run** | Run 为 noexcept 但错误路径调用 std::to_string（可能抛异常） | 已知 P2 风险（与 RmsNorm/Linear 同）；后续统一改为不抛异常的格式化 |

---

## 12. 方案总结

RoPEOp 实现复用 RmsNormOp/LinearOp 样板模式，核心差异在于：

1. **多输入多输出**：3 输入（q, k, position_ids）→ 2 输出（q_rope, k_rope），首个多输出算子
2. **ValidateParams 实质校验**：6 项参数校验（含 head_dim 偶数约束），逻辑最复杂
3. **config 传递**：走 CpuRoPEParams 字段而非 attrs（多字段 config 不适合 attrs 序列化）
4. **shape 推导**：双 pass-through 输出 + 2 条 runtime_checks（seq_len 一致性）
5. **CPU kernel**：per-head per-position 旋转，含即时 cos/sin 计算（Llama HF split-half 约定）

总计新增 7 个文件，预计实现量略大于 LinearOp（多输入多输出 + 实质 ValidateParams）。

---

## 附录 A：与 RmsNormOp/LinearOp 样板对比

| 方面 | RmsNormOp | LinearOp | RoPEOp |
|------|-----------|----------|--------|
| 语义 | 归一化 | 线性变换 | 旋转位置编码 |
| 输入数量 | 2 | 2 | 3 |
| 输出数量 | 1 | 1 | 2 |
| Params | `RmsNormParams{eps}` | `LinearParams{}`（空） | `RoPEParams{head_dim, heads, theta, ...}` |
| ValidateParams | 仅 eps > 0 | 空校验 | 6 项校验 |
| dtype (activation) | float32 | float32 | float32 |
| dtype (position_ids) | 不适用 | 不适用 | int64 |
| rank 约束 | input rank=2 | input rank>=1 | q/k rank=2, pos rank=1 |
| shape 推导 | output = input shape | output = input[:-1]+[w[0]] | output = input shape（双 pass-through） |
| runtime_checks | 1 条（hidden 一致） | 1 条（in_features 一致） | 2 条（seq_len 一致 × 2） |
| config 传递 | attrs（eps） | 无 | CpuRoPEParams 字段 |
| CPU kernel | elementwise | GEMM/GEMV | per-head 旋转 + cos/sin |
| Kernel 注册 | 2 个（Scalar + AVX2） | 1 个（Scalar） | 1 个（Scalar，第一版） |
| Workspace | 无 | 无 | 无 |

---

## 附录 B：RoPE 算法细节（Llama HF split-half 约定）

### B.1 inv_freq 计算

```
inv_freq[i] = 1.0 / (theta ^ (2i / head_dim))   for i in [0, head_dim/2)
```

### B.2 cos/sin 构建（split-half）

```
freqs = position * inv_freq              shape [head_dim/2]
emb = concat(freqs, freqs)               shape [head_dim]  (前后两半相同)
cos = cos(emb)                           shape [head_dim]
sin = sin(emb)                           shape [head_dim]
```

### B.3 rotate_half

```
rotate_half(x):
    x1 = x[:head_dim/2]
    x2 = x[head_dim/2:]
    return concat(-x2, x1)
```

### B.4 旋转应用

```
x_rope = x * cos + rotate_half(x) * sin
```

展开为逐元素：

```
x_rope[i]          = x[i] * cos[i] - x[i + half] * sin[i]           for i in [0, half)
x_rope[i + half]   = x[i + half] * cos[i + half] + x[i] * sin[i + half]
```

注：由于 `cos[i] == cos[i + half]`、`sin[i] == sin[i + half]`（emb 前后两半相同），可进一步简化，但第一版保持显式实现以匹配数学定义。
