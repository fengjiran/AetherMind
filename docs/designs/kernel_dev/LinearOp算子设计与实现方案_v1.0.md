# LinearOp 算子设计与实现方案 v1.0

> **文档定位**：Operator 语义层第二个算子（LinearOp）的详细设计与实现方案
> **前置文档**：`Operator语义层接口实施步骤_v1.0.md` Section 18、`RMSNorm算子契约.md`
> **样板参照**：`RmsNormOp`（已完成审核与测试覆盖）
> **状态**：设计方案（未实现）

---

## 1. 功能目标

**语义**：计算 `output = input @ weight.T`（可选 `+ bias`），即线性变换层。

**在 Llama 推理中的角色**：每层 DecoderLayer 包含 7 个 Linear 投影（q/k/v/o_proj + gate/up/down_proj）+ 顶层 lm_head，是推理最大热点。

**第一版范围**（遵循设计文档 Section 18）：

| 维度 | 第一版 | 后续扩展 |
|------|--------|---------|
| dtype | float32 | BF16/FP16 |
| layout | contiguous | 非 contiguous |
| bias | 不实现 | 可选 bias |
| packing | 不启用（仅 `kPlain`） | `kPacked` selector |
| phase | `kBoth`（reference） | `kPrefill`/`kDecode` 分离 |
| isa | `kScalar`（reference） | `kAVX2`/`kAVX512`/`kAMX` |

---

## 2. 文件结构规划

参照 RmsNormOp 样板的目录结构，新增 6 个文件：

```
include/aethermind/operators/
  └── linear_op.h                    # LinearOp 类声明

src/operators/
  └── linear_op.cpp                  # LinearOp 实现 + AM_REGISTER_OPERATOR

include/aethermind/backend/cpu/kernels/
  └── cpu_linear_kernel.h            # 公有 Launch 接口 + CpuLinearParams

src/backend/cpu/kernels/linear/
  ├── linear_internal.h             # LinearFp32KernelArgs + 内部声明
  ├── linear_entry.cpp               # KernelContext entry + AM_REGISTER_KERNEL
  └── linear_fp32_scalar.cpp         # Reference GEMM/GEMV 实现

tests/unit/operators/
  └── test_linear_op.cpp             # 单元测试（Validate/Infer/Prepare/Run）
```

---

## 3. 类结构设计

### 3.1 linear_op.h

```cpp
#ifndef AETHERMIND_OPERATORS_LINEAR_OP_H
#define AETHERMIND_OPERATORS_LINEAR_OP_H

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator.h"

namespace aethermind {

/// Semantic operator for linear transformation: output = input @ weight.T
///
/// Weight shape convention: [out_features, in_features] (PyTorch/HF row-major).
/// Input shape: [..., in_features] (rank >= 1).
/// Output shape: [..., out_features] (same rank as input).
class LinearOp final : public Operator {
public:
    using Params = LinearParams;

    explicit LinearOp(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kLinear;
    }

    AM_NODISCARD const char* Name() const noexcept override {
        return "Linear";
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

- 与 RmsNormOp 结构完全对称，降低认知负担
- `LinearParams` 当前为空，`ValidateParams()` 直接返回 Ok
- `ComputeWorkspaceRequirement` 返回空（Scalar kernel 原地计算，不需要 scratch）
- 后续 AVX2/AMX blocked kernel 可能需要 workspace（tiled buffer），届时再覆写

---

## 4. 核心方法定义与实现逻辑

### 4.1 ValidateParams

```cpp
Status LinearOp::ValidateParams() const {
    // LinearParams is empty in Phase 1; nothing to validate.
    return Status::Ok();
}
```

**逻辑**：`LinearParams` 当前无字段，直接返回 Ok。后续如增加 bias/alpha 等参数再扩展。

---

### 4.2 CheckInputSpecs

实现设计文档 Section 18.3 的 7 项检查：

```cpp
Status LinearOp::CheckInputSpecs(std::span<const TensorSpec> inputs) const {
    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                "Linear expects exactly 2 inputs, got " + std::to_string(inputs.size()));
    }

    const auto& input_spec = inputs[0];
    const auto& weight_spec = inputs[1];

    // Check 5: dtype must be float32
    if (input_spec.dtype != DataType::Float32() || weight_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("Linear only supports float32 in Phase 1");
    }

    // Check 1: input rank >= 1
    if (!HasRank(input_spec.shape, 1)) {
        return Status::InvalidArgument("Linear input must have rank >= 1");
    }

    // Check 2: weight rank == 2
    if (!HasRank(weight_spec.shape, 2)) {
        return Status::InvalidArgument("Linear weight must be rank-2 [out_features, in_features]");
    }

    // Check 3: input.shape[-1] == weight.shape[1] (in_features)
    const ShapeSymbol& in_features = input_spec.shape[input_spec.shape.rank() - 1];
    if (!IsPositiveIfStatic(in_features)) {
        return Status::InvalidArgument("Linear input last dimension must be positive");
    }
    if (!UnifyShapeSymbol(in_features, weight_spec.shape[1]).ok()) {
        return Status::InvalidArgument(
                "Linear input last dimension must equal weight in_features");
    }

    // Check 4 (partial): output shape derived in InferOutputShapes
    // Check 6: contiguity is a runtime check (deferred to kernel)
    // Check 7: bias not implemented in Phase 1

    return Status::Ok();
}
```

**验证项映射**（设计文档 7 项 → 实现）：

| 设计文档检查项 | 实现位置 | 说明 |
|---------------|---------|------|
| 1. input rank >= 1 | CheckInputSpecs | `HasRank(input, 1)` |
| 2. weight rank == 2 | CheckInputSpecs | `HasRank(weight, 2)` |
| 3. input.shape[-1] == weight.shape[1] | CheckInputSpecs | `UnifyShapeSymbol` |
| 4. output shape | InferOutputShapes | 推导而非校验 |
| 5. dtype float32 | CheckInputSpecs | 直接比较 |
| 6. contiguous | Run/kernel | 运行时检查 |
| 7. bias | 暂不实现 | Phase 1 无 bias |

---

### 4.3 InferOutputShapes

```cpp
StatusOr<InferenceResult> LinearOp::InferOutputShapes(std::span<const TensorSpec> inputs) const {
    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                "Linear expects exactly 2 shape inputs, got " + std::to_string(inputs.size()));
    }

    const auto& input_spec = inputs[0];
    const auto& weight_spec = inputs[1];

    if (!HasRank(input_spec.shape, 1)) {
        return Status::InvalidArgument("Linear input shape must have rank >= 1");
    }
    if (!HasRank(weight_spec.shape, 2)) {
        return Status::InvalidArgument("Linear weight shape must be rank-2");
    }

    // output = input[:-1] + [weight[0]]
    // Reuse input's leading dims, replace last dim with weight's out_features.
    const size_t input_rank = input_spec.shape.rank();
    std::vector<ShapeSymbol> output_dims;
    output_dims.reserve(input_rank);
    for (size_t i = 0; i < input_rank - 1; ++i) {
        output_dims.push_back(input_spec.shape[i]);
    }
    output_dims.push_back(weight_spec.shape[0]);  // out_features

    TensorSpec output_spec{
            .dtype = DataType::Float32(),
            .shape = SymbolicShape(IntArrayView{output_dims}),
    };

    InferenceResult result;
    result.outputs.push_back(std::move(output_spec));

    // Emit runtime check: input.shape[-1] == weight.shape[1] if symbolic.
    const ShapeSymbol& in_features = input_spec.shape[input_rank - 1];
    const ShapeSymbol& weight_in = weight_spec.shape[1];
    if (!IsStaticAndEqual(in_features, weight_in)) {
        result.runtime_checks.push_back(MakeDimEqualConstraint(
                /*lhs*/ TensorPortType::kInput, 0, input_rank - 1,
                /*rhs*/ TensorPortType::kInput, 1, 1));
    }

    return result;
}
```

**shape 推导逻辑**：

- `output.shape = input.shape[:-1] + [weight.shape[0]]`
- 即：保持 input 的前 N-1 维，最后一维替换为 `out_features`
- 示例：input `[4, 4096]` + weight `[4096, 4096]` → output `[4, 4096]`

**runtime_checks**：

- 当 `in_features` 和 `weight_in` 中有 symbolic dim 时，生成 `DimEqualConstraint`
- 参考 RmsNormOp 的 `EmitsRuntimeCheckForDistinctSymbolicHiddenDimension` 模式

---

### 4.4 Prepare

```cpp
Status LinearOp::Prepare(OperatorContext& ctx) {
    if (ctx.backend == nullptr) {
        return Status::InvalidArgument("Linear Prepare requires OperatorContext.backend");
    }

    const auto resolved = ctx.backend->ResolveKernelInfo(
            OpType::kLinear,
            ctx.selector);

    if (!resolved.ok()) {
        return resolved.status();
    }

    resolved_kernel_ = resolved.value();
    if (resolved_kernel_.fn == nullptr) {
        return Status::Internal("Linear Prepare resolved a kernel with null fn");
    }
    // LinearParams is empty; no attrs to write.
    return Status::Ok();
}
```

**与 RmsNormOp 的差异**：

- RmsNormOp 在 Prepare 中将 `eps`（4 字节 float）写入 `resolved_kernel_.attrs`
- LinearOp 的 `LinearParams` 为空，**不覆写 attrs**（与 EmbeddingOp 一致）

---

### 4.5 Run

```cpp
Status LinearOp::Run(KernelContext& ctx,
                     const RuntimeBindingContext& bindings,
                     size_t step_index) const noexcept {
    if (resolved_kernel_.fn == nullptr) {
        return Status::FailedPrecondition("Linear Run called before Prepare");
    }

    const auto binding = bindings.GetStepTensorBinding(step_index);
    if (!binding.ok()) {
        return binding.status();
    }

    const auto* b = binding.value();
    if (b->inputs.size() != 2) {
        return Status::InvalidArgument(
                "Linear requires 2 input tensor bindings, got " +
                std::to_string(b->inputs.size()));
    }
    if (b->outputs.size() != 1) {
        return Status::InvalidArgument(
                "Linear requires 1 output tensor binding, got " +
                std::to_string(b->outputs.size()));
    }

    // Phase 1 CPU-first: construct CPU-specific params directly. Phase 2 should
    // inject params construction via Backend to support multiple backends.
    cpu::CpuLinearParams params{
            .input_tensor = b->inputs[0],
            .weight_tensor = b->inputs[1],
            .output_tensor = b->outputs[0],
    };
    ctx.kernel_params = &params;
    return resolved_kernel_.fn(ctx);
}
```

**与 RmsNormOp 的差异**：

- 构造 `cpu::CpuLinearParams`（无 bias 字段，Phase 1 不实现）
- 错误消息用 "Linear" 前缀

---

## 5. CPU Kernel 实现方案

### 5.1 cpu_linear_kernel.h（公有接口）

```cpp
#ifndef AETHERMIND_BACKEND_CPU_KERNELS_CPU_LINEAR_KERNEL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_CPU_LINEAR_KERNEL_H

#include "aethermind/base/tensor_view.h"

namespace aethermind::cpu {

/// Per-call kernel params for CPU Linear kernel.
/// Lifetime: stack-bound during LinearOp::Run, valid for the duration of fn(ctx).
struct CpuLinearParams {
    TensorView input_tensor{};
    TensorView weight_tensor{};
    MutableTensorView output_tensor{};
};

}// namespace aethermind::cpu

#endif
```

---

### 5.2 linear_internal.h（内部结构）

```cpp
#ifndef AETHERMIND_BACKEND_CPU_KERNELS_LINEAR_LINEAR_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_LINEAR_LINEAR_INTERNAL_H

#include <cstdint>

namespace aethermind::cpu {

/// Stripped-down kernel args extracted from TensorView for raw compute loops.
struct LinearFp32KernelArgs {
    const float* input{};
    const float* weight{};
    float* output{};
    int64_t m{};       // batch dimension (seq_len or 1)
    int64_t n{};       // out_features
    int64_t k{};       // in_features
    int64_t lda{};     // input stride: k (row-major)
    int64_t ldb{};     // weight stride: in_features (row-major [out, in])
    int64_t ldc{};     // output stride: n (row-major)
};

/// Reference scalar GEMM/GEMV: output[m, n] = sum_k(input[m, k] * weight[n, k])
/// Weight is [out, in] row-major, so weight[n, k] = weight[n * ldb + k].
/// This computes input @ weight.T (not input @ weight).
Status LinearKernel_CPU_FP32_Scalar(const LinearFp32KernelArgs& args) noexcept;

}// namespace aethermind::cpu

#endif
```

**关键设计**：

- `weight` shape `[out, in]`，`ldb = in_features`
- 计算语义：`output[m, n] = Σ_k input[m, k] * weight[n, k]`
- 即 `output = input @ weight.T`（weight 不转置，直接按行访问）

---

### 5.3 linear_fp32_scalar.cpp（Reference 实现）

```cpp
#include "linear_internal.h"

namespace aethermind::cpu {

Status LinearKernel_CPU_FP32_Scalar(const LinearFp32KernelArgs& a) noexcept {
    // Naive triple-loop: O(m * n * k)
    // output[m, n] = sum_k(input[m, k] * weight[n, k])
    for (int64_t m_idx = 0; m_idx < a.m; ++m_idx) {
        const float* in_row = a.input + m_idx * a.lda;
        float* out_row = a.output + m_idx * a.ldc;

        for (int64_t n_idx = 0; n_idx < a.n; ++n_idx) {
            const float* w_row = a.weight + n_idx * a.ldb;
            float acc = 0.0f;

            // Inner product of input_row and weight_row over k dimension.
            for (int64_t k_idx = 0; k_idx < a.k; ++k_idx) {
                acc += in_row[k_idx] * w_row[k_idx];
            }

            out_row[n_idx] = acc;
        }
    }
    return Status::Ok();
}

}// namespace aethermind::cpu
```

**性能特征**：

- Naive 三重循环，无向量化、无分块、无线程并行
- 适合作为 reference 验证正确性
- Decode 阶段（M=1）退化为 GEMV，k 循环是热点
- Prefill 阶段（M>>1）是 GEMM，后续应替换为 blocked/tiled 实现

---

### 5.4 linear_entry.cpp（Kernel 注册入口）

```cpp
#include "aethermind/backend/cpu/kernels/cpu_linear_kernel.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "linear_internal.h"

#include <cstring>

namespace aethermind {

namespace {

// Extract LinearFp32KernelArgs from KernelContext (set by LinearOp::Run).
LinearFp32KernelArgs ExtractArgs(const KernelContext& ctx) noexcept {
    const auto& p = *static_cast<const cpu::CpuLinearParams*>(ctx.kernel_params);
    const auto& in = p.input_tensor;
    const auto& wt = p.weight_tensor;
    const auto& out = p.output_tensor;

    const int64_t m = (in.rank() >= 2) ? in.shape()[0] : 1;
    const int64_t k = in.shape()[in.rank() - 1];
    const int64_t n = wt.shape()[0];  // out_features

    return LinearFp32KernelArgs{
            .input = static_cast<const float*>(in.data()),
            .weight = static_cast<const float*>(wt.data()),
            .output = static_cast<float*>(out.data()),
            .m = m,
            .n = n,
            .k = k,
            .lda = k,
            .ldb = k,   // weight [out, in], contiguous → ldb = in_features = k
            .ldc = n,
    };
}

Status CpuLinearKernelEntry_FP32_Scalar(const KernelContext& ctx) noexcept {
    return cpu::LinearKernel_CPU_FP32_Scalar(ExtractArgs(ctx));
}

}// namespace

// Phase 1: register one reference kernel for kPlain + kBoth + Scalar.
// Future: add AVX2/Prefill/Decode/Packed variants.
AM_REGISTER_KERNEL(OpType::kLinear,
                   KernelSelector{
                           .device_type = DeviceType::kCPU,
                           .act_dtype = DataType::Float32(),
                           .weight_dtype = DataType::Float32(),
                           .weight_format = WeightFormat::kPlain,
                           .isa = IsaLevel::kScalar,
                           .phase = ExecPhase::kBoth,
                   },
                   &CpuLinearKernelEntry_FP32_Scalar);

}// namespace aethermind
```

**注册策略**：

- 第一版只注册 1 个 kernel：`kPlain + kBoth + kScalar`
- `kBoth` 意味着 prefill 和 decode 都用同一个 reference kernel（naive 实现）
- 后续扩展优先级：
  1. `kScalar + kDecode`（GEMV 优化，复用 `DotProductAvx2Unroll`）
  2. `kAVX2 + kPrefill`（blocked GEMM）
  3. `kPacked` selector 系列（消费 WeightPrepacker 输出）

---

## 6. 参数说明

### 6.1 Operator 层参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `LinearParams` | struct | `{}`（空） | Phase 1 无字段；后续可扩展 bias/alpha |

### 6.2 Kernel 层参数

| 参数 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `input_tensor` | TensorView | binding.inputs[0] | 激活输入，shape `[..., in_features]` |
| `weight_tensor` | TensorView | binding.inputs[1] | 权重，shape `[out_features, in_features]` |
| `output_tensor` | MutableTensorView | binding.outputs[0] | 输出，shape `[..., out_features]` |

### 6.3 KernelSelector 配置

| 字段 | 第一版取值 | 后续扩展 |
|------|-----------|---------|
| `device_type` | `kCPU` | GPU |
| `act_dtype` | `Float32` | BF16/FP16 |
| `weight_dtype` | `Float32` | Int8/Int4 |
| `weight_format` | `kPlain` | `kPacked` |
| `isa` | `kScalar` | `kAVX2`/`kAVX512`/`kAMX` |
| `phase` | `kBoth` | `kPrefill`/`kDecode` 分离 |

---

## 7. 数据处理流程

### 7.1 构建期（Graph Construction → Plan Building）

```
ModelGraphBuilder::AddWeightedNode(kLinear)
    ↓
operator_schema.cpp: kLinear schema = {input(Activation), weight(Weight)} → {output}
    ↓
ExecutionPlanBuilder::Build()
    ├── op.CheckInputSpecs(inputs)        → 7 项校验
    ├── op.InferOutputShapes(inputs)      → output shape 推导 + runtime_checks
    ├── op.ComputeWorkspaceRequirement()  → 空（Phase 1）
    └── op.Prepare(op_ctx)                → ResolveKernelInfo(kLinear, selector)
```

### 7.2 执行期（Runtime Execution）

```
Executor::Execute(plan, bindings)
    ↓
for each step:
    LinearOp::Run(kernel_ctx, bindings, step_index)
        ├── 检查 resolved_kernel_.fn != nullptr
        ├── bindings.GetStepTensorBinding(step_index)
        ├── 校验 inputs.size()==2, outputs.size()==1
        ├── 构造 CpuLinearParams{input, weight, output}
        ├── ctx.kernel_params = &params
        └── resolved_kernel_.fn(ctx)
                ↓
            CpuLinearKernelEntry_FP32_Scalar(ctx)
                ├── ExtractArgs(ctx) → LinearFp32KernelArgs{m, n, k, ...}
                └── LinearKernel_CPU_FP32_Scalar(args)
                        └── triple-loop: output[m,n] = Σ_k input[m,k] * weight[n,k]
```

### 7.3 Shape 推导示例

| 场景 | input shape | weight shape | output shape | m | n | k | 形态 |
|------|-------------|--------------|-------------|---|---|---|------|
| Prefill q_proj | `[128, 4096]` | `[4096, 4096]` | `[128, 4096]` | 128 | 4096 | 4096 | GEMM |
| Decode q_proj | `[1, 4096]` | `[4096, 4096]` | `[1, 4096]` | 1 | 4096 | 4096 | GEMV |
| Decode gate_proj | `[1, 4096]` | `[11008, 4096]` | `[1, 11008]` | 1 | 11008 | 4096 | GEMV |
| lm_head | `[1, 4096]` | `[32000, 4096]` | `[1, 32000]` | 1 | 32000 | 4096 | GEMV |

---

## 8. 边界情况处理策略

| 边界情况 | 处理策略 | 实现位置 |
|---------|---------|---------|
| input rank=1（单 token） | 支持：`m=1`，output rank=1 | InferOutputShapes + ExtractArgs |
| input rank>2 | 支持：仅取最后一维作为 k，其余展平为 m | 需在 ExtractArgs 中正确计算 m（Phase 1 可仅支持 rank ≤ 2） |
| weight shape 不匹配 | CheckInputSpecs 返回 InvalidArgument | `UnifyShapeSymbol` 校验 |
| 动态 shape（symbolic） | InferOutputShapes 生成 DimEqualConstraint | runtime_checks |
| fn 为 null（backend 返回空） | Prepare 返回 Internal | issue 3 修复模式 |
| 未 Prepare 直接 Run | Run 返回 FailedPrecondition | Run 前置检查 |
| 输入/输出数量错误 | Run 返回 InvalidArgument | Run 前置检查 |
| 非 contiguous 数据 | Phase 1 假定 contiguous，不校验（kernel 直接按 row-major 访问） | 后续 kernel 可加校验 |
| m=0 或 n=0 或 k=0 | kernel 循环不执行，输出为空 | triple-loop 自然处理 |

---

## 9. 使用示例

### 9.1 图构建（ModelGraphBuilder 已有，无需新增代码）

```cpp
// src/model/graph/model_graph_builder.cpp 中已存在：
AddWeightedNode(graph, OpType::kLinear, LinearParams{},
                /*input_port*/ hidden_state,
                /*weight*/ q_proj_weight,
                /*output_name*/ "q_proj_output");
```

### 9.2 单元测试（参照 RmsNormOp 测试模式）

```cpp
// tests/unit/operators/test_linear_op.cpp

// 1. Validate + CheckInputSpecs
TEST(LinearOp, ValidatesInputContract) {
    LinearOp op{LinearOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 8})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({16, 8})},
    };
    EXPECT_TRUE(op.CheckInputSpecs(inputs).ok());
}

// 2. InferOutputShapes
TEST(LinearOp, InfersOutputShapeFromWeight) {
    LinearOp op{LinearOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 4096})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({11008, 4096})},
    };
    auto inference = op.InferOutputShapes(inputs);
    ASSERT_TRUE(inference.ok());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 4);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 11008);
}

// 3. Prepare + Run（FakeBackend + StubKernel，参照 test_rmsnorm_op.cpp 模式）
TEST(LinearOp, RunInvokesKernelAndReturnsOk) {
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel(OpType::kLinear, &StubLinearKernel);
    LinearOp op{LinearOp::Params{}};
    OperatorContext op_ctx{.backend = &backend};
    ASSERT_TRUE(op.Prepare(op_ctx).ok());

    LinearBindingBuilder builder;  // input[4,8] + weight[16,8] → output[4,16]
    RuntimeBindingContext bindings;
    bindings.SetStepTensorBinding(0, builder.Build());

    KernelContext kernel_ctx;
    ASSERT_TRUE(op.Run(kernel_ctx, bindings, 0).ok());
    EXPECT_TRUE(g_stub_state.called);
}
```

### 9.3 端到端推理（未来 Executor 集成）

```cpp
// Executor 自动调度，无需手动调用 LinearOp
RuntimeBindingContext bindings;
bindings.SetStepTensorBinding(step_idx, StepTensorBinding{
        .inputs = {input_view, weight_view},
        .outputs = {output_view},
});
Executor::Execute(plan, bindings);
// LinearOp::Run 被自动调用
```

---

## 10. 实施步骤

| 步骤 | 文件 | 内容 | 依赖 |
|------|------|------|------|
| 1 | `include/aethermind/backend/cpu/kernels/cpu_linear_kernel.h` | `CpuLinearParams` 结构 | 无 |
| 2 | `src/backend/cpu/kernels/linear/linear_internal.h` | `LinearFp32KernelArgs` + 声明 | 步骤 1 |
| 3 | `src/backend/cpu/kernels/linear/linear_fp32_scalar.cpp` | Reference triple-loop | 步骤 2 |
| 4 | `src/backend/cpu/kernels/linear/linear_entry.cpp` | Entry + `AM_REGISTER_KERNEL` | 步骤 1,3 |
| 5 | `include/aethermind/operators/linear_op.h` | `LinearOp` 类声明 | 无 |
| 6 | `src/operators/linear_op.cpp` | 实现 + `AM_REGISTER_OPERATOR` | 步骤 1,5 |
| 7 | `tests/unit/operators/test_linear_op.cpp` | 单元测试（Validate/Infer/Prepare/Run） | 步骤 5,6 |
| 8 | `tests/unit/backend/cpu/test_cpu_resolve_kernel.cpp` | 更新 `MissingKeyReturnsNullptr` 测试 | 步骤 4 |

---

## 11. 风险与注意事项

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| **回归测试** | `test_cpu_resolve_kernel.cpp:73` 断言 `kLinear` 返回 nullptr | 步骤 8 中将断言改为 `EXPECT_NE(..., nullptr)` |
| **WeightPrepackPlanner 前置** | prepack planner 已为每个 Linear 权重创建 `kPacked` 请求 | 第一版只注册 `kPlain` kernel；`kPacked` 请求会被 prepacker 做 memcpy fallback |
| **Prefill 性能** | Naive GEMM 对大 seq_len 极慢 | 第一版仅验证正确性；后续优先实现 `kDecode` GEMV 优化 |
| **ExtractArgs rank>2** | 当前 ExtractArgs 仅处理 rank ≤ 2 | Phase 1 Llama 仅用 rank-2；rank>2 需扩展 m 计算（展平 leading dims） |
| **Contiguity 假定** | kernel 直接按 row-major 访问 | Phase 1 假定 contiguous；后续可在 Run 中加 `is_contiguous()` 校验 |
| **无 bias** | Llama 部分投影有 bias（如 QKV bias） | Phase 1 不实现；后续可通过扩展 schema 为 3 输入或 attrs 传递 |

---

## 12. 方案总结

LinearOp 实现完全复用 RmsNormOp 样板模式，核心差异在于：

1. **shape 推导**：从"保持 input shape"变为"替换最后一维为 weight[0]"
2. **CPU kernel**：从 elementwise 变为 GEMM/GEMV（triple-loop）
3. **attrs**：不覆写（LinearParams 为空）

总计新增 6 个文件 + 修改 1 个测试文件，预计实现量与 RmsNormOp 相当。

---

## 附录 A：与 RmsNormOp 样板对比

| 方面 | RmsNormOp | LinearOp |
|------|-----------|----------|
| 语义 | 归一化 | 线性变换 |
| 输入数量 | 2（input + weight） | 2（input + weight） |
| 输出数量 | 1 | 1 |
| Params | `RmsNormParams{eps}` | `LinearParams{}`（空） |
| dtype | float32 | float32 |
| rank 约束 | input rank=2, weight rank=1 | input rank>=1, weight rank=2 |
| shape 推导 | output = input shape | output = input[:-1] + [weight[0]] |
| attrs 覆写 | 是（4 字节 eps） | 否 |
| CPU kernel | elementwise（乘方+求和+归一化） | GEMM/GEMV（triple-loop） |
| Kernel 注册 | 2 个（Scalar + AVX2） | 1 个（Scalar，第一版） |
| Workspace | 无 | 无 |
