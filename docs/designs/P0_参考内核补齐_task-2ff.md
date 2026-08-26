# P0 CPU 参考内核补齐实施计划

## 背景与范围决策

经调研确认的事实：
- 已注册 kernel 仅 4 类：kAdd / kElementwiseMul / kEmbedding / kRmsNorm（均为 fp32 scalar + 部分 AVX2）。
- [model_graph_builder.cpp](file:///home/richard/project/AetherMind/src/model/model_graph_builder.cpp) 构建的 Llama 图算子集：kEmbedding、kRmsNorm、kLinear、kRoPE、kKVCacheUpdate、kAttention、kAdd、kSiluMul、kArgmax（无 kMatMul/kSoftmax/kReshape/kPermute/kReorder 节点）。
- 默认 `opt_level=2` 的优化 pipeline 会启用 QkvLinear/GateUpLinear/SiluMul/AddRmsNorm 全部融合 pass（[optimize_graph.cpp](file:///home/richard/project/AetherMind/src/compiler/optimize_graph.cpp)），e2e 图将含 kQkvLinear、kGateUpLinear、kAddRmsNorm 节点。
- `LowerModelGraphOptions.enable_packed_weights` 默认 `false` → 默认 e2e 路径 selector 为 `WeightFormat::kPlain`。
- 现有 kernel 注册模式（rmsnorm）为：internal 头（Params + KernelArgs + 函数声明）→ scalar/avx2 实现 cpp → entry cpp（Validate + BuildParams + BuildMetadata + AM_REGISTER_KERNEL）。
- CMake 用 `GLOB_RECURSE CONFIGURE_DEPENDS` 自动收集 src 与 tests 源文件，新增 .cpp 无需改 CMakeLists（仅新增 AVX2 文件需在 src/CMakeLists.txt 登记编译 flags，P0 不做 SIMD）。

**范围决策**：
- dtype 仅 fp32（fp16/bf16 内核、量化 INT8/INT4 属后续方向）。
- 阶段 1 实现 unfused 最小集（opt_level=1 验证 e2e）；阶段 2 实现 fused 变体对齐默认 opt_level=2。
- kMatMul/kSoftmax/kSilu/kReshape/kPermute/kReorder 不在 llama 图路径，P0 不注册；softmax 作为 kAttention 内部实现。
- 每个 kernel 均注册 `WeightFormat::kPlain` 变体；kPacked 变体（cpu_identity 布局）作为阶段 2 末尾的可选增量，用于 `enable_packed_weights=true` 路径。

## 通用实现契约（每个 kernel 以此为准）

每个 kernel 目录 `src/backend/cpu/kernels/<op>/` 含三文件：

1. `<op>_internal.h`：`<Op>Params`（TensorView input/output 视图，仿 rmsnorm 的 `RmsNormParams`）、`<Op>Fp32KernelArgs`（原始指针 + 形状 + stride 的平面结构）、kernel 函数声明。
2. `<op>_fp32_scalar.cpp`：naive 参考实现（Float32 累积，任意 stride 访问，不做连续化假设），返回 `Status`。
3. `<op>_entry.cpp`：
   - `Validate<Op>Entry(ctx, args)`：dtype/rank/形状/stride 校验（含空张量分支，仿 rmsnorm 的 `empty_batch` 处理）；attrs 中参数反序列化校验。
   - `Build<Op>Params(inputs, outputs, params_buffer)`：arity 校验 + placement new。
   - `Build<Op>Metadata(op_params, attrs)`：OpParams → attrs 字节序列化（有参数的算子；无参算子返回 Ok）。
   - `AM_REGISTER_KERNEL` 注册：selector `{CPU, Float32, Float32, kPlain, kScalar, kBoth}`，`priority = 10`，`params_size = sizeof(<Op>Params)`。

测试（`tests/unit/backend/cpu/kernels/test_cpu_<op>_kernel.cpp`）三段式，仿 [test_cpu_rmsnorm_kernel.cpp](file:///home/richard/project/AetherMind/tests/unit/backend/cpu/kernels/test_cpu_rmsnorm_kernel.cpp)：
- 直接调用 kernel 函数的数值 golden 测试；
- `CpuBackend::PrepareKernel` + `resolved->fn(KernelContext{...})` 调用测试；
- `ExecutionPlanBuilder::Build(runtime, nodes)` + `RuntimeBindingContext` + `Executor::Execute` 全链路测试（仿 `ExecutionPlanBuilderRunsResolvedKernel`）。

## 阶段 1：unfused 最小集（增量按序执行，每增量独立可合并）

### 增量 1.1 kLinear（GEMM 参考实现）
语义：input `[..., in]`，weight `[out, in]`（行主序，`linear_op.cpp` 的 Infer 契约），output `[..., out]`，无 bias。`LinearParams` 为空，metadata 为空。
- 修改 [linear_internal.h](file:///home/richard/project/AetherMind/src/backend/cpu/kernels/linear/linear_internal.h)：保留 `LinearParams`，新增 `LinearFp32KernelArgs` 与 `LinearKernel_CPU_FP32_Scalar` 声明。
- 新增 `src/backend/cpu/kernels/linear/linear_fp32_scalar.cpp`：naive 三重循环 GEMM，支持任意 rank ≥ 2 输入（按行遍历 `[..., in]` 的尾两维）。
- 新增 `src/backend/cpu/kernels/linear/linear_entry.cpp`：注册 `cpu::linear_f32_scalar`。
- 新增 `tests/unit/backend/cpu/kernels/test_cpu_linear_kernel.cpp`：golden（小矩阵手算）、rank-3 批处理、PrepareKernel 路径、ExecutionPlanBuilder 全链路（含 weight 输入绑定）。

### 增量 1.2 kRoPE
语义：q/k 为 rank-2 `[seq, num_heads*head_dim]` / `[seq, num_kv_heads*head_dim]`，position_ids rank-1 Int64；输出形状与输入一致。旋转为 interleaved 成对旋转，freq = theta^(-2i/head_dim)；`scaling_type == kLinear` 时角度除以 `scaling_factor`。
- 新增 `src/backend/cpu/kernels/rope/rope_internal.h`、`rope_fp32_scalar.cpp`、`rope_entry.cpp`。
- metadata：`BuildRoPEMetadata` 将 `RoPEParams`（head_dim、num_attention_heads、num_key_value_heads、theta、scaling_type、scaling_factor）序列化进 attrs；entry 校验 position_ids 非负、seq 与 q/k 一致。
- 新增 `tests/unit/backend/cpu/kernels/test_cpu_rope_kernel.cpp`：与手算旋转 golden 对比；position 0 恒等（旋转角 0）；kLinear 缩放路径；非法 position_ids 拒绝。

### 增量 1.3 kKVCacheUpdate
语义：k/v rank-2 `[seq, kv_heads*head_dim]`，k_cache/v_cache rank-3 `[kv_heads, cache_len, head_dim]`（head 间可能 padding，用 stride 访问）。追加写入：`cache[head, cache_len - seq_len + t, d] = k[t, head*head_dim + d]`（cache_len 绑定为已使用长度，prefill 时 cache_len == seq_len 从 0 写）。输出为 cache 视图（state alias，kernel 原地写）。
- 新增 `src/backend/cpu/kernels/kvcache_update/kvcache_update_internal.h`、`kvcache_update_fp32_scalar.cpp`、`kvcache_update_entry.cpp`。
- 新增 `tests/unit/backend/cpu/kernels/test_cpu_kvcache_update_kernel.cpp`：prefill（0 起写）与 decode（追加）两路径；head 间 padding stride 布局。

### 增量 1.4 kAttention
语义：q rank-2 `[seq, num_heads*head_dim]`，k_cache/v_cache rank-3 `[kv_heads, cache_len, head_dim]`；GQA 分组（每组 `num_heads/num_kv_heads` 个 query 头共享一个 kv 头）；causal 掩码（query 位置 i 只能 attend cache 位置 ≤ cache_len - seq_len + i）；scale = 1/sqrt(head_dim)；内部两遍 naive softmax（max → exp/sum → 归一），Float32 累积。输出 `[seq, num_heads*head_dim]`。
- 新增 `src/backend/cpu/kernels/attention/attention_internal.h`、`attention_fp32_scalar.cpp`、`attention_entry.cpp`。
- metadata：`AttentionParams` 序列化。
- 新增 `tests/unit/backend/cpu/kernels/test_cpu_attention_kernel.cpp`：单头手算、GQA 分组、causal 掩码（未来位置为 -inf 效果）、padding stride cache、seq=1 decode 路径。

### 增量 1.5 kSiluMul
语义：gate/up rank-2 `[seq, intermediate]`，`output = silu(gate) * up`。`SiluMulParams` 为空。
- 新增 `src/backend/cpu/kernels/silu_mul/silu_mul_internal.h`、`silu_mul_fp32_scalar.cpp`、`silu_mul_entry.cpp`。
- 新增 `tests/unit/backend/cpu/kernels/test_cpu_silu_mul_kernel.cpp`。

### 增量 1.6 kArgmax
语义：logits rank-2 `[seq, vocab]`（axis=-1），输出 Int64 `[seq]`（去掉 axis 维，见 `argmax_op.cpp` Infer）；贪婪语义：并列取第一个最大值。
- 新增 `src/backend/cpu/kernels/argmax/argmax_internal.h`、`argmax_fp32_scalar.cpp`、`argmax_entry.cpp`。
- 新增 `tests/unit/backend/cpu/kernels/test_cpu_argmax_kernel.cpp`。

### 增量 1.7 端到端 smoke 测试
- 新增 `tests/unit/execution/test_llama_e2e.cpp`：用 `.models/tiny-random-LlamaForCausalLM`（hidden=16、heads=4、head_dim=4、layers=2、vocab=32000、fp32）走完整链路：`ModelLoader` 加载 → `ModelCompiler`（`opt_level=1`，仅 CF+DCE）→ `ExecutionPlanBuilder::BuildFromLoweredGraph` → `Executor::Execute` 逐层执行（embedding → 2×block → final norm → lm_head → argmax）。
- 断言：两次执行同输入输出一致（确定性）；argmax 输出 shape `[seq]` 且 dtype Int64；逐步执行无 resolve 失败（即所有 step 的 selector 均被新注册 kernel 覆盖）。
- 说明：该测试验证"链路贯通"，精确 logits 数值对拍（vs HF）留待后续 `LlamaDecode.*` 验收套件（阶段 2 末尾）。

## 阶段 2：fused 变体（对齐默认 opt_level=2）

### 增量 2.1 kAddRmsNorm
语义：`[input, residual, weight]` → `output = RmsNorm(input + residual, weight, eps)`、`new_residual = input + residual`。`AddRmsNormParams.eps` 入 attrs。
- 新增 `src/backend/cpu/kernels/add_rmsnorm/add_rmsnorm_internal.h`、`add_rmsnorm_fp32_scalar.cpp`、`add_rmsnorm_entry.cpp`；测试 `test_cpu_add_rmsnorm_kernel.cpp`（与 unfused kAdd+kRmsNorm 组合数值对拍）。

### 增量 2.2 kQkvLinear
语义：input `[seq, in]`、qkv_weight `[q_out + k_out + v_out, in]` → q/k/v 三个输出。`QkvLinearParams` 入 attrs（q_out/k_out/v_out 用于行切分）。
- 新增 `src/backend/cpu/kernels/qkv_linear/qkv_linear_internal.h`、`qkv_linear_fp32_scalar.cpp`、`qkv_linear_entry.cpp`；测试 `test_cpu_qkv_linear_kernel.cpp`（与 3×unfused kLinear 对拍）。

### 增量 2.3 kGateUpLinear
语义：input `[seq, in]`、gate_up_weight `[gate_out + up_out, in]` → gate/up 两个输出（行序 Gate 在前，见 `GateUpLinearParams` 注释）。
- 新增 `src/backend/cpu/kernels/gate_up_linear/gate_up_linear_internal.h`、`gate_up_linear_fp32_scalar.cpp`、`gate_up_linear_entry.cpp`；测试 `test_cpu_gate_up_linear_kernel.cpp`（与 2×unfused kLinear 对拍）。

### 增量 2.4 默认 pipeline e2e 与验收
- 扩展 `test_llama_e2e.cpp`：增加默认 `opt_level=2` 用例（走全部融合 pass），断言确定性 + 无 resolve 失败。
- 新增对拍用例：同一模型分别以 opt_level=1 与 opt_level=2 编译执行，各 step 输出的中间激活值逐张量近似相等（融合不改变数值语义的回归守卫）。

### 增量 2.5（可选）weight 算子 kPacked 变体
- 为 kLinear/kQkvLinear/kGateUpLinear 补注册 `WeightFormat::kPacked` 变体：权重从 `ctx.packed_weights`（`PackedWeights` storage buffer，cpu_identity 布局 = 逻辑行主序拷贝）读取，`expected_packing_recipe = {layout: "cpu_identity", alignment: 64}` 由 `CpuWeightPrepacker::RecipeFor` 自动填充（见 [cpu_backend.cpp](file:///home/richard/project/AetherMind/src/backend/cpu/cpu_backend.cpp) `PrepareKernel` 对 kPacked 的处理）。
- 新增用例：`enable_packed_weights=true` 的 e2e 走查（与 kPlain 路径输出对拍）。

## 验证计划（每增量完成后）

```bash
# 构建与单测（增量级：先跑新 kernel 测试套件）
cmake --build build --target aethermind_unit_tests -j
./build/tests/unit/aethermind_unit_tests --gtest_filter=CPUKernelLinear.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=CPUKernelRoPE.*
# ... 各 kernel 套件

# 阶段 2 末尾全量回归 + e2e
./build/tests/unit/aethermind_unit_tests
./build/tests/unit/aethermind_unit_tests --gtest_filter=LlamaE2E.*

# 格式（改动文件）
clang-format -i <改动文件列表>

# 阶段 2 末尾：TSAN 验证（kernel 无共享状态，主要防注册静态初始化问题）
cmake -S . -B build-tsan -DENABLE_TSAN=ON -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF
cmake --build build-tsan --target aethermind_unit_tests -j
```

## 假设与风险

- 假设 1：KVCacheUpdate 追加语义为 `offset = cache_len - seq_len`（cache_len 绑定为已使用长度）；与 kv_cache_design.md 的"按 token position 单调递增写入"一致，若 execution 侧另有约定需在增量 1.3 前确认。
- 假设 2：Attention 为因果掩码（decoder-only）；q 位置 i 的可见范围为 cache 前缀 `cache_len - seq_len + i`。
- 假设 3：P0 全部 kernel 先注册 kPlain 变体（默认 `enable_packed_weights=false` 路径），kPacked 变体在阶段 2.5 增量补齐。
- 风险 1：形状/stride 契约与 Infer 漂移 → 每个 kernel 的 Validate 与 Infer 契约测试互相对拍。
- 风险 2：state alias 绑定错误（KVCacheUpdate/Attention 的 cache 输入输出）→ 全链路测试中显式绑定同一 buffer 验证原地语义。
- 风险 3：默认 pipeline 融合 pass 改变 e2e 图结构 → 阶段 2.4 的对拍用例作为回归守卫。
- 风险 4：新增文件经 GLOB 自动收集，构建时 CMake 会自动重扫描（CONFIGURE_DEPENDS），无需手动改 CMakeLists。
