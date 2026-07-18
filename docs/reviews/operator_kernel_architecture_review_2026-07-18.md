# AetherMind 算子 Kernel 架构系统性审核报告

> 审核日期：2026-07-18  
> 审核对象：当前工作树中的 Operator、Kernel、Dispatch、Execution、Workspace 及相关测试和构建配置  
> 项目阶段：Phase 1，桌面/服务器 CPU 本地推理运行时  
> 审核方式：源码与设计文档交叉审查、调用链分析、单元测试、TSAN/ASAN 和聚焦 benchmark 验证  
> 审核性质：只读审核，未修改产品代码

## 1. 总体结论

**审核结论：不通过，当前架构尚未满足 Phase 1 交付要求。**

整体分层方向合理，Operator、Kernel Registry、ExecutionPlan、Runtime Binding 和 ISA 实现之间已有较清晰的边界。但当前存在一个已被 ASAN 证实的内存生命周期缺陷，以及 ISA、workspace、并行计算和完整 Llama 执行链尚未闭环的问题。

从架构成熟度看，当前实现适合作为继续开发的基础框架，但不能视为满足生产级 Phase 1 技术需求和性能指标的完成状态。

## 2. 阻断问题

### 2.1 P0：Kernel 参数发生真实的栈生命周期越界

位置：

- `include/aethermind/operators/operator.h:170`
- `include/aethermind/operators/operator.h:174`

`Operator::InvokeResolvedKernel()` 在 `if` 作用域内创建参数缓冲区，并将其地址赋给 `ctx.kernel_params`：

```cpp
if (resolved.params_builder != nullptr) {
    alignas(std::max_align_t) std::byte buffer[kMaxKernelParamsSize];
    AM_RETURN_IF_ERROR(resolved.params_builder(inputs, outputs, buffer));
    ctx.kernel_params = buffer;
}
return resolved.fn(ctx);
```

`buffer` 的生命周期在离开 `if` 作用域时结束，而 kernel 在作用域结束后才被调用，因此 `ctx.kernel_params` 已成为悬空指针。

普通构建测试没有发现该问题，但聚焦 ASAN 用例稳定报告：

```text
AddressSanitizer: stack-use-after-scope
Operator::InvokeResolvedKernel -> EmbeddingKernel -> TensorView::is_valid
```

该缺陷影响所有使用 `KernelParamsBuilder` 的生产执行路径，包括 Add、Embedding、RMSNorm 和 ElementwiseMul。由于它属于未定义行为并直接违反 ASAN/LSAN 硬门禁，应作为最高优先级修复项。

### 2.2 P0：`LaunchRmsNorm` 无条件执行 AVX2

位置：`src/backend/cpu/kernels/rmsnorm/rmsnorm_api.cpp:27`

公开 SDK API 直接调用：

```cpp
return cpu::detail::RmsNormKernel_CPU_FP32_AVX2(kernel_args);
```

当前路径存在以下问题：

- 没有运行时 CPU capability 检测；
- 没有 scalar fallback；
- 绕过 `ValidateRmsNormEntry()` 的尺寸、步长和 epsilon 校验；
- 非 AVX2/FMA CPU 可能触发非法指令；
- 无效尺寸或步长可能导致越界访问。

与此同时，`include/aethermind/backend/cpu/cpu_capabilities.h` 当前只声明基础 CPU backend 信息，没有实际 ISA capability 数据或探测逻辑。

### 2.3 P1：Workspace 合同没有闭环

涉及位置：

- `include/aethermind/operators/operator.h:104`
- `src/execution/execution_plan_builder.cpp:113`
- `include/aethermind/runtime/workspace.h:191`

`Operator::ComputeWorkspaceRequirement()` 当前没有调用者。执行计划构建器直接采用调用方预填的 `ExecutionPlanNodeSpec::workspace_requirement`，没有根据已创建的 Operator 计算 workspace。

另外，`WorkspaceLifetime` 和 `reusable` 虽已进入数据模型，但当前规划算法仍然顺序累加每个 requirement，没有根据生命周期进行空间复用。

后果包括：

- 未来需要 scratch 的算子可能得到零字节 workspace；
- workspace 声明与实际执行需求可能漂移；
- 顺序执行算子的临时空间无法复用，增加内存占用；
- 难以验证 Decode 稳态零分配合同。

### 2.4 P1：优化 Kernel 默认不会被自动选择

位置：`include/aethermind/model/graph/compilation/graph_lowering.h:19`

`GraphLoweringConfig` 默认 ISA 为 `IsaLevel::kScalar`，当前没有从 CPU capability 推导最大可用 ISA 的逻辑。因此 RMSNorm AVX2 虽然已注册，默认 graph lowering 仍然请求 scalar kernel。

当前 selector 和 priority 机制具备 ISA fallback 能力，但缺少 capability 到 selector 的上游决策环节。

### 2.5 P1：OpenMP 源码与构建配置脱节

涉及位置：

- `src/backend/cpu/kernels/rmsnorm/rmsnorm_fp32_scalar.cpp:49`
- `src/backend/cpu/kernels/rmsnorm/rmsnorm_fp32_avx2.cpp:111`
- `src/CMakeLists.txt:15`

RMSNorm scalar 和 AVX2 实现中使用了 `#pragma omp parallel for`，但默认 CMake 配置没有：

- `find_package(OpenMP)`；
- 链接 `OpenMP::OpenMP_CXX`；
- OpenMP 编译选项；
- RuntimeContext 统一线程数配置；
- 嵌套并行和 oversubscription 控制。

因此默认构建中的 OpenMP pragma 不会形成实际并行加速，当前架构也无法稳定控制线程资源。

### 2.6 P1：Raw-kernel fallback 无法传递 Kernel 参数

涉及位置：

- `src/execution/execution_plan_builder.cpp:143`
- `include/aethermind/operators/function_operator.h:54`

当某个 OpType 没有注册语义 Operator 时，ExecutionPlanBuilder 会将解析出的 kernel 包装为 `FunctionOperator`。但该构造过程只保留 `fn`、`attrs` 和 `debug_name`，丢弃 `params_builder` 和 `params_size`。

`FunctionOperator::Run()` 又忽略 `RuntimeBindingContext` 和 `step_index`，直接调用 kernel。此时 `KernelContext::kernel_params` 默认为空，因此绝大多数需要输入输出 TensorView 的 kernel 无法通过该 fallback 正常执行。

建议删除该隐式 fallback，或者让 `FunctionOperator` 完整持有 `ResolvedKernel` 并通过统一的参数构建路径执行。

## 3. 系统性评价

| 审核维度 | 评价 | 主要依据 |
|---|---|---|
| 架构整体设计 | 良好但未闭环 | 语义层、dispatch、entry、ISA 实现分层合理，但 capability、workspace、runtime binding 存在断点 |
| 模块划分与职责边界 | 基本清晰 | Operator 负责语义和 shape，entry 负责运行时校验，ISA TU 负责计算；Add dtype helper 当前放在 `add_op.h`，造成 backend/const-eval 对具体 Operator 头的反向依赖 |
| 核心算法正确性 | 基本通过 | Add 广播、RMSNorm scalar/AVX2、Embedding 和 ElementwiseMul 的现有单元测试通过 |
| 数据处理流程 | 部分通过 | Graph → Lowering → Plan → Binding → Kernel 主链清晰，但 FunctionOperator fallback 和 workspace 规划没有闭环 |
| 内存管理 | 不通过 | borrowed view、workspace arena 和栈参数方向正确，但存在已证实的 stack-use-after-scope |
| 并行计算策略 | 不通过 | OpenMP 未接入构建和统一线程策略，只有源代码 pragma |
| 接口设计 | 部分通过 | `Status`、`TensorView`、`ResolvedKernel` 接口明确；SDK API 与 engine entry 的行为和校验不一致 |
| 错误处理 | 部分通过 | kernel entry 校验较充分；部分 `noexcept` 路径构造可能分配的 `std::string`，低内存时可能 `terminate`；部分边界使用 `AM_CHECK` 直接终止 |
| 性能优化 | 部分通过 | 计划期 resolve、运行期函数指针、AVX2 TU 隔离是优势；缺少完整线程、ISA 和端到端性能闭环 |
| 可扩展性 | 中等 | selector 维度完整；全局冻结 registry、静态注册和 priority tie-break 限制插件化和多后端扩展 |
| Phase 1 完整度 | 不满足 | 关键 Llama kernel、量化 Linear、端到端 Decode 和 C ABI 尚未完成 |
| 性能指标 | 无法证明 | 缺少 7B INT4 Decode、TTFT、启动时间和总内存 benchmark |

## 4. 架构优势

### 4.1 计划期解析、运行期直接调用

`KernelRegistry` 在初始化后冻结，ExecutionPlan 构建时解析并缓存 `ResolvedKernel`，运行时不再执行 map、hash 或字符串查找。

`NoHotpathResolve.ExecutorConsumesFrozenKernelWithoutBackendLookup` 测试覆盖了该约束。该设计符合推理运行时低开销 dispatch 的目标。

### 4.2 Selector 维度合理

`KernelSelector` 同时表达：

- device；
- activation dtype；
- weight dtype；
- weight format；
- ISA；
- Prefill/Decode phase。

`candidate.isa <= request.isa` 支持高级 ISA 请求回退到低级 ISA kernel，适合后续扩展 AVX512、AMX 和量化实现。

### 4.3 Kernel 入口防御性校验较完善

Add kernel entry 已覆盖：

- 参数与 TensorView 有效性；
- dtype 一致性和支持集合；
- rank 和广播兼容性；
- 空 Tensor；
- offset 和 numel 溢出；
- null data pointer；
- 整数加法溢出。

RMSNorm entry 也对 dtype、rank、shape、stride、epsilon 和 data pointer 进行了分层校验。

### 4.4 Reference 与优化实现分离

RMSNorm scalar 与 AVX2 实现位于独立 TU，并有 reference/optimized 数值对照测试。AVX2/FMA 编译选项只施加到特定 TU，避免整个核心库被强制编译为 AVX2 指令集。

### 4.5 Shape inference 支持延迟约束

`InferenceResult` 同时携带输出 TensorSpec 和运行时 ShapeConstraint。无法在符号阶段证明的维度相等或广播条件会在执行前重新验证，结构上适合动态 prompt 长度等场景。

### 4.6 Ownership 基本明确

`Tensor` 持有 Buffer，`TensorView`/`MutableTensorView` 借用数据及 shape/stride 元数据，`ResolvedKernel` 持有 attrs，`KernelContext` 借用 attrs 和 runtime binding。核心类型的注释总体清楚地描述了生命周期边界。

## 5. 其他潜在问题

### 5.1 Operator 热路径仍有虚调用

PRD 要求使用 Concepts 和静态分发，避免虚函数开销；当前 `LayerRunner` 每个执行步骤仍通过 `step.op->Run()` 发生一次虚调用，然后再调用 kernel function pointer。

单次虚调用相对于大型 kernel 的代价通常很小，但它与当前 PRD 的严格表述不一致。建议二选一：

- 修改 PRD，明确“允许每个算子一次语义层虚调用，kernel dispatch 必须为缓存函数指针”；
- 将 ExecutionStep 冻结为直接执行 thunk，彻底移除热路径 Operator 虚调用。

### 5.2 全局 Registry 生命周期限制

`CpuBackend` 构造函数会冻结全局 KernelRegistry。该设计已在 dispatch 文档中记录为有意偏离，Phase 1 下简单有效，但存在以下扩展限制：

- 首次构造 backend 后无法注册新 kernel；
- 动态插件较难支持；
- 测试之间共享全局状态；
- 多 backend 隔离依赖 selector，而非 registry ownership。

### 5.3 Selector 同优先级存在静态注册顺序依赖

当多个匹配 kernel priority 相同时，当前 Resolve 逻辑保留先注册项。跨 TU 静态初始化顺序不稳定，因此未来增加同优先级 scalar/AVX2 或 phase-specific kernel 时可能产生非显式选择结果。

建议使用确定性的匹配评分：priority、ISA specificity、phase specificity，再以稳定名称或显式 registration order 作为最终 tie-break。

### 5.4 Add 广播慢路径成本较高

位置：`src/backend/cpu/kernels/add/add_scalar.cpp:83`

当前每个输出元素都执行完整的 flat-index 到多维坐标分解，并为两个输入和输出分别计算 offset，复杂度为 `O(numel * rank)`，同时包含大量整数除法和取模。

该实现易于验证，适合作为 reference kernel，但在大尺寸 Prefill 广播中可能明显受限。建议先增加 benchmark，再改为预计算广播 stride 和增量式坐标迭代。

### 5.5 Alias/In-place 合同未显式表达

算子合同要求明确声明是否允许 in-place 和输入输出别名，但当前 KernelDescriptor、Operator 和 Tensor binding 中没有统一 alias policy。

Add 的简单同形 in-place 可能正确，但广播输入与输出重叠时可能覆盖后续仍需读取的数据；RMSNorm 输出与 weight 重叠也可能破坏后续行。应默认拒绝未声明的 overlap，允许的 in-place 模式必须作为显式 contract。

### 5.6 AVX2 与 Scalar 数值累积不同

RMSNorm scalar 使用 double 累积平方和，AVX2 使用 float/FMA 累积。优化路径与 reference 路径只要求 tolerance 一致，这一方向合理，但当前测试规模较小，尚未系统覆盖：

- hidden size 4096/8192；
- 极大或极小输入；
- NaN/Inf 行为；
- 接近 argmax 决策边界的误差传播。

## 6. Phase 1 技术需求符合性

### 6.1 当前 Kernel 覆盖

`OpType` 当前定义 13 个有效算子类型。已注册 CPU kernel 的算子只有：

- Add；
- RMSNorm；
- Embedding；
- ElementwiseMul。

语义 Operator 已存在但没有可执行 CPU kernel 的包括：

- Linear；
- MatMul；
- Silu；
- SiluMul。

尚未形成完整 Operator/Kernel 实现的包括：

- RoPE；
- Attention；
- Softmax；
- Argmax；
- KVCacheUpdate。

PRD 最小 Llama kernel 集中，当前只有 Embedding 和 RMSNorm 完整落地。INT8/INT4 Linear、Attention、RoPE、SwiGLU、Softmax 和 Argmax 均未形成可执行链路。

### 6.2 Hard Gates

| PRD Hard Gate | 当前状态 |
|---|---|
| 与 HF 参考实现端到端一致 | 未验证；没有 `LlamaDecode.*` 完整测试 |
| Decode 稳态零分配 | 未验证；没有 malloc/free hook 验收测试 |
| 同平台确定性 | 部分覆盖；没有 100 次完整 Decode hash 测试 |
| TSAN 无竞争 | 聚焦算子/kernel 子集通过；全量验证尚不完整 |
| ASAN/LSAN 无错误 | 失败；已确认 `stack-use-after-scope` |

### 6.3 性能指标

现有 benchmark 主要覆盖：

- RMSNorm scalar/AVX2；
- AVX2 dot product；
- ammalloc；
- string。

缺少以下 PRD 级 benchmark：

- `BM_LlamaDecode`；
- `BM_KVCache`；
- `BM_QuantLinear`；
- 7B INT4、4-core、2k context 的 tok/s；
- 2k Prefill TTFT；
- 冷启动时间；
- 完整运行时内存峰值。

因此当前不能证明以下指标：

- 吞吐量不低于 10 tok/s；
- 总内存不超过 4 GB；
- 启动时间不超过 2 秒；
- TTFT 不超过 500 ms。

在 INT4 Linear 和完整 Decode 尚未实现的情况下，当前实现实际上不具备执行上述正式测量的条件。

## 7. 改进优先级

### 7.1 P0：立即处理

1. 修复 `InvokeResolvedKernel()` 参数缓冲区生命周期，确保 params 对象覆盖完整 kernel 调用。
2. 为 Add、Embedding、RMSNorm 和 ElementwiseMul 增加 execution-plan ASAN 回归测试。
3. 重构 `LaunchRmsNorm`，接入运行时 ISA 检测、scalar fallback 和统一 entry 校验。

### 7.2 P1：形成架构闭环

1. 由 Operator 计算 workspace requirement，并在 ExecutionPlanBuilder 中统一规划。
2. 根据 lifetime/reusable 实现 workspace 区间复用。
3. 实现 CPU capability detection，并由 Runtime/Lowering 自动生成 ISA selector。
4. 在 CMake 中正式接入 OpenMP，线程数由 RuntimeContext 统一控制。
5. 删除或修复 FunctionOperator raw-kernel fallback。
6. 明确 alias/in-place contract。
7. 将 kernel 选择 tie-break 改为确定性评分。

### 7.3 P2：完成 Phase 1 能力

1. 实现 INT8/INT4 Linear/GEMM 和权重重排。
2. 实现 RoPE、Attention、SwiGLU、Softmax、Argmax 和 KVCacheUpdate。
3. 完成真实 RuntimeBindingContext 到完整 Llama Decode 的生产绑定流程。
4. 实现版本化 C ABI 的 runtime/model/session/generate 接口。
5. 增加 HF parity、零分配、确定性、TSAN、ASAN/LSAN 和性能门禁。
6. 在 benchmark 证据支持下优化 Add/ElementwiseMul 广播路径和 RMSNorm 线程阈值。

## 8. 验证结果

### 8.1 普通构建

聚焦执行以下核心范围：

- Add kernel/operator；
- RMSNorm kernel/entry/operator；
- Embedding kernel；
- KernelRegistry 和 resolve；
- NoHotpathResolve；
- RuntimeBindingContext；
- WorkspaceRequirementPlanning。

结果：**119/119 测试通过**。

独立 QA 线重新构建并运行全量普通单元测试，结果：**2594/2594 测试通过**。

### 8.2 ThreadSanitizer

聚焦 operator/kernel 子集在 TSAN 下未发现数据竞争。由于完整测试套件在环境时间限制内未完成，TSAN Hard Gate 只能评为部分验证。

### 8.3 AddressSanitizer

执行：

```bash
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
    ./build-asan/tests/unit/aethermind_unit_tests \
    --gtest_filter=AddKernel.EndToEndThroughGraphLoweringAndExecutor
```

结果：失败，稳定报告 `stack-use-after-scope`，调用链为：

```text
Operator::InvokeResolvedKernel
  -> EmbeddingOp::Run
  -> EmbeddingKernel
  -> TensorView::is_valid
```

问题地址位于 `Operator::InvokeResolvedKernel()` 的局部 `buffer` 内，证明第 2.1 节问题为真实缺陷而非静态分析误报。

### 8.4 Benchmark

RMSNorm scalar、RMSNorm AVX2 和 AVX2 dot product 聚焦 benchmark 可以运行。当前执行的是 Debug build，只能证明 benchmark 路径存在，不能作为正式性能指标。

## 9. 最终判断

当前算子/kernel 架构具备良好的基础方向：计划期解析、语义与实现分离、ISA selector、borrowed TensorView、workspace 抽象和 reference/optimized 分层均值得保留。

但从生产级 Phase 1 验收角度，当前实现存在明确内存安全缺陷，且完整 Llama kernel、量化、并行、自动 ISA dispatch、C ABI、端到端测试和性能门禁尚未完成。

因此最终判断为：

> **架构方向可继续演进，但当前实现不满足 Phase 1 技术交付与性能验收条件。应先修复 P0 生命周期和 ISA 安全问题，再补齐 workspace/线程/dispatch 闭环，最后完成完整 Llama 执行链及验收门禁。**
