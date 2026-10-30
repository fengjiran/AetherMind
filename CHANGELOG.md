# Changelog

本文件记录 AetherMind 的行为可见变更（API 变更、配置变更、性能基线变化、缺陷修复），按语义化版本组织。细节与设计背景见 `docs/designs/` 与 `docs/decisions/`（[文档系统规范](docs/guides/documentation-guide.md)）。

## [Unreleased]

### Added

- Runtime C ABI v1：M5 冻结配置/version、opaque handles、error/owner/buffer 合同，M6 实现 version 与 create/load/create/generate/destroy 链；支持 CPU FP32 Llama Dense、O1/plain 与 O2/packed、调用方输出缓冲区，模型/Session 保活必要 owner。
- inference 公共 `model_execution_policy.h` 提供唯一的 `ValidateFp32ExecutionInputs` 和 `MakeFp32KVCacheOptions`；新增 `cpu_inference_options.h` 的 `CpuInferencePreset` / `MakeCpuCompileOptions`。CLI 与 C adapter 共用准入、模型 KV 映射及 preset，删除应用旧实现与旧类型。
- 文档系统基础设施：`docs/README.md`（索引与术语表）、`docs/guides/documentation-guide.md`（文档系统规范）、`docs/templates/`（模块设计/ADR 模板）、`tools/verify_docs.py`（漂移检测脚本）。
- `docs/decisions/` 决策记录目录与 `docs/archive/` 归档区建立。

### Changed

- 删除未使用的公共 `function.h`、`function_traits.h`、`registry.h` 与 `src/function.cpp`：移除 `Function`/`TypedFunction`/`Registry`、`DEFINE_STATIC_FUNCTION`/`REG_VAR_DEF` 宏及 Any 的 `Function` 类型名。原 C++ 调用方需迁移并重新编译，C ABI 不变。
- 删除无生产消费者的公共 `utils/qualified_name.h` 及其单测 `test_qualified_name.cpp`：`QualifiedName` 原为已删除的 `Function` 体系配套，仓库内无源码消费者；C ABI 与其余 utils 接口不变。
- 删除无生产消费者的遗留顶层头 `memory_format.h` 及其单测 `test_memory_format.cpp`：`MemoryFormat` 与 `GetChannelsLastStrides*`/`IsChannelsLastStrides*` 在 Tensor 重构后仅剩测试引用；C ABI 与现有张量/layout 接口不变。
- 删除无消费者的遗留顶层头 `layout.h`：`Layout` 枚举、`kStrided` 等常量与 `operator<<` 无任何源码或测试引用；C ABI 与既有 KV/layout 契约不变。
- 删除无生产消费者的 `Any` + `Array` 子系统：移除公共 `any.h`、`container/array.h`、实现 `any.cpp`/`array.cpp` 及对应单测，并清理仅服务于该子系统的 `is_valid_array_type` 与 `Any` 类型名特化；`any_utils.h` 的通用 concepts 保留供 Map/amstring 使用，C ABI 不变。
- 清理公共 utils：删除未使用的 `lru.h`、`xxh3.h`、`bits.h` 及 DataType 的无效 include；`FillMemoryJunk` 移入 memory 私有实现，删除公共 `memory_utils.h` 与 pattern getters，不留转发。复合标准类型键改用显式 `aethermind::hash`，不再注入通用 `std::hash` 偏特化；Map 的 Fibonacci helper 归入 container detail。原 C++ 调用方需迁移并重新编译，C ABI 不变。
- `logging.h` 只提供断言，不再传递包含 spdlog；实际日志调用方直接包含并链接 spdlog，核心库将该依赖设为 PRIVATE。
- 整数可表示性检查统一复用 `utils/overflow_check.h` 的 `CheckNarrow`：失败时保留输出值；从 inference 公共头删除 `PositiveSize`，正值要求和字段诊断改由 policy、runner 与 C adapter 的私有包装保留，C ABI 错误合同保持不变。
- `KVAppendTransaction` 的逐层 append 状态统一使用 execution 层的 `KVAppendLayerState : uint8_t`；`ExecutionContext` 拥有 typed vector，`ResetKVLayerTransactionScratch()` 返回 `std::span<KVAppendLayerState>`，事务的 `layer_states` 只借用该存储。原字节 span 调用方需迁移并重新编译，冷路径预分配与整计划提交行为保持不变。
- `BoundValue` 使用私有 `variant<monostate, TensorView, MutableTensorView>` 表达未绑定/只读/可写状态，移除重复 view 和 `has_writable`；调用方改用按值返回的 `ReadView()` 与借用指针 `TryGetWritableView()`，需迁移并重新编译。
- InferenceSession 与 plan 校验共享 `inference::detail::PhaseContract`，移除重复结构和逐字段转换；C++ 调用方需重新编译。
- `docs/designs/architecture/architecture_overview.md` 确立为全系统唯一架构总览；`docs/designs/aethermind_arch_design.md` 标记 Deprecated 并归档。
- `KVCacheManager` 禁止复制；移动仅限没有 borrowed view 的资源装配阶段，并明确共享 Runtime 请求须由调用方串行化。
- Workspace 纯数据契约统一放入 `base/workspace_types.h`，移除 `base/workspace_binding.h`；原调用方需更新 include 路径。类型、字段布局及行为保持不变，arena 接口与布局规划分别保留在 `base/workspace_arena.h` 和 `runtime/workspace_planner.h`。
- Workspace、activation 和 kernel params 的 offset 对齐统一为 `base/alignment.h` 的 `AlignOffsetUp()`，实现位于 `src/base/alignment.cpp`；删除公开的 `AlignWorkspaceOffset()` 和转发谓词 `IsValidWorkspaceAlignment()`，直接使用 `IsValidAlignment()` 校验，不保留转发接口。调用方需迁移并重新编译、链接核心库；workspace 布局规划仍位于 `src/runtime/workspace_planner.cpp`。
- `BuildKernelContext` 现在一次性填入 workspace binding、KV bindings 和 `PreparedExecutionBindings` 所拥有的 `kernel_params`；`InvokePreparedKernel` 收敛为 `const ResolvedKernel&` 与 `const KernelContext&` 两参数接口，只校验 params contract 后调用 kernel，不再修改 context。直接调用该低层接口的 caller 必须预设 `KernelContext::kernel_params` 并重新编译；`BuildAndInvokeKernelForTesting` 保持原有行为。
- Prefill 的 plan 执行、KV 前置条件与 watermark 校验、以及最后位置首 token 读取收敛到 inference internal 的 private `RunPrefill` helper；`Generate` 仍负责请求输入/资源准备、首 token 推入、EOS/limit 判断、Decode 准备与 RAII 清理。该 helper 不引入公开 C++ 或 C ABI 变化，`RunDecodeLoop` 签名保持不变；Prefill 已提交后若 output 校验失败不回滚，由请求清理释放 reservation。
- `KVCacheSessionView::CommitAppend(size_t begin, size_t end) const noexcept` 成为公开提交入口，移除 `KVCacheCommitAccess` 及其私有 header；调用方必须先证明完整 append plan 成功且所有 layer 写完，方法自身只验证 generation、lifecycle、range 与容量，不证明写入完成。生产路径仍由 `LayerRunner` 在完整 plan 成功后调用；旧 `CommitUntil` 不恢复。

### Deprecated

- `docs/designs/aethermind_arch_design.md`（内容并入架构总览）。

### Fixed

- 修复 checked arithmetic portable fallback 的有符号溢出误报/漏报及窄整数提升 UB；修复 hash 头的 include 顺序依赖、定制点歧义和 pair 复制，以及 Map 的全位宽右移 UB。
- 浮点解析的原生与 POSIX fallback 统一完整输入校验，失败保留输出；fallback 处理 locale/分配失败、可表示 subnormal 及宿主 rounding mode 的恢复。
- memory junk-fill 支持任意字节对齐并保留原 pattern；断言宏限定 helper 命名空间，可在全局或其他命名空间调用。
- C ABI 在 KV 分配前检查 FP32 行字节/64-byte 对齐溢出，极大 head_dim 返回 Overflow，不落入既有 KV padding 的无符号回绕错误路径。
- `am_error_create` 的 message 构造抛出时自动释放已分配 error 对象；对象/message OOM 均返回 NULL，不传播异常。
- `HfModelValidator` 的 `hidden_act` 支持范围收窄为仅 `silu`；`gelu/relu` 在模型加载与直接构图入口返回 `InvalidArgument`，不再被静默编译成 SwiGLU。
- Kernel params 对齐在错误诊断分配失败时返回空 message 的 `ResourceExhausted`，避免从 `noexcept` 对齐函数传播 `std::bad_alloc`。
- CPU activation 分配遇到 `ENOMEM` 时上报可恢复失败；`InferenceSession::Generate` 将可传播的 `std::bad_alloc` 转为 `ResourceExhausted`，清理 KV 后允许重试。
- `KVCacheManager` 校验 Reset/Release 的 view owner，并在 re-init 时保留 generation、拒绝 generation 回绕，防止跨 Manager 操作和旧 view 复活。
- 修正 InferenceSession 文档的共享 Runtime 并发安全、输入角色校验、Attention 复杂度和清理错误传播表述。
- Workspace 规划先完整验证再填写 offsets，失败时保持输入不变；对齐与规划接口将诊断分配失败转换为无额外分配的 `ResourceExhausted`。
