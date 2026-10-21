# Changelog

本文件记录 AetherMind 的行为可见变更（API 变更、配置变更、性能基线变化、缺陷修复），按语义化版本组织。细节与设计背景见 `docs/designs/` 与 `docs/decisions/`（[文档系统规范](docs/guides/documentation-guide.md)）。

## [Unreleased]

### Added

- 文档系统基础设施：`docs/README.md`（索引与术语表）、`docs/guides/documentation-guide.md`（文档系统规范）、`docs/templates/`（模块设计/ADR 模板）、`tools/verify_docs.py`（漂移检测脚本）。
- `docs/decisions/` 决策记录目录与 `docs/archive/` 归档区建立。

### Changed

- `KVAppendTransaction` 的逐层 append 状态统一使用 execution 层的 `KVAppendLayerState : uint8_t`；`ExecutionContext` 拥有 typed vector，`ResetKVLayerTransactionScratch()` 返回 `std::span<KVAppendLayerState>`，事务的 `layer_states` 只借用该存储。原字节 span 调用方需迁移并重新编译，冷路径预分配与整计划提交行为保持不变。
- `BoundValue` 使用私有 `variant<monostate, TensorView, MutableTensorView>` 表达未绑定/只读/可写状态，移除重复 view 和 `has_writable`；调用方改用按值返回的 `ReadView()` 与借用指针 `TryGetWritableView()`，需迁移并重新编译。
- InferenceSession 与 plan 校验共享 `inference::detail::PhaseContract`，移除重复结构和逐字段转换；C++ 调用方需重新编译。
- `docs/designs/architecture/architecture_overview.md` 确立为全系统唯一架构总览；`docs/designs/aethermind_arch_design.md` 标记 Deprecated 并归档。
- `KVCacheManager` 禁止复制；移动仅限没有 borrowed view 的资源装配阶段，并明确共享 Runtime 请求须由调用方串行化。
- Workspace 纯数据契约统一放入 `base/workspace_types.h`，移除 `base/workspace_binding.h`；原调用方需更新 include 路径。类型、字段布局及行为保持不变，arena 接口与布局规划分别保留在 `base/workspace_arena.h` 和 `runtime/workspace_planner.h`。
- Workspace、activation 和 kernel params 的 offset 对齐统一为 `base/alignment.h` 的 `AlignOffsetUp()`，实现位于 `src/base/alignment.cpp`；删除公开的 `AlignWorkspaceOffset()` 和转发谓词 `IsValidWorkspaceAlignment()`，直接使用 `IsValidAlignment()` 校验，不保留转发接口。调用方需迁移并重新编译、链接核心库；workspace 布局规划仍位于 `src/runtime/workspace_planner.cpp`。

### Deprecated

- `docs/designs/aethermind_arch_design.md`（内容并入架构总览）。

### Fixed

- Kernel params 对齐在错误诊断分配失败时返回空 message 的 `ResourceExhausted`，避免从 `noexcept` 对齐函数传播 `std::bad_alloc`。
- CPU activation 分配遇到 `ENOMEM` 时上报可恢复失败；`InferenceSession::Generate` 将可传播的 `std::bad_alloc` 转为 `ResourceExhausted`，清理 KV 后允许重试。
- `KVCacheManager` 校验 Reset/Release 的 view owner，并在 re-init 时保留 generation、拒绝 generation 回绕，防止跨 Manager 操作和旧 view 复活。
- 修正 InferenceSession 文档的共享 Runtime 并发安全、输入角色校验、Attention 复杂度和清理错误传播表述。
- Workspace 规划先完整验证再填写 offsets，失败时保持输入不变；对齐与规划接口将诊断分配失败转换为无额外分配的 `ResourceExhausted`。
