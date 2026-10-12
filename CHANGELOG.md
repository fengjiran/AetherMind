# Changelog

本文件记录 AetherMind 的行为可见变更（API 变更、配置变更、性能基线变化、缺陷修复），按语义化版本组织。细节与设计背景见 `docs/designs/` 与 `docs/decisions/`（[文档系统规范](docs/guides/documentation-guide.md)）。

## [Unreleased]

### Added

- 文档系统基础设施：`docs/README.md`（索引与术语表）、`docs/guides/documentation-guide.md`（文档系统规范）、`docs/templates/`（模块设计/ADR 模板）、`tools/verify_docs.py`（漂移检测脚本）。
- `docs/decisions/` 决策记录目录与 `docs/archive/` 归档区建立。

### Changed

- `docs/designs/architecture/architecture_overview.md` 确立为全系统唯一架构总览；`docs/designs/aethermind_arch_design.md` 标记 Deprecated 并归档。
- `KVCacheManager` 禁止复制；移动仅限没有 borrowed view 的资源装配阶段，并明确共享 Runtime 请求须由调用方串行化。
- Workspace 公共头文件按职责改名为 `base/workspace_binding.h` 与 `runtime/workspace_planner.h`；调用方需更新 include 路径，类型及行为保持不变。
- Workspace 对齐及布局规划的实现移入 `src/runtime/workspace_planner.cpp`；头文件保留公共声明和 `constexpr` 对齐校验，调用方需链接核心库。

### Deprecated

- `docs/designs/aethermind_arch_design.md`（内容并入架构总览）。

### Fixed

- CPU activation 分配遇到 `ENOMEM` 时上报可恢复失败；`InferenceSession::Generate` 将可传播的 `std::bad_alloc` 转为 `ResourceExhausted`，清理 KV 后允许重试。
- `KVCacheManager` 校验 Reset/Release 的 view owner，并在 re-init 时保留 generation、拒绝 generation 回绕，防止跨 Manager 操作和旧 view 复活。
- 修正 InferenceSession 文档的共享 Runtime 并发安全、输入角色校验、Attention 复杂度和清理错误传播表述。
- Workspace 规划先完整验证再填写 offsets，失败时保持输入不变；对齐与规划接口将诊断分配失败转换为无额外分配的 `ResourceExhausted`。
