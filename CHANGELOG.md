# Changelog

本文件记录 AetherMind 的行为可见变更（API 变更、配置变更、性能基线变化、缺陷修复），按语义化版本组织。细节与设计背景见 `docs/designs/` 与 `docs/decisions/`（[文档系统规范](docs/guides/documentation-guide.md)）。

## [Unreleased]

### Added

- 文档系统基础设施：`docs/README.md`（索引与术语表）、`docs/guides/documentation-guide.md`（文档系统规范）、`docs/templates/`（模块设计/ADR 模板）、`tools/verify_docs.py`（漂移检测脚本）。
- `docs/decisions/` 决策记录目录与 `docs/archive/` 归档区建立。

### Changed

- `docs/designs/architecture/architecture_overview.md` 确立为全系统唯一架构总览；`docs/designs/aethermind_arch_design.md` 标记 Deprecated 并归档。

### Deprecated

- `docs/designs/aethermind_arch_design.md`（内容并入架构总览）。

### Fixed

- （无）
