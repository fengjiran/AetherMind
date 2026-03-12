---
kind: handoff
schema_version: "1.1"
created_at: 2026-03-12T08:00:00Z
session_id: ses_31f8f5b8affe1bnCOcehtKLMvq
task_id: task_memory_system_v1_1_fixes
module: project
submodule: null
slug: agent-memory-v1.1
agent: sisyphus
status: active
memory_status: applied
supersedes: 20260312T060000Z--ses_31f8f5b8affe1bnCOcehtKLMvq--sisyphus.md
closed_at: null
closed_reason: null
---

# Agent Memory System v1.1 修复与改进

## 目标
修复 Agent Memory System 在实际使用中发现的两个关键问题，确保项目级工作和模块级工作的加载流程清晰可区分。

## 当前状态
- ✅ **已完成**：
  - 修复 `docs/agent/prompts/quick_resume.md` 的项目级/模块级工作区分问题
    - 分离检查清单（模块工作6步 vs 项目级工作4步）
    - 分离 Resume Gate 模板（两种工作流独立）
    - 分离输出模板（明确标注跳过项）
    - 强化路径说明（区分模板 vs 存储目录）
  - 重命名 `handoff.md` → `handoff_template.md`
    - 消除与存储目录的命名混淆
    - 更新所有17处路径引用

- ⏳ **待回写 stable memory**：
  - `docs/agent/memory/project.md`：更新 slug 映射表，记录 v1.1 修复内容

## 涉及文件
- `docs/agent/prompts/quick_resume.md`（修复项目级/模块级区分）
- `docs/agent/prompts/handoff_template.md`（重命名，原 handoff.md）
- `docs/agent/memory/QUICKSTART.md`（更新引用）
- `docs/agent/memory/project.md`（待更新 slug 映射表）
- `docs/agent/memory/README.md`（更新引用）
- `docs/agent/prompts/README.md`（更新引用）
- `docs/agent/prompts/new_session_template.md`（更新引用）
- `docs/agent/memory_system.md`（更新引用）

## 已确认接口与不变量
- **接口**：项目级 workstream 使用 `project__<slug>` 格式
- **不变量**：
  - 项目级工作跳过 `module.md` 和 `submodule.md`
  - handoff 模板在 `prompts/handoff_template.md`，存储在 `handoff/workstreams/`

## 阻塞点
- 无

## 推荐下一步
1. 更新 `docs/agent/memory/project.md` 的 slug 映射表，添加本次修复记录
2. 将当前 handoff 标记为 `memory_status: applied`
3. 提交所有更改到 git

## 验证方式
```bash
# 检查文件重命名
ls docs/agent/prompts/handoff_template.md

# 检查旧引用已清理
grep -r "prompts/handoff\.md" docs/ --include="*.md" || echo "无旧引用"

# 检查 quick_resume.md 区分明确
grep -A2 "项目级工作跳过" docs/agent/prompts/quick_resume.md
```

## 沉淀计划
在 `docs/agent/memory/project.md` 的 slug 映射表中添加：
- `agent-memory-v1.1-fixes` - Agent Memory System v1.1 修复：区分项目级/模块级工作流，重命名 handoff.md → handoff_template.md
