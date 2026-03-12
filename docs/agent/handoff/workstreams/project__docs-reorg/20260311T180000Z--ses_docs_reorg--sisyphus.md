---
kind: handoff
schema_version: "1.1"
created_at: 2026-03-11T18:00:00Z
session_id: ses_docs_reorg
task_id: task_docs_struct_001
module: project
slug: docs-reorg
submodule: null                   # 项目级工作：必须为 null
agent: sisyphus
status: active
memory_status: not_needed
supersedes: null
closed_at: null
closed_reason: null
---

# docs/ 目录结构重构

## 目标
重构 docs/ 目录，将 Agent Memory System 文档收拢到 docs/agent/，设计文档移到 docs/designs/，避免命名冲突。

## 当前状态
- 已完成：
  - ✅ 创建 docs/agent/ 目录结构
  - ✅ 移动 memory/ → docs/agent/memory/
  - ✅ 移动 prompts/ → docs/agent/prompts/
  - ✅ 移动 handoff/ → docs/agent/handoff/
  - ✅ 移动 decisions/ → docs/agent/decisions/
  - ✅ 重命名 docs/modules/ → docs/designs/
  - ✅ 更新所有路径引用
  - ✅ 提交 git commit

- 未完成：
  - ⏳ 验证所有文档链接有效
  - ⏳ 更新 README.md 主说明
  - ⏳ 通知团队成员目录变更

## 涉及文件
- docs/agent/memory/README.md（主规范）
- docs/agent/memory/QUICKSTART.md（快速入门）
- docs/agent/prompts/quick_resume.md（恢复流程）
- docs/designs/ammalloc/*.md（设计文档）

## 目录结构变更
```
重构前：
docs/
├── agent_memory_system.md
├── memory/
├── prompts/
├── handoff/
├── decisions/
└── modules/          # 设计文档

重构后：
docs/
├── agent/            # Agent Memory System 完整收拢
│   ├── memory_system.md
│   ├── memory/
│   ├── prompts/
│   ├── handoff/
│   └── decisions/
├── designs/          # 设计文档（原 modules）
└── ...其他目录
```

## 已确认接口与不变量
- 接口：无特殊接口
- 前置条件：无
- 后置条件：无
- 不变量：无

## 阻塞点
- 无

## 推荐下一步
1. 运行 `find docs/ -name "*.md" -exec grep -l "docs/modules/" {} \;` 检查遗漏引用
2. 更新根目录 README.md 说明新的文档组织方式
3. 在团队频道通知目录变更

## 验证方式
```bash
# 检查是否有遗漏的旧路径引用
grep -r "docs/modules/" docs/ --include="*.md" || echo "无遗漏"
grep -r "docs/memory/" docs/ --include="*.md" | grep -v "docs/agent/memory" || echo "无遗漏"

# 检查关键文件存在
ls docs/agent/memory/README.md
ls docs/agent/prompts/quick_resume.md
ls docs/designs/ammalloc/
```

## 沉淀计划
完成后，将稳定的目录结构约定写入 `docs/agent/memory/project.md`：
- `docs/agent/` - Agent Memory System 专属
- `docs/designs/` - 设计文档
- `docs/guides/` - 通用指南
- `docs/products/` - 产品文档
