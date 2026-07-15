# Handoff 目录

Agent Memory System 的会话交接文件存储目录。

## 目录结构

```
docs/agent/handoff/workstreams/
├── <module>__<submodule-or-none>/
│   └── YYYYMMDDTHHMMSSZ--<session_id>--<agent>.md
└── project__<slug>/
    └── YYYYMMDDTHHMMSSZ--<session_id>--<agent>.md
```

## 读取规则（关键）

恢复工作时，按以下算法选择 handoff：

1. **扫描所有候选 handoff 的 YAML frontmatter**（仅元数据，不读取正文）
2. **排除 frontmatter 缺失或损坏**的无效文件
3. **筛选 `status: active`** 的有效条目
4. **排序**：`created_at` 降序；相同时按文件名字典序 tie-break
5. **仅读取排序后第一个 handoff** 的完整正文（同一 workstream 只读取一个）

> 元数据扫描（frontmatter 收集）不计为加载多个 handoff 正文。已加载文件列表中仅包含选中的那一个 handoff。

⚠️ **完整规范**：见 `docs/agent/memory/README.md`（按需升级读取）
