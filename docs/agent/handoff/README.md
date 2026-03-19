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

恢复工作时：
1. **只读取** `status: active` 的 handoff（一个 workstream 一个）
2. **忽略** `status: superseded` 和 `status: closed`
3. 排序：`created_at` 降序，文件名字典序 tie-break

⚠️ **完整规范**：见 `docs/agent/memory/README.md`（按需升级读取）
