# 算子优化实验日志索引

本目录保存算子开发与优化的追加式实验日志。规范和元数据要求见[算子开发与优化工作流](../../guides/operator-development-workflow.md)，新日志从[算子实验日志模板](../../templates/operator-experiment-log.md)复制。

## 目录约定

```text
docs/logs/operators/<operator>/
├── README.md
├── <work-package>-log.md
└── <next-work-package>-log.md
```

- 一个独立假设、参数 sweep 或失败结论对应一个日志条目；
- 日志只追加，不覆盖或删除失败实验；
- 完整 JSON、perf 和反汇编放入 raw artifact，不粘贴到日志；
- 达到证据门禁后，在 `docs/tests/operators/<operator>/` 新建正式 validation report。

## 算子索引

当前尚无正式算子实验日志。新增首篇日志时，在此登记算子目录、关联专项提案和当前工作包。

