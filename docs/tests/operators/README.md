# 算子实验与验证记录索引

本目录保存算子实验过程记录，以及 correctness、performance 和 integration 的正式验证报告。规范和证据门禁见[算子开发与优化工作流](../../guides/operator-development-workflow.md)；实验日志从[算子实验日志模板](../../templates/operator-experiment-log.md)复制，新报告从[算子验证报告模板](../../templates/operator-validation-report.md)复制。

## 目录约定

```text
docs/tests/operators/<operator>/
├── README.md
├── <work-package>-log.md
└── <operator>_<scope>_validation_<YYYY-MM-DD>.md
```

- 每份报告绑定 candidate/baseline commit、机器环境、运行命令和 raw artifact；
- 报告是不可变证据快照，新证据新建文件，不覆盖旧报告；
- `Current` 表示当前采用的证据，旧报告可标记 `Superseded` 但不删除；
- 未运行项、测试限制和可能失效范围必须明确记录。

## 算子索引

| 算子 | 目录 | 当前证据 |
|---|---|---|
| GEMM | [gemm/](gemm/) | G0 本地 reference baseline、配对 A/B 噪声 floor 与 Roofline 定位 Accepted；production gate Needs More Data |
