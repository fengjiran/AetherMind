# <算子名> 优化实验日志

> 本文是追加式过程记录。不要覆盖旧实验或删除失败结果。正式结论写入使用[算子验证报告模板](operator-validation-report.md)创建的 validation report。

- **算子/工作包**: <operator / work-package>
- **专项提案**: <link>
- **日志范围**: <日期或工作包范围>
- **原始数据位置**: <CI artifact / benchmark-results path>

## 日志索引

| 日期 | 实验 ID | 假设 | 结论 | 状态 |
|---|---|---|---|---|
| YYYY-MM-DD | EXP-001 | <一句话> | <一句话> | Accepted / Rejected / Needs More Data |

## YYYY-MM-DD — EXP-001：<实验标题>

### 1. 假设

- 预期机制：
- 目标 shape/layout：
- 成功判定：
- 可能的反例：

### 2. 代码与环境

```text
git commit:
working tree clean/dirty:
baseline commit:
candidate commit:
CPU / stepping / microcode:
OS / kernel:
compiler / version:
build type / flags:
effective CPU features:
threads / affinity / NUMA:
governor / turbo / SMT:
run ID / artifact / checksum:
```

### 3. 改动

- 修改内容：
- 保持不变的合同：
- 预期影响：

### 4. 验证命令

```bash
# unit tests

# benchmark

# perf / disassembly
```

### 5. Correctness

| 测试 | 结果 | 备注 |
|---|---|---|
| | PASS/FAIL/NOT RUN | |

### 6. 性能结果摘要

| Case | Baseline | Candidate | Delta | Repetitions/CI | Raw artifact |
|---|---:|---:|---:|---|---|
| | | | | | |

不要把完整 JSON 或长终端输出粘贴到此处。

### 7. 分析

**已验证事实**：

-

**推断**：

-

**尚未验证的假设**：

-

### 8. 决定

- Accepted / Rejected / Needs More Data
- 原因：
- 下一步：
- 是否需要正式 validation report/ADR：

