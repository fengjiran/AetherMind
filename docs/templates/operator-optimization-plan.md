# <算子名> 优化方案

> 复制本模板新建算子专项提案，存放于 `docs/improvement-plan/`。流程与证据要求见 [算子开发与优化工作流](../guides/operator-development-workflow.md)。

- **状态**: Draft / In Progress / Implemented / Superseded
- **版本**: 1.0
- **日期**: YYYY-MM-DD
- **产品边界**: [AetherMind 当前产品 PRD](../products/aethermind_prd.md)
- **Change Profile**: Semantic / Reference / Optimized / Packing/Layout / Fusion / Threading / Tuning-only
- **关联代码**: <paths/symbols>
- **关联测试**: <unit/benchmark paths>
- **关联 ADR**: <无或链接>

## 1. 结论与范围

- 推荐方案和原因。
- 包含与不包含的范围。
- 明确当前事实、推断、假设和建议。

## 2. 已验证的当前状态

| 能力 | 当前实现 | 证据 | 限制 |
|---|---|---|---|
| <能力> | <实现> | <代码/测试链接> | <限制> |

## 3. 语义、约束与 invariant

- input/output/port semantics；
- dtype/rank/shape/stride/layout；
- alias/in-place/output injectivity；
- zero-size/null/overflow；
- numerical accumulation/rounding/error budget；
- unsupported/unknown behavior；
- ownership/lifetime/thread-safety。

## 4. 瓶颈与假设

| 项目 | 类型 | 证据或验证方式 |
|---|---|---|
| <瓶颈/假设> | 已验证事实 / 推断 / 假设 | <profile/benchmark/inspection> |

## 5. 目标架构与数据流

- 职责边界；
- 核心数据结构；
- 接口；
- cold/hot path；
- fallback；
- control/data flow。

## 6. 方案与备选

| 方案 | 优点 | 缺点 | 结论 |
|---|---|---|---|
| <方案 A> | | | 推荐/拒绝 |

重大且长期的决策必须另建 ADR，本节只摘要并链接。

## 7. Benchmark 与验证设计

- microkernel/primitive；
- prepared production path；
- preparation/packing/binding；
- integration/end-to-end；
- shape/layout/cache/thread matrix；
- correctness guard；
- counters 与原始数据保存方式；
- baseline/candidate identity。

## 8. 工作包与依赖

```text
O0/O1/O2 baseline
    → <工作包 A>
    → <工作包 B>
```

### <工作包 ID>：<名称>

- 目标：
- 实施内容：
- 依赖：
- 风险：
- 退出条件：
- 当前状态：Not Started / In Progress / Complete / Blocked
- 正式证据：<validation report link>

## 9. 风险与依赖

| 风险 | 影响 | 缓解 |
|---|---|---|
| | | |

## 10. 验收标准

### Correctness

- [ ]

### Architecture/API

- [ ]

### Performance

- [ ]

### Integration

- [ ]

## 11. 相关文档

- 工作流：
- 当前设计：
- 实验日志：
- 验证报告：
- ADR：

## 12. 变更记录

| 日期 | 版本 | 变更 | 原因 | 证据/PR |
|---|---|---|---|---|
| YYYY-MM-DD | 1.0 | 初始提案 | | |

