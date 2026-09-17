# <算子名> <范围> 验证报告

> 本文是指定代码版本、机器和环境的不可变证据快照。新证据应新建报告，不覆盖本报告历史结论。

- **状态**: Current / Superseded
- **日期**: YYYY-MM-DD
- **专项提案**: <link>
- **工作包**: <ID>
- **Change Profile**: <profiles>
- **Candidate commit**: `<sha>`
- **Baseline commit**: `<sha>`
- **Working tree**: clean / dirty（dirty 时附 diff artifact）
- **Raw artifact**: <ID/path/URL>
- **Artifact checksum**: <SHA256>
- **关联 ADR**: <无或链接>

## 1. 验证目标与结论

- 要证明的能力：
- 不在本报告证明范围内的内容：
- 最终判定：Accepted / Rejected / Needs More Data

## 2. 被验证实现

- production/source call path；
- descriptor/selector/recipe；
- fallback；
- ownership/workspace/threading；
- 相对 baseline 的行为差异。

## 3. 环境

| 项目 | 值 |
|---|---|
| CPU/model/stepping | |
| Microcode | |
| OS/kernel | |
| Compiler/version | |
| Build type/flags | |
| Effective CPU features | |
| Threads/affinity | |
| NUMA binding | |
| Governor/turbo/SMT | |
| Memory configuration | |

## 4. Correctness 与安全

| 维度 | 覆盖 | 结果 | 证据 |
|---|---|---|---|
| canonical/boundary | | PASS/FAIL/NOT RUN | |
| dtype/shape/stride | | | |
| alias/in-place/injectivity | | | |
| zero-size/null/overflow | | | |
| numerical error | atol/rtol/max error | | |
| fallback/feature policy | | | |
| allocation/workspace | | | |
| concurrency | | | |

未运行项必须解释原因，不能省略。

## 5. Benchmark 协议

- benchmark 层级：microkernel / prepared operator / preparation / integration / end-to-end；
- shape/layout/cache/thread matrix；
- warmup/min_time/repetitions；
- baseline/candidate 交错次序；
- correctness guard 的位置；
- timed loop 包含/排除内容；
- 运行命令。

```bash
# exact commands
```

## 6. 性能结果

### 6.1 Canonical cases

| Case | Baseline | Candidate | Delta | CI/variance | Gate |
|---|---:|---:|---:|---|---|
| | | | | | PASS/FAIL |

### 6.2 Boundary/fallback cases

| Case | 结果 | 备注 |
|---|---:|---|
| | | |

### 6.3 Preparation/packing/binding

| 项目 | 成本 | Break-even/影响 |
|---|---:|---|
| | | |

## 7. 硬件计数器与反汇编

- cycles/instructions/IPC；
- cache/TLB/branch；
- arithmetic events；
- spill、目标指令和意外调用；
- 只记录与结论直接相关的摘要，完整输出保存在 raw artifact。

## 8. Integration 与端到端

- ExecutionPlan/binding/workspace；
- steady-state allocation；
- Prefill/Decode 或产品场景；
- 尚不具备端到端条件时明确写 `Not Available`。

## 9. 事实、推断与限制

**已验证事实**：

-

**基于数据的推断**：

-

**限制和可能失效范围**：

-

## 10. 门禁判定

- [ ] correctness 与 safety；
- [ ] production-path benchmark；
- [ ] 无未解释 canonical regression；
- [ ] preparation/packing 成本可接受；
- [ ] allocation/workspace/ownership；
- [ ] dispatch/fallback/portability；
- [ ] integration/end-to-end（适用时）；
- [ ] 原始数据和环境可追溯。

## 11. 后续动作

- descriptor priority：保持 / 提高 / 回退；
- proposal 状态更新：
- design/ADR/CHANGELOG 更新：
- 未完成问题：

