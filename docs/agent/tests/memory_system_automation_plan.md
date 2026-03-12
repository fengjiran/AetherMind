# Agent Memory System 自动化测试方案（初版保留）

> 目的：保留“最初自动化测试方案”的核心结构，并明确需要修订的点，作为后续落地实现蓝图。
> 不影响现有 `docs/agent/tests/memory_system_test_suite.md`。

## 1. 总体评价

初版方案方向正确，可作为实施基础：

- 分层测试结构合理：L1 静态检查、L2 契约测试、L3 集成/E2E。
- 核心约束覆盖到位：frontmatter 合规、`active` 唯一性、`supersedes` 链、加载顺序、确认闸门。
- 具备 CI 接入思路，适合逐步上线（先快检，再扩展场景）。

结论：**可用，但建议先做 4 项修订再落地**。

## 2. 必要修订（先改再实现）

### 修订 A：Markdown frontmatter 检查方式

- 问题：不能直接对 `*.md` 全文做 YAML lint，会产生高误报。
- 修订：先提取 frontmatter（首个 `--- ... ---`），再做 YAML/schema 校验。

### 修订 B：`project.md` 缺失语义统一

- 问题：初版中存在把 `project.md` 缺失判为 `partial` 的描述。
- 修订：统一为 `blocked`（与当前记忆系统规则一致）。

### 修订 C：v1.1 frontmatter 契约显式化

- 修订要求：
  - 模块 workstream：`slug: null` 必须存在。
  - 项目级 workstream：`module: project`、`submodule: null`、`slug` 非空字符串。

### 修订 D：`supersedes` 完整性校验补强

- 修订要求：
  - `supersedes` 非空时，目标文件必须存在于同一 workstream。
  - 目标文件状态必须是 `status: superseded`。

## 3. 分层自动化方案（保留初版结构）

### L1：静态与结构检查

- 目录命名规则：`<module>__<submodule-or-none>` / `project__<slug>`。
- 文件名规则：`YYYYMMDDTHHMMSSZ--<session_id>--<agent_id>.md`。
- frontmatter 存在性与基础可解析性。

### L2：契约测试（核心）

- Schema 校验（含 v1.0 向后兼容默认值）。
- `active` 唯一性。
- `supersedes` 存在性与状态链路一致性。
- `closed` 状态字段完整性（`closed_at`、`closed_reason`）。

### L3：加载与语义行为测试

- 模块/项目级加载顺序与跳过规则。
- `resume_status` 判定（`complete/partial/blocked`）。
- 确认闸门：加载后必须等待用户明确确认。

### L4：场景集成测试（fixture）

- 完整恢复（module/project）。
- 缺失子模块 memory。
- 缺失模块。
- 缺失 `project.md`。
- 多个 active handoff。

## 4. 最小可落地回归集（建议先实现）

先上线以下检查，成本低、收益高：

1. frontmatter 提取 + schema 校验。
2. 每个 workstream `active` 唯一性。
3. `supersedes` 目标存在 + `superseded` 状态校验。
4. 项目级/模块级加载路径语义检查（文档断言）。
5. 缺失 `project.md` -> `blocked` 场景测试。

## 5. 建议实现栈

- 语言：Python
- 测试框架：`pytest`
- 解析：`pyyaml`
- 契约：`jsonschema`

建议目录：

```text
tests/agent_memory/
  fixtures/
    valid/
    invalid/
  test_l1_structure.py
  test_l2_contract.py
  test_l3_semantics.py
  test_l4_scenarios.py
```

## 6. CI 接入建议

- 触发路径：`docs/agent/**`、`tests/agent_memory/**`。
- 阶段化执行：
  - Stage 1：L1+L2（快速失败）
  - Stage 2：L3+L4（语义与场景）

失败输出至少包含：规则 ID、文件路径、期望/实际、修复建议。
