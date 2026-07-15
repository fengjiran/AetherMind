# Agent Memory System 自动化测试方案（精炼版）

> 目的：在保留的核心结构基础上，加入 frontmatter 扫描/选单一正文、引导操作与确认闸门状态机覆盖。
> 不影响现有 `docs/agent/tests/memory_system_test_suite.md`。
> 初始版本（`memory_system_automation_plan_initial.md`）为历史源文本，其中的 README-first 伪代码已被当前根契约取代。当前默认启动路径为 `AGENTS.md → project.md → 选中的 handoff 正文`，README.md/module.md/submodule.md 均为按需升级读取。

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

### 修订 C：README-first 伪代码已过时

- 问题：初始版（`memory_system_automation_plan_initial.md`）中的 L2 伪代码以 `README.md` 为默认启动第一步，且要求所有加载路径都包含 README.md。
- 修订：当前根契约 `AGENTS.md` 第12节定义默认启动为 `AGENTS.md → project.md → 选中的 handoff 正文`，`README.md`、`module.md`、`submodule.md` 均为按需升级读取。初始版伪代码仅保留作历史参考，不作为实现依据。

### 修订 D：v1.1 frontmatter 契约显式化

- 修订要求：
  - 模块 workstream：`slug: null` 必须存在。
  - 项目级 workstream：`module: project`、`submodule: null`、`slug` 非空字符串。

### 修订 E：`supersedes` 完整性校验补强

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
- **Frontmatter 扫描契约**：确认实现扫描所有候选 frontmatter（仅元数据），排除无效候选，筛选 active，按 created_at 降序 + 文件名 tie-break 排序，只读选中的第一个正文。元数据扫描不计为加载多个正文。

### L3：加载与语义行为测试

- 模块/项目级加载顺序与跳过规则：
  - 默认启动路径为 `AGENTS.md → project.md → 选中的 handoff 正文`。
  - 项目级工作默认跳过 `module.md` 和 `submodule.md`；按需读取受影响的模块 memory 时须在 Resume Gate 中记录。
  - `README.md` 仅作为按需升级读取，不在默认启动路径中。
- `resume_status` 判定（`complete/partial/blocked`）：
  - `project.md` 缺失 → `blocked`（与当前规则一致）。
- **引导操作与确认闸门状态机**：
  - Resume Gate 前允许根目录 `AGENTS.md` 第12节列出的只读引导操作（解析类型、列出候选路径、读取 AGENTS.md 和 project.md、扫描候选 frontmatter、读取唯一选中 handoff 正文、按需升级读取 memory 文档）。
  - 禁止非恢复/业务工具操作（代码扫描、构建、测试、编辑、写入、外部副作用）。
  - 第一轮"继续" = workstream 选择请求；第二轮用户明确确认后才授权执行业务操作。
- 确认词白名单：`继续`、`执行`、`是`；模糊反馈（`好的`、`嗯`）不放行。

### L4：场景集成测试（fixture）

- 完整恢复（module/project）。
- 缺失子模块 memory。
- 缺失模块。
- 缺失 `project.md` → `blocked`。
- 多个 active handoff（frontmatter 全部扫描，只读最新 active 正文，告警）。
- Frontmatter 损坏文件（扫描时排除，不影响有效 handoff 加载）。
- 第一轮"继续"触发引导操作，第二轮确认后执行业务操作。
- 项目级工作按需读取受影响的模块 memory 并在 Resume Gate 中记录。

## 4. 最小可落地回归集（建议先实现）

先上线以下检查，成本低、收益高：

1. frontmatter 提取 + schema 校验。
2. 每个 workstream `active` 唯一性。
3. `supersedes` 目标存在 + `superseded` 状态校验。
4. 项目级/模块级加载路径语义检查（文档断言）—— 默认启动为 AGENTS.md → project.md → 选中的 handoff，README.md 为按需升级。
5. 缺失 `project.md` → `blocked` 场景测试。
6. Frontmatter 扫描契约：扫描所有候选 frontmatter，排除无效，筛选 active，排序，只读选中第一个正文。
7. 引导操作与确认闸门状态机：Resume Gate 前允许只读引导操作，禁止非恢复/业务工具操作；区分第一轮"继续"（选择 workstream）和第二轮确认（执行业务操作）。

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
