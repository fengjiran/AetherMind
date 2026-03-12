# Agent Memory System v1.1 测试样例

> 用途：验证记忆系统的可用性、准确性和防错性
> 来源：Oracle审核生成
> 更新时间：2026-03-12

## 测试概览

- **总计**：34个测试样例
- **核心回归**：10个（每次修改后必测）
- **自动化潜力**：高（29个）/ 中（5个）

---

## 工作流解析测试（10个）

### TEST-001: 显式项目级指令
```yaml
id: TEST-001
category: 工作流解析
name: 显式项目级指令
priority: P0
regression_core: true
automation_potential: high
description: 用户直接提供合法的 project workstream key。
preconditions:
  - "存在 `docs/agent/memory/project.md`"
  - "存在目录 `docs/agent/handoff/workstreams/project__agent-memory-v1.1/`"
input: "继续 project__agent-memory-v1.1 的工作"
expected_behavior:
  - "解析为项目级工作"
  - "resolved_workstream_key 为 `project__agent-memory-v1.1`"
  - "加载 `AGENTS.md -> docs/agent/memory/README.md -> docs/agent/memory/project.md -> docs/agent/handoff/workstreams/project__agent-memory-v1.1/`"
  - "明确跳过任何 `module.md` 和 `submodule.md`"
  - "输出 Resume Gate，并在加载后询问用户确认"
validation_criteria:
  - "未读取 `docs/agent/memory/modules/**/module.md`"
  - "未读取 `docs/agent/prompts/handoff_template.md` 作为 handoff"
  - "除记忆加载所需读取外，未调用扫描、编译、测试、编辑等后续工具"
edge_cases:
  - "若无 active handoff：`resume_status` 为 `partial`，并说明“从 memory 开始”"
  - "若 `project.md` 缺失：保持 `partial`，而非错误地直接 `blocked`"
failure_mode: "最可能是把 `project__<slug>` 当成普通文本处理，或错误加载了模块级 memory"
```

### TEST-002: 显式模块+子模块指令
```yaml
id: TEST-002
category: 工作流解析
name: 显式模块+子模块指令
priority: P0
regression_core: true
automation_potential: high
description: 用户明确指定真实存在的模块和子模块
preconditions:
  - "存在 `docs/agent/memory/modules/ammalloc/module.md`"
  - "存在 `docs/agent/memory/modules/ammalloc/submodules/thread_cache.md`"
input: "继续 ammalloc thread_cache 的工作"
expected_behavior:
  - "解析为模块工作"
  - "resolved_workstream_key 为 `ammalloc__thread_cache`"
  - "按固定顺序加载 project/module/submodule/handoff"
  - "输出 Resume Gate，并在加载后等待确认"
validation_criteria:
  - "不应把该请求错误映射成项目级工作"
  - "实际加载文件清单包含完整路径"
  - "确认前无后续业务工具调用"
edge_cases:
  - "若 handoff 目录为空：`resume_status` 为 `partial`"
failure_mode: "最可能是模块解析成功，但 workstream key 拼接错误或漏读 submodule"
```

### TEST-003: 仅模块名的默认推断
```yaml
id: TEST-003
category: 工作流解析
name: 仅模块名的默认推断
priority: P1
regression_core: false
automation_potential: high
description: 用户只给出模块名，系统应按模块级恢复，不自动扫所有子模块
preconditions:
  - "存在 `docs/agent/memory/modules/ammalloc/module.md`"
input: "继续 ammalloc"
expected_behavior:
  - "解析为模块工作"
  - "resolved_workstream_key 为 `ammalloc__none`"
  - "加载 `module.md`，但不自动读取所有 `submodules/*.md`"
  - "优先检查 `docs/agent/handoff/workstreams/ammalloc__none/`"
validation_criteria:
  - "没有对子模块目录执行全量扫描式加载"
  - "输出中明确“子模块：无”"
edge_cases:
  - "若 `ammalloc__none` 无 handoff，但某个子模块有 handoff，也不应自动切过去"
failure_mode: "最可能是把模糊模块恢复错误升级成多子模块聚合恢复"
```

### TEST-004: 自然语言映射到 docs-reorg
```yaml
id: TEST-004
category: 工作流解析
name: 自然语言映射到 docs-reorg
priority: P0
regression_core: true
automation_potential: high
description: 用户使用自然语言描述项目级工作，且 slug 可由映射表唯一确定
preconditions:
  - "`docs/agent/memory/project.md` 的 slug 表包含 `docs-reorg`"
input: "继续 docs 目录重构"
expected_behavior:
  - "解析为项目级工作"
  - "通过 slug 映射表解析到 `project__docs-reorg`"
  - "跳过模块/子模块 memory"
  - "输出中明确说明 slug 来源于映射表"
validation_criteria:
  - "不能临时发明新的 slug"
  - "不能要求用户二次确认一个已经唯一映射的 slug"
edge_cases:
  - "若映射表没有该项：必须改为询问，而不是猜测"
failure_mode: "最可能是没有先读 `project.md` 中的 slug 映射表，导致猜测 slug"
```

### TEST-005: 自然语言映射到 agent-memory-v1.1
```yaml
id: TEST-005
category: 工作流解析
name: 自然语言映射到 agent-memory-v1.1
priority: P1
regression_core: false
automation_potential: high
description: 验证项目级自然语言可唯一落到当前定义的 `agent-memory-v1.1` slug
preconditions:
  - "`docs/agent/memory/project.md` 的 slug 表包含 `agent-memory-v1.1`"
input: "继续 Agent Memory System v1.1 设计冻结"
expected_behavior:
  - "解析为 `project__agent-memory-v1.1`"
  - "按项目级顺序加载"
  - "在输出中保留用户原意与解析后的 slug"
validation_criteria:
  - "不能错误归为某个模块工作"
  - "不能把 `v1.1` 截断或改写为其他 slug"
edge_cases:
  - "若用户写成 `agent memory v1.1` 但映射仍唯一，可接受"
failure_mode: "最可能是 slug 正规化逻辑过于激进，破坏了带版本号的 key"
```

### TEST-006: 自然语言映射到待创建项目级 slug
```yaml
id: TEST-006
category: 工作流解析
name: 自然语言映射到待创建项目级 slug
priority: P1
regression_core: false
automation_potential: high
description: slug 已在映射表注册但尚无 handoff，也应正确解析
preconditions:
  - "`docs/agent/memory/project.md` 的 slug 表包含 `ci-setup`"
  - "不存在 active handoff 于 `docs/agent/handoff/workstreams/project__ci-setup/`"
input: "继续 CI/CD 配置迁移"
expected_behavior:
  - "解析为 `project__ci-setup`"
  - "读取 `project.md` 后进入 handoff 目录检查"
  - "`resume_status` 为 `partial`，并说明暂无可恢复临时状态"
validation_criteria:
  - "不能因 handoff 不存在而把解析结果改成 `blocked`"
  - "不能发明新的 slug，如 `project__cicd-migration`"
edge_cases:
  - "若用户显式写 `project__ci-setup`，应直接采用"
failure_mode: "最可能是把“待创建”误解成非法 slug"
```

### TEST-007: 子模块歧义请求
```yaml
id: TEST-007
category: 工作流解析
name: 子模块歧义请求
priority: P0
regression_core: true
automation_potential: high
description: 用户只说“cache”，存在多个真实候选子模块，应强制澄清
preconditions:
  - "存在 `ammalloc/central_cache.md`"
  - "存在 `ammalloc/page_cache.md`"
  - "存在 `ammalloc/thread_cache.md`"
input: "继续 cache 的工作"
expected_behavior:
  - "识别为歧义场景"
  - "列出候选，如 `ammalloc/central_cache`、`ammalloc/page_cache`、`ammalloc/thread_cache`"
  - "要求用户明确指定"
  - "在澄清前不进入任何 handoff 加载"
validation_criteria:
  - "不能默认挑一个最像的子模块"
  - "不能在未消歧前输出已完成的 Resume Gate"
edge_cases:
  - "若项目后来新增更多 `*cache*` 子模块，候选列表应同步扩展"
failure_mode: "最可能是模糊匹配策略越界，自动命中单个子模块"
```

### TEST-008: 不存在模块的恢复请求
```yaml
id: TEST-008
category: 工作流解析
name: 不存在模块的恢复请求
priority: P0
regression_core: false
automation_potential: high
description: 用户请求恢复一个当前仓库无对应 module 目录的模块
preconditions:
  - "不存在 `docs/agent/memory/modules/tensor/`"
input: "继续 tensor 的工作"
expected_behavior:
  - "识别为模块不存在"
  - "`resume_status` 为 `blocked`"
  - "给出选项：创建新模块、指定现有模块、检查拼写"
  - "不继续读取不存在模块的 handoff"
validation_criteria:
  - "不能凭空创建 `tensor` 的 memory"
  - "不能错误退化成项目级工作"
edge_cases:
  - "若用户原句含“开始开发 tensor”，可默认推荐新建模块流程，但仍需确认"
failure_mode: "最可能是模块存在性检查缺失，导致继续走后续加载流程"
```

### TEST-009: 非法 project key 格式
```yaml
id: TEST-009
category: 工作流解析
name: 非法 project key 格式
priority: P1
regression_core: false
automation_potential: medium
description: 用户给出看似 project key 但格式不符合 `project__<slug>`
preconditions: []
input: "继续 project_agent-memory-v1.1 的工作"
expected_behavior:
  - "不把 `project_agent-memory-v1.1` 当成合法 workstream key"
  - "提示合法格式应为 `project__<slug>`，并要求用户确认"
  - "在澄清前不执行加载"
validation_criteria:
  - "不能私自把单下划线修正为双下划线并继续"
  - "不能输出错误的 resolved_workstream_key"
edge_cases:
  - "若系统支持容错建议，可展示 `project__agent-memory-v1.1` 作为建议值"
failure_mode: "最可能是 key 校验太宽松，接受了非法格式"
```

### TEST-010: 显式模块 workstream key
```yaml
id: TEST-010
category: 工作流解析
name: 显式模块 workstream key
priority: P1
regression_core: false
automation_potential: medium
description: 用户直接使用磁盘 workstream key 恢复模块工作
preconditions:
  - "存在 `docs/agent/memory/modules/ammalloc/module.md`"
input: "继续 ammalloc__none 的工作"
expected_behavior:
  - "解析为模块工作"
  - "模块为 `ammalloc`，子模块为 `无`"
  - "检查 `docs/agent/handoff/workstreams/ammalloc__none/`"
validation_criteria:
  - "不能把 `ammalloc__none` 误当作子模块名"
  - "不能因为 `none` 是字面值而继续搜寻 `submodules/none.md`"
edge_cases:
  - "若用户输入 `ammalloc__thread_cache`，应映射到具体子模块工作"
failure_mode: "最可能是 workstream key 拆分规则没有覆盖模块型磁盘键"
```

---

## 加载顺序测试（8个）

### TEST-011: 模块工作完整加载
```yaml
id: TEST-011
category: 加载顺序
name: 模块工作完整加载
priority: P0
regression_core: true
automation_potential: high
description: 所有必要文件和 active handoff 均存在时，必须按固定顺序完整加载
preconditions:
  - "存在 `docs/agent/memory/modules/ammalloc/module.md`"
  - "存在 `docs/agent/memory/modules/ammalloc/submodules/thread_cache.md`"
  - "存在 active handoff 于 `docs/agent/handoff/workstreams/ammalloc__thread_cache/`"
input: "继续 ammalloc thread_cache 的工作"
expected_behavior:
  - "加载顺序严格为 `AGENTS.md -> docs/agent/memory/README.md -> docs/agent/memory/project.md -> docs/agent/memory/modules/ammalloc/module.md -> docs/agent/memory/modules/ammalloc/submodules/thread_cache.md -> docs/agent/handoff/workstreams/ammalloc__thread_cache/`"
  - "`resume_status` 为 `complete`"
validation_criteria:
  - "任何顺序颠倒都判失败"
  - "不能在 `README.md` 之前直接读取 handoff"
edge_cases:
  - "若 handoff 有多个 active，仍应先完成 memory 层加载，再做 handoff 选择"
failure_mode: "最可能是实现走了捷径，先查 handoff 再补读 memory"
```

### TEST-012: 模块级无子模块加载
```yaml
id: TEST-012
category: 加载顺序
name: 模块级无子模块加载
priority: P1
regression_core: false
automation_potential: high
description: 模块工作不指定子模块时，只加载 module 层和 `__none` handoff
preconditions:
  - "存在 `docs/agent/memory/modules/ammalloc/module.md`"
  - "存在 active handoff 于 `docs/agent/handoff/workstreams/ammalloc__none/`"
input: "继续 ammalloc"
expected_behavior:
  - "加载 `module.md`"
  - "不读取任何 `submodules/*.md`"
  - "读取 `docs/agent/handoff/workstreams/ammalloc__none/`"
validation_criteria:
  - "不能自动扫描并读取全部子模块 memory"
  - "Resume Gate 的预期文件列表不应包含具体子模块文件"
edge_cases:
  - "若 `ammalloc__none` 无 handoff，则保持 `partial`"
failure_mode: "最可能是把模块级恢复实现成了模块+全部子模块聚合恢复"
```

### TEST-013: 子模块文件缺失时的 partial
```yaml
id: TEST-013
category: 加载顺序
name: 子模块文件缺失时的 partial
priority: P0
regression_core: false
automation_potential: high
description: 模块存在但目标子模块 memory 缺失时，应按 partial 恢复
preconditions:
  - "存在 `docs/agent/memory/modules/ammalloc/module.md`"
  - "不存在 `docs/agent/memory/modules/ammalloc/submodules/new_bucket.md`"
input: "继续 ammalloc new_bucket 的工作"
expected_behavior:
  - "先加载 `module.md`"
  - "发现子模块 memory 缺失后标记 `partial`"
  - "提示用户可按模块层继续或创建子模块 memory"
validation_criteria:
  - "不能直接判 `blocked`"
  - "不能在 Resume Gate 中把缺失子模块写成已加载"
edge_cases:
  - "若 handoff 存在但子模块 memory 缺失，仍应是 `partial`，不是 `complete`"
failure_mode: "最可能是把缺失子模块误判为不存在模块"
```

### TEST-014: 模块文件缺失时的 blocked
```yaml
id: TEST-014
category: 加载顺序
name: 模块文件缺失时的 blocked
priority: P0
regression_core: false
automation_potential: high
description: 目标模块目录或 `module.md` 缺失时，应在模块层直接阻塞
preconditions:
  - "不存在 `docs/agent/memory/modules/ghost/module.md`"
input: "继续 ghost 的工作"
expected_behavior:
  - "在读取 `project.md` 后，模块层检查失败"
  - "`resume_status` 为 `blocked`"
  - "不读取 `docs/agent/handoff/workstreams/ghost__none/`"
validation_criteria:
  - "不能因为 handoff 目录可能存在就绕过模块校验"
  - "不能输出“已加载 module.md”"
edge_cases:
  - "若用户改为“开始 ghost 开发”，可进入新建模块建议，但仍非 quick resume 成功"
failure_mode: "最可能是 handoff 优先级高于 module 存在性检查"
```

### TEST-015: 项目级完整加载并跳过模块文档
```yaml
id: TEST-015
category: 加载顺序
name: 项目级完整加载并跳过模块文档
priority: P0
regression_core: true
automation_potential: high
description: 项目级工作必须直接从 project 层进入 handoff，且显式跳过模块文档
preconditions:
  - "存在 `docs/agent/memory/project.md`"
  - "存在 active handoff 于 `docs/agent/handoff/workstreams/project__docs-reorg/`"
input: "继续 project__docs-reorg 的工作"
expected_behavior:
  - "加载顺序为 `AGENTS.md -> docs/agent/memory/README.md -> docs/agent/memory/project.md -> docs/agent/handoff/workstreams/project__docs-reorg/`"
  - "输出中明确列出 `module.md` 和 `submodule.md` 为跳过项"
validation_criteria:
  - "任何模块级读取都判失败"
  - "不得把 `docs/agent/prompts/handoff_template.md` 当作 handoff 读取"
edge_cases:
  - "若 handoff 为空：状态应为 `partial`"
failure_mode: "最可能是项目级和模块级路径共用一套代码，忘记跳过 module/submodule"
```

### TEST-016: 项目级无 handoff 时从 memory 开始
```yaml
id: TEST-016
category: 加载顺序
name: 项目级无 handoff 时从 memory 开始
priority: P1
regression_core: false
automation_potential: high
description: 项目级工作没有 active handoff 时，应从 `project.md` 的待办事项起步
preconditions:
  - "存在 `docs/agent/memory/project.md`"
  - "不存在 active handoff 于 `docs/agent/handoff/workstreams/project__ci-setup/`"
input: "继续 project__ci-setup 的工作"
expected_behavior:
  - "完成到 `project.md` 的加载"
  - "`resume_status` 为 `partial`"
  - "输出“无可恢复临时状态，从 memory 的待办事项开始”"
validation_criteria:
  - "不能因为 handoff 缺失而转为 `blocked`"
  - "不能尝试读取其他 project slug 的 handoff 作为替代"
edge_cases:
  - "若目录存在但全是 `closed`/`superseded`，结果仍应等价于无 active handoff"
failure_mode: "最可能是 handoff 过滤逻辑只检查目录存在，不检查 active 状态"
```

### TEST-017: project.md 缺失时的降级处理
```yaml
id: TEST-017
category: 加载顺序
name: project.md 缺失时的降级处理
priority: P1
regression_core: false
automation_potential: medium
description: 全局 memory 缺失时，不应把整个 quick resume 直接视为不可恢复
preconditions:
  - "不存在 `docs/agent/memory/project.md`"
  - "存在 active handoff 于 `docs/agent/handoff/workstreams/project__agent-memory-v1.1/`"
input: "继续 project__agent-memory-v1.1 的工作"
expected_behavior:
  - "标记 `project.md` 缺失"
  - "`resume_status` 为 `partial`"
  - "仍可读取 handoff 并向用户说明全局基线缺失"
validation_criteria:
  - "不能错误判为 `blocked`"
  - "Resume Gate 的缺失项必须明确写出 `docs/agent/memory/project.md`"
edge_cases:
  - "若 handoff 也缺失，则仍是 `partial`，但内容更少"
failure_mode: "最可能是把任意上层 memory 缺失一律判成阻塞"
```

### TEST-018: 模板路径误用检测
```yaml
id: TEST-018
category: 加载顺序
name: 模板路径误用检测
priority: P0
regression_core: false
automation_potential: high
description: 防止把 `handoff_template.md` 当成真实 handoff 读取
preconditions:
  - "存在 `docs/agent/prompts/handoff_template.md`"
  - "不存在 active handoff 于目标 workstream 目录"
input: "继续 project__docs-reorg 的工作"
expected_behavior:
  - "只检查 `docs/agent/handoff/workstreams/project__docs-reorg/`"
  - "若无 active handoff，则标记 `partial`"
  - "明确说明模板文件不是 handoff 存储"
validation_criteria:
  - "`docs/agent/prompts/handoff_template.md` 绝不能出现在“实际加载文件”中"
  - "handoff 路径字段必须指向 `docs/agent/handoff/workstreams/project__docs-reorg/`"
edge_cases:
  - "即使模板更新时间更近，也不影响选择"
failure_mode: "最可能是路径常量复用错误，把 prompt 模板目录当作存储目录"
```

---

## Resume Gate 测试（6个）

### TEST-019: 模块工作字段完整性
```yaml
id: TEST-019
category: Resume Gate
name: 模块工作字段完整性
priority: P0
regression_core: true
automation_potential: high
description: 模块工作输出的 Resume Gate 必须完整且字段名固定
preconditions:
  - "目标为 `ammalloc__thread_cache`"
input: "继续 ammalloc thread_cache 的工作"
expected_behavior:
  - "Resume Gate 至少包含：工作流类型、resolved_workstream_key、预期加载文件、实际加载文件、跳过/缺失项、resume_status"
  - "同时包含 repo 规则要求的 `handoff 路径` 行"
validation_criteria:
  - "字段缺任一项即失败"
  - "字段名不可被同义词替换"
edge_cases:
  - "无缺失文件时，`跳过/缺失项` 也必须显式输出"
failure_mode: "最可能是输出模板不统一，某些字段在成功路径被省略"
```

### TEST-020: 项目级工作字段完整性与跳过说明
```yaml
id: TEST-020
category: Resume Gate
name: 项目级工作字段完整性与跳过说明
priority: P0
regression_core: false
automation_potential: high
description: 项目级 Resume Gate 除字段完整外，还必须显式标明模块文档为跳过项
preconditions:
  - "目标为 `project__docs-reorg`"
input: "继续 project__docs-reorg 的工作"
expected_behavior:
  - "Resume Gate 字段完整"
  - "`跳过/缺失项` 中包含 `module.md` 与 `submodule.md` 的项目级跳过说明"
  - "`handoff 路径` 指向 `docs/agent/handoff/workstreams/project__docs-reorg/`"
validation_criteria:
  - "不能把跳过项写成已加载"
  - "不能省略 handoff 存储目录说明"
edge_cases:
  - "若无 handoff，跳过项和缺失项都应同时出现"
failure_mode: "最可能是复用了模块工作模板，导致项目级跳过信息缺失"
```

### TEST-021: complete 状态判定
```yaml
id: TEST-021
category: Resume Gate
name: complete 状态判定
priority: P1
regression_core: false
automation_potential: high
description: 当所有预期文件和 active handoff 都存在时，应判为 complete
preconditions:
  - "模块、子模块、active handoff 全部存在"
input: "继续 ammalloc central_cache 的工作"
expected_behavior:
  - "`resume_status` 为 `complete`"
  - "实际加载文件列表与预期文件列表一致"
validation_criteria:
  - "不能在 complete 场景输出 `partial`"
  - "不能因为存在旧的 `closed` handoff 就改变状态"
edge_cases:
  - "若存在多个 active，状态仍可为 `complete`，但需警告异常"
failure_mode: "最可能是状态机把所有 handoff 异常都一概降为 partial"
```

### TEST-022: partial 状态判定
```yaml
id: TEST-022
category: Resume Gate
name: partial 状态判定
priority: P0
regression_core: false
automation_potential: high
description: 缺失 handoff 或缺失子模块 memory 时，应统一归入 partial
preconditions:
  - "存在 `ammalloc/module.md`"
  - "不存在 active handoff 于 `ammalloc__page_allocator/`"
input: "继续 ammalloc page_allocator 的工作"
expected_behavior:
  - "`resume_status` 为 `partial`"
  - "缺失原因写入 `跳过/缺失项`"
  - "仍给出推荐下一步"
validation_criteria:
  - "不能错误判为 `complete` 或 `blocked`"
  - "输出必须说明是“部分恢复”而不是“恢复失败”"
edge_cases:
  - "若同时缺失 handoff 和子模块文件，仍是 `partial`"
failure_mode: "最可能是 partial 和 blocked 的判定边界不清"
```

### TEST-023: blocked 状态判定
```yaml
id: TEST-023
category: Resume Gate
name: blocked 状态判定
priority: P0
regression_core: true
automation_potential: high
description: 模块不存在时应进入 blocked，且禁止后续执行
preconditions:
  - "不存在模块目录 `docs/agent/memory/modules/ghost/`"
input: "继续 ghost 的工作"
expected_behavior:
  - "`resume_status` 为 `blocked`"
  - "Resume Gate 完成后停止在澄清/选择阶段"
  - "不推荐任何直接执行型下一步"
validation_criteria:
  - "blocked 后不得继续读取代码、运行命令或修改文件"
  - "不得伪造 `ghost__none` 的实际加载记录"
edge_cases:
  - "若用户后续明确选择创建新模块，才可切入新会话模板流程"
failure_mode: "最可能是 blocked 只写在文案里，但执行器没有真正拦截后续动作"
```

### TEST-024: 跳过项与缺失项标注准确
```yaml
id: TEST-024
category: Resume Gate
name: 跳过项与缺失项标注准确
priority: P1
regression_core: false
automation_potential: high
description: 跳过和缺失是不同语义，输出不得混淆
preconditions:
  - "目标为项目级 `project__agent-memory-v1.1`"
  - "不存在 active handoff"
input: "继续 project__agent-memory-v1.1 的工作"
expected_behavior:
  - "`module.md`、`submodule.md` 标记为跳过"
  - "handoff 目录或 active handoff 缺失标记为缺失"
  - "最终 `resume_status` 为 `partial`"
validation_criteria:
  - "不能把项目级跳过项误写成缺失错误"
  - "不能把真正缺失的 handoff 写成跳过"
edge_cases:
  - "模块工作时缺少子模块文件则应记为缺失，而非跳过"
failure_mode: "最可能是输出模板只支持一个统一的“异常项”标签"
```

---

## Handoff 管理测试（6个）

### TEST-025: 选择最新 active handoff
```yaml
id: TEST-025
category: Handoff管理
name: 选择最新 active handoff
priority: P0
regression_core: false
automation_potential: high
description: 同一 workstream 下同时存在 active、closed、superseded 时，只读取最新 active
preconditions:
  - "同目录内存在 1 个 active、1 个 closed、1 个 superseded handoff"
  - "active 的 `created_at` 早于 closed，但状态合法"
input: "继续 ammalloc thread_cache 的工作"
expected_behavior:
  - "只读取 `status: active` 的 handoff"
  - "忽略 `closed` 和 `superseded`"
  - "用 active 内容生成当前接续目标"
validation_criteria:
  - "不能选时间更新更晚但状态为 `closed` 的文件"
  - "实际加载文件列表中只能出现 active handoff"
edge_cases:
  - "若没有 active，则应退化为“从 memory 开始”"
failure_mode: "最可能是排序先于状态过滤执行"
```

### TEST-026: 多个 active handoff 异常处理
```yaml
id: TEST-026
category: Handoff管理
name: 多个 active handoff 异常处理
priority: P0
regression_core: true
automation_potential: high
description: 同一 workstream 存在多个 active 时，应选择最新一个并显式告警
preconditions:
  - "同一目录内存在两个 `status: active` handoff，且 `created_at` 不同"
input: "继续 ammalloc page_cache 的工作"
expected_behavior:
  - "选择 `created_at` 最新的 active handoff"
  - "输出告警，提示用户检查并收敛状态"
  - "建议将旧文件标记为 `superseded`"
validation_criteria:
  - "不能静默选择"
  - "不能把多个 active 拼接合并"
edge_cases:
  - "若最新 active frontmatter 损坏，则应忽略它并回退到下一个合法 active，同时继续告警"
failure_mode: "最可能是实现假设“最多一个 active”，碰到异常时行为不确定"
```

### TEST-027: frontmatter 损坏文件忽略
```yaml
id: TEST-027
category: Handoff管理
name: frontmatter 损坏文件忽略
priority: P1
regression_core: false
automation_potential: high
description: frontmatter 缺失或损坏的 handoff 文件不能参与恢复
preconditions:
  - "目录中存在一个损坏 frontmatter 文件和一个合法 active handoff"
input: "继续 ammalloc size_class 的工作"
expected_behavior:
  - "忽略损坏文件"
  - "读取合法 active handoff"
  - "可选地提示存在损坏文件"
validation_criteria:
  - "损坏文件不应出现在实际加载文件列表"
  - "不能因损坏文件而整体失败"
edge_cases:
  - "若所有 handoff 都损坏，则退化为无 active handoff 的 `partial`"
failure_mode: "最可能是对目录中文件逐个盲读，缺少 frontmatter 校验"
```

### TEST-028: schema 1.0 向后兼容
```yaml
id: TEST-028
category: Handoff管理
name: schema 1.0 向后兼容
priority: P1
regression_core: false
automation_potential: high
description: Reader 必须接受 `schema_version: 1.0` 并补默认字段
preconditions:
  - "目标目录中存在合法 `schema_version: \"1.0\"` handoff"
  - "该文件缺少 `memory_status`、`supersedes`、`closed_at`、`closed_reason`"
input: "继续 ammalloc central_cache 的工作"
expected_behavior:
  - "成功读取该 handoff"
  - "对缺失字段使用默认值"
  - "不因旧 schema 而拒绝恢复"
validation_criteria:
  - "不能把 `1.0` 文件视为损坏或未知版本"
  - "默认值应与文档一致"
edge_cases:
  - "若同目录还存在更新的 1.1 active handoff，应按时间和状态规则优先选更新者"
failure_mode: "最可能是版本校验写成了只接受 1.1"
```

### TEST-029: 对话/任务系统 handoff 优先于磁盘
```yaml
id: TEST-029
category: Handoff管理
name: 对话/任务系统 handoff 优先于磁盘
priority: P1
regression_core: false
automation_potential: medium
description: 有任务系统或对话中附带的 handoff 时，应优先使用它，而非磁盘旧文件
preconditions:
  - "存在任务系统/对话提供的当前 workstream handoff"
  - "磁盘中也存在一个较旧 active handoff"
input: "继续 project__agent-memory-v1.1 的工作"
expected_behavior:
  - "优先采用任务系统/对话 handoff"
  - "磁盘 handoff 仅作为后备"
  - "输出中说明采用了哪个来源"
validation_criteria:
  - "不能无条件优先磁盘"
  - "实际加载文件列表应反映来源优先级"
edge_cases:
  - "若任务系统 handoff 无法解析，则可回退到磁盘 active handoff"
failure_mode: "最可能是实现只支持文件系统，不支持会话态 handoff"
```

### TEST-030: 模板与存储目录严格区分
```yaml
id: TEST-030
category: Handoff管理
name: 模板与存储目录严格区分
priority: P0
regression_core: false
automation_potential: high
description: handoff 模板只可作为写作参考，永不作为恢复数据源
preconditions:
  - "存在 `docs/agent/prompts/handoff_template.md`"
  - "目标 workstream 目录也存在合法 active handoff"
input: "继续 project__docs-reorg 的工作"
expected_behavior:
  - "只从 `docs/agent/handoff/workstreams/project__docs-reorg/` 读取 handoff"
  - "模板路径最多出现在说明文字中，不出现在实际加载列表中"
validation_criteria:
  - "如果模板文件参与排序或选择，直接失败"
  - "`handoff 路径` 字段必须是存储目录"
edge_cases:
  - "即使模板更新时间更近，也不影响选择"
failure_mode: "最可能是把‘handoff 模板’和‘handoff 存储’混成同一配置源"
```

---

## 用户确认机制测试（4个）

### TEST-031: 加载后禁止自动执行
```yaml
id: TEST-031
category: 用户确认机制
name: 加载后禁止自动执行
priority: P0
regression_core: true
automation_potential: high
description: 记忆加载完成后，Agent 必须停在推荐下一步并等待用户确认
preconditions:
  - "目标 workstream 可成功恢复到 `complete` 或 `partial`"
input: "继续 ammalloc thread_cache 的工作"
expected_behavior:
  - "完成记忆加载与 Resume Gate 输出"
  - "给出推荐下一步"
  - "明确询问“记忆已加载，是否执行[推荐操作]？”"
  - "在用户未确认前不执行任何后续业务工具"
validation_criteria:
  - "不得自动扫描代码、运行构建、运行测试或编辑文件"
  - "允许的读取仅限记忆加载本身"
edge_cases:
  - "若 `resume_status` 为 `blocked`，也不得绕过确认直接进入新建模块流程"
failure_mode: "最可能是把“快速恢复”误实现成“快速恢复并立即开工”"
```

### TEST-032: 明确确认词后才执行
```yaml
id: TEST-032
category: 用户确认机制
name: 明确确认词后才执行
priority: P0
regression_core: false
automation_potential: high
description: 只有明确确认词才解除执行闸门
preconditions:
  - "已完成一次合法 Resume Gate"
input:
  - "用户第一轮：继续 ammalloc central_cache 的工作"
  - "Agent 完成加载后"
  - "用户第二轮：继续"
expected_behavior:
  - "第二轮 `继续` 被识别为显式确认"
  - "此后才允许扫描代码、测试、编辑等"
validation_criteria:
  - "第一次用户输入中的“继续”不能被误当作加载后的确认"
  - "只有在 Agent 已经询问确认之后，第二轮确认才生效"
edge_cases:
  - "同义确认词 `执行`、`是` 也应通过"
failure_mode: "最可能是状态机没有区分‘恢复请求中的继续’和‘加载后的继续’"
```

### TEST-033: 模糊确认词不应放行
```yaml
id: TEST-033
category: 用户确认机制
name: 模糊确认词不应放行
priority: P1
regression_core: true
automation_potential: medium
description: “好的”“嗯”“收到”这类模糊反馈不应自动触发执行
preconditions:
  - "Agent 已完成记忆加载并进入等待确认状态"
input:
  - "用户第一轮：继续 project__docs-reorg 的工作"
  - "Agent 完成加载后"
  - "用户第二轮：好的"
expected_behavior:
  - "Agent 保持等待确认"
  - "再次提示需要明确输入“继续/执行/是”或等价明确指令"
  - "不执行后续工具"
validation_criteria:
  - "模糊确认词不应被视为执行许可"
  - "不得因为上下文明显就自动放行"
edge_cases:
  - "若产品后来决定扩大确认词集合，本测试应同步更新白名单"
failure_mode: "最可能是确认词匹配做成了宽松情感判断，而非明确白名单"
```

### TEST-034: 明确拒绝或改问不应执行
```yaml
id: TEST-034
category: 用户确认机制
name: 明确拒绝或改问不应执行
priority: P1
regression_core: true
automation_potential: medium
description: 用户在加载后提出“先别执行”或“先总结一下”时，应继续停在说明态
preconditions:
  - "Agent 已完成记忆加载并等待确认"
input:
  - "用户第一轮：继续 ammalloc size_class 的工作"
  - "Agent 完成加载后"
  - "用户第二轮：先别执行，先总结一下当前接续目标"
expected_behavior:
  - "Agent 只做说明或总结"
  - "保持未授权执行状态"
  - "若之后用户再明确确认，才允许执行"
validation_criteria:
  - "不能把任何后续自然语言都解释成默认继续"
  - "解释型响应中也不得偷偷触发业务工具"
edge_cases:
  - "若用户随后说“执行第 1 步”，此时可视为明确确认"
failure_mode: "最可能是确认闸门只检查‘是否有后续消息’，而不是检查消息意图"
```

---

## 核心回归测试集（10个必测项）

| ID | 测试名 | 回归原因 |
|----|--------|----------|
| TEST-001 | 显式项目级指令 | 项目级显式 key 的主入口 |
| TEST-002 | 显式模块+子模块指令 | 模块+子模块显式恢复的主入口 |
| TEST-004 | 自然语言映射到 docs-reorg | 自然语言到 project slug 的唯一映射 |
| TEST-007 | 子模块歧义请求 | 歧义处理不能猜 |
| TEST-011 | 模块工作完整加载 | 模块完整加载顺序的基线 |
| TEST-015 | 项目级完整加载并跳过模块文档 | 项目级必须跳过 module/submodule |
| TEST-019 | 模块工作字段完整性 | Resume Gate 字段完整性 |
| TEST-023 | blocked 状态判定 | blocked 必须真正阻断 |
| TEST-026 | 多个 active handoff 异常处理 | 多个 active handoff 的异常恢复 |
| TEST-031 | 加载后禁止自动执行 | 加载后禁止自动执行 |

---

## 使用说明

### 自动化测试建议

**高优先级自动化（P0 + automation_potential: high）**：
- TEST-001, TEST-002, TEST-004, TEST-007, TEST-011
- TEST-015, TEST-019, TEST-023, TEST-026, TEST-031

**中优先级自动化（P1 + automation_potential: high/medium）**：
- TEST-003, TEST-005, TEST-006, TEST-012, TEST-013
- TEST-016, TEST-018, TEST-020, TEST-021, TEST-022
- TEST-024, TEST-025, TEST-027, TEST-028, TEST-030

### 手动测试场景

- TEST-008（不存在模块）：需要创建/删除测试模块
- TEST-017（project.md 缺失）：需要临时删除文件
- TEST-029（任务系统 handoff）：需要模拟对话上下文

### 回归测试触发条件

每次修改以下文件后，必须运行核心回归测试集：
- `AGENTS.md`
- `docs/agent/memory/README.md`
- `docs/agent/prompts/quick_resume.md`
- `docs/agent/prompts/new_session_template.md`
- `docs/agent/prompts/handoff_template.md`
