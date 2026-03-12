# 记忆系统自动化测试方案（最初版原文）

## 1. 测试分层架构

```text
┌─────────────────────────────────────────┐
│  L3: 集成测试 (E2E Workflow)            │
│  - 完整恢复流程模拟                      │
│  - 跨文件引用验证                        │
├─────────────────────────────────────────┤
│  L2: 契约测试 (Contract Test)           │
│  - Frontmatter schema 验证              │
│  - 状态转换规则                          │
│  - 路径加载顺序                          │
├─────────────────────────────────────────┤
│  L1: 静态检查 (Lint/Structure)          │
│  - 文件命名规范                          │
│  - YAML 语法                             │
│  - 必填字段存在性                        │
└─────────────────────────────────────────┘
```

## 2. 各层测试方法

### L1: 静态结构检查

工具：`yamllint` + 自定义 Python 脚本

测试点：
- 文件名格式：`YYYYMMDDTHHMMSSZ--<session_id>--<agent_id>.md`
- Frontmatter 存在性：文件必须以 `---` 开头
- 必填字段检查：`kind`, `schema_version`, `created_at`, `status` 等
- YAML 语法有效性

示例规则：

```yaml
# .yamllint.yml
rules:
  document-start: {present: true}
  line-length: disable
  empty-lines: {max: 2}
```

### L2: 契约测试（核心）

方法：基于 JSON Schema 的字段验证 + 自定义业务规则引擎

#### 2.1 Frontmatter Schema 验证

为 v1.1 定义 JSON Schema：

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "oneOf": [
    {
      "description": "模块工作 handoff",
      "properties": {
        "module": {"type": "string", "not": {"const": "project"}},
        "submodule": {"type": ["string", "null"]},
        "slug": {"type": "null"},
        "status": {"enum": ["active", "superseded", "closed"]},
        "memory_status": {"enum": ["not_needed", "pending", "applied"]}
      },
      "required": ["kind", "schema_version", "module", "submodule", "slug"]
    },
    {
      "description": "项目级工作 handoff",
      "properties": {
        "module": {"const": "project"},
        "submodule": {"type": "null"},
        "slug": {"type": "string"},
        "status": {"enum": ["active", "superseded", "closed"]}
      },
      "required": ["kind", "schema_version", "module", "submodule", "slug"]
    }
  ]
}
```

#### 2.2 业务规则验证

| 规则 ID | 规则描述 | 验证方法 |
|---------|---------|---------|
| R001 | 同一 workstream 最多一个 `status: active` | 按目录分组统计 active 数量 |
| R002 | `supersedes` 指向的文件必须存在 | 文件系统存在性检查 |
| R003 | `supersedes` 目标必须已标记 `superseded` | 目标文件 frontmatter 检查 |
| R004 | `closed` 状态必须有 `closed_at` 和 `closed_reason` | 条件字段存在性检查 |
| R005 | `project.md` 的 slug 映射表必须包含所有 project__* 目录 | 目录与表格交叉验证 |

#### 2.3 加载顺序测试

方法：模拟 Agent 加载流程，验证读取顺序是否符合规范

```python
def test_load_order(workstream_key: str, expected_type: str):
    """
    验证给定 workstream 的加载顺序是否符合规范
    expected_type: 'module' | 'project'
    """
    load_sequence = []

    if not read_file("docs/agent/memory/README.md"):
        return Fail("必须加载 README.md")

    load_sequence.append("README.md")

    if not read_file("docs/agent/memory/project.md"):
        return Fail("必须加载 project.md")

    load_sequence.append("project.md")

    if expected_type == "module":
        module_file = f"docs/agent/memory/modules/{module}/module.md"
        if not read_file(module_file):
            return Fail(f"模块工作必须加载 {module_file}")
        load_sequence.append(module_file)

        if submodule:
            submodule_file = f"docs/agent/memory/modules/{module}/submodules/{submodule}.md"
            if read_file(submodule_file):
                load_sequence.append(submodule_file)

    elif expected_type == "project":
        pass

    handoff_dir = f"docs/agent/handoff/workstreams/{workstream_key}/"
    return Pass(load_sequence)
```

### L3: 集成测试（E2E）

方法：使用临时文件系统模拟完整工作流

测试场景：

| 场景 ID | 场景描述 | 预期结果 |
|---------|---------|---------|
| E001 | 正常恢复模块工作 | 正确加载 6 层文件，Resume Gate 通过 |
| E002 | 项目级工作跳过 module.md | Resume Gate 显示 skip，不加载 module 层 |
| E003 | 无 handoff 时从 memory 恢复 | resume_status = partial，从待办事项开始 |
| E004 | 多个 active handoff 冲突 | 警告并选择最新，建议收敛 |
| E005 | supersedes 链完整性 | 能追溯完整替代历史 |

实现方式：

```bash
mkdir -p /tmp/memory-test-{uuid}/docs/agent/{memory,handoff}
```

## 3. 测试触发机制

### 本地开发

```bash
# pre-commit hook
make memory-lint
make memory-contract
```

### CI/CD 集成

```yaml
name: Memory System Contract Test
on:
  push:
    paths:
      - 'docs/agent/**'
      - '.github/workflows/memory-system.yml'

jobs:
  contract-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: L1 - Static Check
        run: |
          yamllint docs/agent/handoff/workstreams/**/*.md
          python scripts/check_frontmatter.py

      - name: L2 - Contract Test
        run: python -m pytest tests/memory_contract/ -v

      - name: L3 - E2E Workflow Test
        run: python tests/memory_e2e/test_workflow.py
```

## 4. 测试数据管理

### fixtures 结构

```text
tests/fixtures/memory/
├── valid/
│   ├── module_handoff/
│   ├── project_handoff/
│   └── memory_files/
└── invalid/
    ├── missing_slug/
    ├── multiple_active/
    ├── broken_supersedes/
    └── invalid_status_flow/
```

### 生成策略

- 合法样本：从生产环境脱敏抽取
- 非法样本：基于变异测试故意破坏规则

## 5. 监控与回归

### 持续监控

- 每周自动生成 `memory-system-health-report.md`
- 监控指标：
  - active handoff 数量趋势
  - superseded 链平均长度
  - 缺失必填字段比例

### 回归测试触发条件

- 修改 `docs/agent/memory/README.md`
- 新增 workstream 目录
- handoff schema 版本升级

## 6. 关键成功因素

1. 契约即代码：将 JSON Schema 和规则清单纳入版本控制
2. 失败快速：L1 静态检查应尽量快
3. 可解释性：测试失败时输出明确修复建议
4. 向后兼容：v1.0 handoff 可通过 v1.1 宽松模式验证
