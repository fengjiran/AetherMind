# AetherMind Agent Memory System Architecture（架构设计说明）

> 本文档描述 AetherMind 项目的智能代理（Agent）记忆系统架构设计，用于帮助读者理解系统设计原理、整体结构和工作流概览。
> 操作规范见: `docs/agent/memory/README.md`
>
> 版本: 1.1  
> 日期: 2026-03-10  
> 适用范围: AetherMind 大模型推理引擎项目

---

## 目录

1. [系统概述](#1-系统概述)
2. [架构设计](#2-架构设计)
3. [文件结构](#3-文件结构)
4. [工作流](#4-工作流)
5. [模板规范](#5-模板规范)
6. [使用指南](#6-使用指南)
7. [示例](#7-示例)
8. [附录](#8-附录)

---

## 1. 系统概述

### 1.1 设计目标

Agent Memory System 旨在解决以下问题：

- **上下文漂移**: 多会话间知识丢失，代理重复询问已确认事实
- **决策丢失**: 重要架构决策散落在对话中，难以追溯
- **可发现性差**: 大型项目中难以定位相关模块的约束和接口
- **冲突处理**: 多个代理修改同一模块时缺乏协调机制
- **知识沉淀**: 临时讨论未经验证即成为"事实"

### 1.2 核心原则

| 原则 | 描述 |
|------|------|
| **分层存储** | 稳定事实写入记忆文档，临时状态保留在 handoff |
| **单点事实** | 同一事实只保留一处主描述，避免重复 |
| **显式引用** | 依赖关系和ADR通过元数据显式追踪 |
| **版本控制** | 每次更新标注验证日期和责任人 |
| **冲突优先级** | 明确定义信息来源的优先级 |

### 1.3 文档定位

本文档是**记忆系统的架构设计说明**，面向希望理解系统设计原理和整体结构的读者。

**与操作文档的关系**:
- 本文档 ← **架构视角**: 设计原理、工作流概览、示例
- `docs/agent/memory/README.md` ← **操作规范**: 详细的规则、约束、元数据规范
- 实际使用时，以 `docs/agent/memory/README.md` 为最终操作依据

---

## 2. 架构设计

### 2.1 三层记忆模型 + Handoff 临时层

```
┌─────────────────────────────────────────┐
│           全局层 (Global)                │
│      docs/agent/memory/project.md             │
│  - 跨模块稳定事实                        │
│  - 目录约定、默认规则                     │
└───────────────────┬─────────────────────┘
                    │
┌───────────────────▼─────────────────────┐
│           模块层 (Module)                │
│   docs/agent/memory/modules/<module>/         │
│           module.md                     │
│  - 主模块职责、边界                       │
│  - 核心抽象、对外接口                     │
│  - 不变量、并发/性能约束                   │
└───────────────────┬─────────────────────┘
                    │
┌───────────────────▼─────────────────────┐
│          子模块层 (Submodule)            │
│   docs/agent/memory/modules/<module>/         │
│        submodules/<submodule>.md        │
│  - 独立边界子组件                        │
│  - 详细实现约束                          │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│        Handoff 层 (临时状态)             │
│        docs/agent/handoff/workstreams/        │
│  - 当前进度、阻塞点、下一步               │
│  - 会话级上下文，通过 git 同步            │
│  - 支持跨机器恢复                        │
└─────────────────────────────────────────┘
```

### 2.2 记忆层级说明

| 层级 | 范围 | 稳定性 | 适用内容 | 存储位置 |
|------|------|--------|----------|----------|
| **全局层** | 整个项目 | 高 | 构建约定、代码规范、验证流程 | `docs/agent/memory/project.md` |
| **模块层** | 单个主模块 | 高 | 职责边界、核心抽象、对外接口 | `docs/agent/memory/modules/<module>/module.md` |
| **子模块层** | 子组件 | 中高 | 实现细节、内部不变量、优化策略 | `docs/agent/memory/modules/<module>/submodules/<submodule>.md` |
| **handoff** | 单次会话 | 低 | 当前进度、阻塞点、下一步行动 | `docs/agent/handoff/workstreams/<key>/` (git 同步) |

### 2.3 冲突优先级

当不同来源信息冲突时，按源优先级与已验证事实处理；完整覆盖关系见 `3.3 源优先级与覆盖关系`。

**处理规则**:
- 优先回到用户显式指令、已验证代码/测试事实确认
- handoff 与稳定记忆冲突时，先回到代码、测试或用户指令验证
- 未验证前不要直接覆盖 memory 文件

---

## 3. 文件结构

### 3.1 目录布局

```
docs/agent/memory/
├── README.md                    # 操作规范（最终操作依据）
├── QUICKSTART.md                # 快速入门示例
├── project.md                   # 项目级记忆
├── mainmodule_memory_template.md    # 主模块模板
├── submodule_memory_template.md     # 子模块模板
└── modules/
    └── <module>/                # 模块目录
        ├── module.md            # 主模块记忆
        ├── submodules/          # 子模块目录
        │   └── <submodule>.md
        └── adrs/                # ADR 目录
            └── ADR-XXX.md

docs/agent/prompts/
├── README.md                    # Prompt 说明
├── quick_resume.md              # 快捷恢复（默认入口）
├── new_session_template.md      # 显式启动（备选）
├── handoff.md                   # 会话交接
├── memory_update_and_adr.md     # 记忆更新
└── generate_module_memory.md    # 生成完整记忆

docs/agent/handoff/
├── README.md                    # handoff 存储说明
└── workstreams/
    └── <module>__<submodule-or-none>/
        └── YYYYMMDDTHHMMSSZ--<session_id>--<agent_id>.md

docs/agent/decisions/
└── template.md                  # ADR 模板
```

- `docs/agent/memory_system.md` 位于 `docs/agent/` 目录，作为记忆系统的架构设计说明。
- handoff 输出存储在任务记录/对话中，同时持久化到 `docs/agent/handoff/workstreams/<workstream_key>/`。
- 通过 git 同步实现跨机器恢复。

### 3.2 文件命名规范

| 类型 | 命名规则 | 示例 |
|------|----------|------|
| 模块目录 | ASCII, 小写, snake_case | `ammalloc/` |
| 子模块文件 | 同上 | `thread_cache.md` |
| ADR 文件 | 零填充编号 | `ADR-001.md` |
| 模板文件 | 描述性名称 | `mainmodule_memory_template.md` |

### 3.3 源优先级与覆盖关系

数字越小优先级越高；高优先级来源可以覆盖低优先级来源的描述。

| 优先级 | 来源 |
|--------|------|
| 1 | 用户显式指令与已验证代码/测试事实 |
| 2 | `AGENTS.md` (项目执行指南) |
| 3 | `docs/aethermind_prd.md` (Phase 1 产品需求) |
| 4 | ADR (架构决策记录) |
| 5 | 模块/子模块记忆文档 |
| 6 | `docs/agent/memory/project.md` |
| 7 | handoff (临时状态) |
| 8 | `GEMINI.md` (架构蓝图参考) |

`docs/agent/memory/README.md` 用于约束操作流程和文档结构；遇到与事实来源冲突的内容时，仍按上表回到更高优先级来源确认。

---

## 4. 工作流

### 4.1 完整工作流图

```
旧会话结束
    ↓
┌──────────────────┐
│   handoff.md     │  生成交接摘要
│   (临时状态)      │
└──────────────────┘
    ↓
新会话启动
    ↓
┌──────────────────────────────────────────────────────────┐
│  AGENTS.md -> docs/agent/memory/README.md -> project.md ->     │
│   module.md -> submodule.md (如存在) -> handoff           │
└──────────────────────────────────────────────────────────┘
    ↓
执行任务
    ↓
任务完成
    ↓
选择一种回写方式
    ↓
    ├─ 增量回写 / 新增 ADR
    │   ↓
    │  ┌──────────────────────────────────┐
    │  │ memory_update_and_adr.md         │
    │  │  - 增量更新记忆                    │
    │  │  - 创建新 ADR (如需要)             │
    │  └──────────────────────────────────┘
    │
    └─ 首次建档 / 整体重写
        ↓
       ┌──────────────────────────────────┐
       │ generate_module_memory.md        │
       │  - 生成完整模块记忆                 │
       └──────────────────────────────────┘
    ↓
新一轮 handoff
```

### 4.2 记忆更新触发条件

创建或更新记忆文档的时机：

1. ✅ 任务完成后形成新的稳定事实、接口契约或不变量
2. ✅ 设计里程碑落定，且结果已被代码/测试/评审确认
3. ✅ 新增、替换或废弃 ADR
4. ✅ handoff 中出现会影响后续开发的稳定结论
5. ❌ 纯试验、调试日志、临时 workaround
6. ❌ 未定案讨论

---

## 5. 模板规范

### 5.1 YAML Frontmatter (所有记忆文件必填)

```yaml
---
scope: module              # module | submodule | global
module: ammalloc           # 当前记忆节点名称
parent: none               # 父模块名，无则写 none
depends_on: []             # 依赖的其他模块/子模块
adr_refs:                  # 引用的 ADR
  - ./adrs/ADR-001.md
last_verified: 2026-03-10  # 验证日期
owner: team                # agent | team
status: active             # active | draft | deprecated
---
```

### 5.2 11节固定结构

所有模块/子模块记忆必须按以下顺序组织：

1. **模块范围**
   - 职责（负责/不负责）
   - 边界（输入/输出/不直接管理）
   - 子模块划分（如适用）

2. **已确认事实**
   - 已验证约束
   - 已验证限制
   - ADR 关联
   - 非阻塞注意事项

3. **核心抽象**
   - 关键抽象（类型/概念）
   - 数据流

4. **对外接口**
   - 头文件/API
   - 入口函数
   - 错误语义

5. **不变量**
   - 必须始终成立的条件

6. **所有权与生命周期**
   - 所有者
   - 借用关系
   - 销毁时机

7. **并发约束**
   - 线程角色
   - 同步要求
   - 禁止事项

8. **性能约束**
   - 热路径
   - 时间复杂度/常数项要求
   - 内存约束

9. **已否决方案**
   - 方案及原因

10. **未决问题**
    - 尚未定案且阻塞实现的问题

11. **待办事项**
    - [ ] 具体工作项

### 5.3 占位符约定

- 信息缺失时写 `无` 或 `未涉及`，禁止编造
- 使用 `[MODULE_NAME]` 等占位符表示待填写内容
- 示例数据必须明确标注为示例

---

## 6. 使用指南

### 6.1 新模块初始化流程

1. **创建目录结构**
   ```bash
   mkdir -p docs/agent/memory/modules/<module>/submodules
   mkdir -p docs/agent/memory/modules/<module>/adrs
   ```

2. **复制模板**
   ```bash
   cp docs/agent/memory/mainmodule_memory_template.md \
      docs/agent/memory/modules/<module>/module.md
   ```

3. **填写 frontmatter**
   - 设置 `scope: module`
   - 填写 `module` 名称
   - 设置 `parent: none`
   - 填写 `last_verified` 和 `owner`

4. **填充内容**
   - 使用 `docs/agent/prompts/generate_module_memory.md` 生成内容
   - 或手动填写，确保每个章节都有内容

5. **创建 ADR (如需要)**
   ```bash
   cp docs/agent/decisions/template.md \
      docs/agent/memory/modules/<module>/adrs/ADR-001.md
   ```

6. **更新索引**
   - 在 `docs/agent/memory/project.md` 中添加子模块划分
   - 更新 `adr_refs` 字段

### 6.2 会话交接流程

1. **结束当前会话**
   - 使用 `docs/agent/prompts/memory_update_and_adr.md` 回写稳定结论
   - 如有新 ADR，创建到 `adrs/` 目录

2. **生成 handoff**
   - 使用 `docs/agent/prompts/handoff.md` 结构输出
   - 包含：目标、当前状态、涉及文件、阻塞点、推荐下一步
   - 输出到任务记录/对话中
   - **同时写入 `docs/agent/handoff/workstreams/<workstream_key>/YYYYMMDDTHHMMSSZ--<session_id>--<agent_id>.md`**
   - 文件包含 YAML frontmatter（kind, schema_version, created_at, session_id, task_id, module, submodule, agent）
   - 通过 `git add docs/agent/handoff/` 和 `git commit` 提交，实现跨机器同步

3. **新会话启动**
   - 先读取 `AGENTS.md`
   - 再读取 `docs/agent/memory/README.md`（操作规范）
   - 然后读取 `docs/agent/memory/project.md`
   - 定位并读取相关 `module.md` 和 `submodule.md`
   - 最后按 `docs/agent/memory/README.md` 的规范获取 handoff（详见该文档"Handoff 存储规范"章节）

### 6.3 冲突处理流程

处理记忆文件冲突前，先按以下顺序完成上下文加载：

- `AGENTS.md`
- `docs/agent/memory/README.md`
- `docs/agent/memory/project.md`
- 相关 `module.md` 和 `submodule.md`
- handoff（如有，最后读取）

当多个代理需要编辑同一记忆文件：

1. **检查活跃编辑者**
   - 同一 memory 文件同时只允许一个活跃编辑者

2. **显式合并评审**
   - 后写入方基于最新文件和已验证代码/测试事实合并
   - 保留共同事实
   - 按作用范围下沉或上提内容

3. **无法确认的冲突**
   - 先留在 handoff 或任务记录
   - 不要写成稳定结论

---

## 7. 示例

### 7.1 ammalloc 模块结构

```
docs/agent/memory/modules/ammalloc/
├── module.md                    # 主模块记忆
├── submodules/
│   ├── thread_cache.md         # → ADR-005
│   ├── central_cache.md        # → ADR-001, ADR-005
│   ├── page_cache.md           # → ADR-002, ADR-004, ADR-005
│   ├── page_allocator.md       # → ADR-003
│   ├── size_class.md
│   ├── spin_lock.md
│   └── page_heap_scavenger.md  # → ADR-004
└── adrs/
    ├── ADR-001.md              # TransferCache 设计
    ├── ADR-002.md              # RadixTree PageMap
    ├── ADR-003.md              # 乐观大页策略
    ├── ADR-004.md              # MADV_DONTNEED
    └── ADR-005.md              # 无 STL 约束
```

### 7.2 ADR 引用示例

在 `central_cache.md` 中：

```yaml
---
scope: submodule
module: central_cache
parent: ammalloc
depends_on: [page_cache, size_class, spin_lock]
adr_refs:
  - ../adrs/ADR-001.md
  - ../adrs/ADR-005.md
last_verified: 2026-03-10
owner: team
status: active
---
```

正文中：
```markdown
## 已确认事实
- ADR 关联：[ADR-001: TransferCache 设计](../adrs/ADR-001.md)、[ADR-005: 无 STL 约束](../adrs/ADR-005.md)
```

### 7.3 依赖关系示例

在 `page_heap_scavenger.md` 中：

```yaml
---
scope: submodule
module: page_heap_scavenger
parent: ammalloc
depends_on: [page_cache]      # 依赖 page_cache
adr_refs:
  - ../adrs/ADR-004.md
last_verified: 2026-03-10
owner: team
status: active
---
```

---

## 8. 附录

### 8.1 Prompt 速查表

| Prompt | 使用时机 | 输出目标 |
|--------|----------|----------|
| `quick_resume.md` | 日常接续工作 | 快速恢复记忆与最新 handoff |
| `new_session_template.md` | 启动新会话 | 收敛上下文、加载记忆 |
| `handoff.md` | 会话结束 | 交接状态、阻塞点、下一步 |
| `memory_update_and_adr.md` | 任务完成 | 记忆增量、新 ADR |
| `generate_module_memory.md` | 首次建档/大修 | 完整模块记忆 |

### 8.2 状态说明

| 状态 | 含义 |
|------|------|
| `active` | 当前有效，持续维护 |
| `draft` | 草案阶段，待验证 |
| `deprecated` | 已废弃，保留历史 |

### 8.3 相关文档

- `AGENTS.md` - AI 助手执行指南
- `docs/agent/memory_system.md` - 记忆系统架构设计说明
- `docs/agent/memory/README.md` - 记忆系统操作规范
- `docs/agent/prompts/README.md` - Prompt 说明
- `GEMINI.md` - 项目架构蓝图

### 8.4 维护清单

- [ ] 更新 `last_verified` 日期
- [ ] 同步 `adr_refs` 与实际 ADR
- [ ] 更新 `project.md` 子模块索引
- [ ] 废弃 ADR 时同步更新引用
- [ ] 标记 `status: deprecated` 时更新相关链接
- [ ] 清理 `docs/agent/handoff/` 中超过 7 天的过期文件（自动或手动）

---

> **维护提示**: 本文档本身应被视为项目级稳定事实。当记忆系统架构发生重大变更时，同步更新此文档。
