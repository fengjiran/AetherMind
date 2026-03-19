开始新会话时，先完成最小上下文恢复，再等待用户明确确认后继续执行。

> 默认启动契约、最小加载路径、确认闸门与 Resume Gate，统一以 `AGENTS.md` 为准。
> 本模板只负责提供**结构化输入**，不重新定义启动顺序。

## 使用场景

- 首次建档新模块
- 跨模块协调任务
- 需要显式填写目标、回写项、ADR 增量
- 需要正式启动一个项目级 workstream

## 强制约束

- 加载完成后，必须等待用户明确说“继续”/“执行”/“是”
- 在此之前，禁止扫描代码、编译、测试、编辑文件
- 若工作流类型不明确，必须先澄清，不得猜测

## 本轮输入

```yaml
workstream: <module>__<submodule-or-none> | project__<slug>
module: <module-name | project>
submodule: <submodule-name | null>
slug: <slug | null>
goal: <本轮目标>
memory_writeback: <需要回写到 memory 的内容>
has_adr_delta: <是 | 否>
notes: <补充说明>
```

## Agent 执行要求

1. 先按根目录 `AGENTS.md` 解析工作流类型
2. 按根目录 `AGENTS.md` 执行默认启动
3. 仅在需要 schema/兼容/写回规则时，按需升级读取 `docs/agent/memory/README.md`
4. 仅在需要模块级约束时，按需升级读取 `module.md` / `submodule.md`
5. 输出状态后，等待用户确认

## 输出要求

### 目标
- 直接写本轮要完成的最小可执行目标

### 当前状态
- 已完成：无则写 `无`
- 未完成：无则写 `无`
- Workstream：`<resolved_workstream_key>`
- 已加载文件：列出实际路径
- 按需升级读取：无则写 `无`
- `resume_status`：`complete | partial | blocked`

### 涉及文件
- 列出本轮最可能涉及的精确文件路径
- 若尚未定位，写 `待定位`

### 已确认接口与不变量
- 接口：无则写 `未涉及`
- 前置条件：无则写 `未涉及`
- 后置条件：无则写 `未涉及`
- 不变量：无则写 `未涉及`

### 阻塞点
- 无则写 `无`

### 推荐下一步
- 先做什么
- 再验证什么
- 若需回写或 ADR，明确指出

### 验证状态
- 已执行：列出已完成验证
- 未执行：列出尚未执行验证

## 边界提醒

- `docs/agent/prompts/handoff_template.md` 只可作为写作模板
- handoff 的真实来源是 `docs/agent/handoff/workstreams/<workstream_key>/`
- `bootstrap_ready: true` 只表示 handoff 可支撑低上下文恢复，不代表稳定事实已经回写
