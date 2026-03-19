请把当前对话压缩成一份"新对话导入摘要"，供新的 agent 立即接手当前大模型推理引擎模块或项目级工作。

输出格式要求：

1. **YAML Frontmatter（必须）**：文件开头必须包含以下元数据

   **模块级示例**：

   ```yaml
   ---
   kind: handoff
   schema_version: "1.1"
   created_at: 2026-03-11T10:30:00Z
   session_id: ses_xxx
   task_id: task_xxx
   module: ammalloc
   submodule: thread_cache
   slug: null                        # 模块工作：必须为 null
   agent: sisyphus
   status: active
   bootstrap_ready: false           # 默认 false；仅当该 handoff 足以支持低上下文恢复时才写 true
   memory_status: not_needed
   supersedes: null
   closed_at: null
   closed_reason: null
   ---
   ```

   **项目级示例**：
   ```yaml
   ---
   kind: handoff
   schema_version: "1.1"
   created_at: 2026-03-11T10:30:00Z
   session_id: ses_xxx
   task_id: task_xxx
   module: project                   # 固定值
   submodule: null                   # 项目级工作：必须为 null
   slug: docs-reorg                  # 项目级工作：填写 slug
   agent: sisyphus
   status: active
   bootstrap_ready: false
   memory_status: not_needed
   supersedes: null
   closed_at: null
   closed_reason: null
   ---
   ```

   **状态字段说明**：
   - `status: active`：当前可继续的 handoff（默认创建时）
   - `bootstrap_ready: false`：默认值；只有当 handoff 已包含继续工作所需的最小恢复信息时，才允许写成 `true`
   - `memory_status: not_needed`：无需回写 stable memory（默认创建时）
   - `memory_status: pending`：有稳定结论待回写 memory
   - `supersedes`：如果是取代旧 handoff，填写旧文件名；否则 null

2. **正文结构**：在 frontmatter 后，按以下章节组织：
   - 目标
   - 当前状态
   - 涉及文件
   - 已确认接口与不变量
   - 阻塞点
   - 推荐下一步
   - 验证方式

内容要求：
1. 若代码、用户指令与讨论内容冲突，优先采信已验证的代码事实和用户显式指令，不要自行推断缺失信息。
2. 只保留继续开发所必需的信息，不要复述完整历史过程。
3. 若某类信息在当前上下文中确实不存在，明确标注`未涉及`或`无`，禁止编造。
4. `涉及文件`中必须给出精确文件路径；没有文件改动时写`无`。
5. 若存在可执行命令，给出与当前状态直接相关的精确命令；不要写泛泛建议。
6. 若本轮涉及构建、测试或基准，必须写明状态：已执行并通过 / 已执行未通过 / 未执行。
7. `推荐下一步`必须具体、可执行，直接说明先做什么、改哪里、如何验证。
8. 保持精炼，但不能丢失会影响接手判断的关键信息。
9. handoff 只记录**当前状态增量**，不要复制长期启动规则、完整加载顺序或稳定 memory 规范。

存储与状态管理：
- 保存到 `docs/agent/handoff/workstreams/<workstream_key>/YYYYMMDDTHHMMSSZ--<session_id>--<agent_id>.md`
  - 模块工作：`<workstream_key>` 为 `<module>__<submodule-or-none>`
  - 项目级工作：`<workstream_key>` 为 `project__<slug>`
- `bootstrap_ready: true` 只用于说明该 handoff 足以支撑低上下文恢复；若缺失该字段，读取方必须按 `false` 处理
- 如果同一 workstream 已有 `active` handoff，新文件写 `supersedes: <旧文件名>`，并更新旧文件为 `status: superseded`
- 工作完成时，将当前 `active` 改为 `status: closed`，填写 `closed_at` 和 `closed_reason`
- 通过 git 提交同步
