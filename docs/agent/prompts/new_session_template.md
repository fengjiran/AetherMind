开始新会话时，先按以下约束收敛上下文，再继续实现。

## ⚠️ 强制约束（HARD CONSTRAINT）

**加载记忆后，必须等待用户明确确认，禁止自动执行任何操作。**

```
Agent: 加载记忆 (AGENTS.md → docs/agent/memory/README.md → docs/agent/memory/project.md → [docs/agent/memory/modules/<module>/module.md → docs/agent/memory/modules/<module>/submodules/<submodule>.md，如适用] → docs/agent/handoff/workstreams/<workstream_key>/)
    ↓
Agent: 输出 "记忆已加载，本轮目标是..."
    ↓
⚠️ 检查点: 必须等待用户说"继续"、"执行"或"是"
    ↓
用户: "继续"  ← 必须显式确认
    ↓
Agent: 才能执行工具操作（扫描代码、编译、测试等）
```

**禁止行为（违反将导致错误）**：
- ❌ 加载记忆后立即扫描代码/编译/测试
- ❌ 假设"继续"等于自动执行
- ❌ 未经确认就修改文件

**正确流程**：
- ✅ 加载记忆 → 输出状态 → **显式询问用户** → 等待确认 → 执行操作

---

> **本模板适用于**：
> - 首次建档新模块
> - 跨模块协调任务
> - 需要明确预填目标/ADR/回写项的正式启动
> 
> **日常快速接续**：使用 [`quick_resume.md`](./quick_resume.md)，一句话即可恢复工作。
>
> **workstream 键与 handoff frontmatter**：以 [`docs/agent/memory/README.md`](../memory/README.md) 为最终依据。

## 前置检查（按此顺序执行）：

按 `docs/agent/memory/README.md` 的规范读取顺序：
1. 阅读 `AGENTS.md`
2. 阅读 `docs/agent/memory/README.md`（操作规范，也是 workstream 键与 handoff frontmatter 的最终依据）
3. 阅读 `docs/agent/memory/project.md`；
   - 若存在，正常加载
   - 若缺失且 workstream 为显式 `project__<slug>`：标记为 `partial`，继续（项目级共识缺失但仍可恢复）
   - 若缺失且需要依赖 project.md 推断 slug：标记为 `blocked`，停止并要求用户显式指定 workstream
4. 若 `[WORKSTREAM_KEY]` 为 `project__<slug>`：
   - 使用 `module: project`、`submodule: null`、`slug: [SLUG]`
   - 跳过 `module.md` 和 `submodule.md`
5. 若 `[WORKSTREAM_KEY]` 为 `<module>__<submodule-or-none>`：
   - 检查 `docs/agent/memory/modules/[MODULE_NAME]/module.md`；存在则加载，不存在时明确记为 `无主模块 memory`
   - 若 `[SUBMODULE_NAME|null]` 不为 `null`，再检查 `docs/agent/memory/modules/[MODULE_NAME]/submodules/[SUBMODULE_NAME].md`；不存在时明确记为 `无子模块 memory`
6. 获取 handoff（临时状态）：
   - 从任务系统/对话上下文中获取（优先）
   - 若不可用，从 `docs/agent/handoff/workstreams/[WORKSTREAM_KEY]/` 读取最新 handoff 文件（fallback）
7. 参考 `docs/agent/prompts/handoff_template.md` 的输出结构规范
8. 如本轮形成稳定结论，按 `docs/agent/prompts/memory_update_and_adr.md` 准备 memory 增量；若 `[是否有 ADR 增量: 是/否]` 为 `是`，同步准备 ADR 草案

**Workstream 键规则**：详见 `docs/agent/memory/README.md` "Handoff 存储规范"章节

## 本轮输入：

- Workstream：`[<module>__<submodule-or-none> | project__<slug>]`
- 主模块：`[MODULE_NAME | project]`
- 子模块：`[SUBMODULE_NAME | null]`
- 项目级 slug：`[SLUG | null]`
- 目标：`[本轮目标]`
- 需要回写到 memory 的内容：`[需要回写到 memory 的内容]`
- 是否有 ADR 增量：`[是否有 ADR 增量: 是/否]`
- 构建/测试/基准状态：`[构建/测试/基准状态]`

### 输入示例（模块工作）

```yaml
workstream: ammalloc__thread_cache
module: ammalloc
submodule: thread_cache
slug: null
goal: 实现 ThreadCache 动态水位线调节
memory_writeback: 更新 thread_cache 的稳定约束
has_adr_delta: 否
```

### 输入示例（项目级工作）

```yaml
workstream: project__docs-reorg
module: project
submodule: null
slug: docs-reorg
goal: 整理 docs/agent/ 文档入口与目录说明
memory_writeback: 更新 project.md 中的文档约定
has_adr_delta: 否
```

## 输出要求：

### 目标
- 直接写本轮要完成的工作：`[本轮目标]`
- 若目标拆分为多步，只保留当前最小可执行范围。

### 当前状态
- 已完成：[已落地内容；没有则写 `无`]
- 未完成：[剩余工作；没有则写 `无`]
- Workstream：`[WORKSTREAM_KEY]`
- 已加载记忆：[始终列出 `docs/agent/memory/project.md`；若为模块工作，再列出实际存在的 `docs/agent/memory/modules/[MODULE_NAME]/module.md` 和 `docs/agent/memory/modules/[MODULE_NAME]/submodules/[SUBMODULE_NAME].md`]
- 所有权与生命周期约束：[资源由谁拥有，谁只能借用，何时释放]
- 线程安全预期：[单线程 / 多线程、可重入性、锁或原子要求]
- 热路径注意事项：[哪些路径必须避免额外分配、复制、锁竞争或隐藏 O(N^2)]
- 构建/测试/基准状态：`[构建/测试/基准状态]`

### 涉及文件
- `docs/agent/memory/project.md`：[已存在 / 不存在 / 本轮未修改 / 本轮已修改]
- `docs/agent/memory/modules/[MODULE_NAME]/module.md`：[已存在 / 不存在 / 不适用（project__<slug>） / 本轮未修改 / 本轮已修改]
- `docs/agent/memory/modules/[MODULE_NAME]/submodules/[SUBMODULE_NAME].md`：[已存在 / 不存在 / 不适用（项目级 workstream 或 submodule=null） / 本轮未修改 / 本轮已修改]
- `docs/agent/handoff/workstreams/[WORKSTREAM_KEY]/[LATEST_HANDOFF].md`：[已存在 / 不存在 / 本轮未修改 / 本轮已修改]
- `[精确文件路径]`：[用途或改动点]
- 无

### 已确认接口与不变量
- 接口：[函数 / 类 / 状态机 / 数据结构 / 头文件路径]
- 前置条件：[调用前必须满足什么]
- 后置条件：[调用后必须保证什么]
- 不变量：[实现过程中不能破坏的条件]
- 若尚未确认：`未涉及`

### 阻塞点
- [阻塞实现推进的问题；无则写 `无`]
- 只写真正阻塞后续工作的事项，不写普通风险提示。

### 推荐下一步
- 先改：[精确文件路径 + 目标]
- 再验证：[对应命令]
- 若需要回写，明确列出：`[需要回写到 memory 的内容]`
- 若需要 ADR，明确标注：`[是否有 ADR 增量: 是/否]`

### 验证方式
- 配置：`cmake -S . -B build -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON`
- 构建库：`cmake --build build --target AetherMind -j`
- 构建单测：`cmake --build build --target aethermind_unit_tests -j`
- 单测：`./build/tests/unit/aethermind_unit_tests --gtest_filter=[Suite].[Case]`
- 构建基准：`cmake --build build --target aethermind_benchmark -j`
- 基准：`./build/tests/benchmark/aethermind_benchmark --benchmark_filter=[BenchmarkName]`
- 如涉及 TSAN：`cmake -S . -B build-tsan -DENABLE_TSAN=ON -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF && cmake --build build-tsan --target aethermind_unit_tests -j`
- 没有执行的项必须明确写 `未执行`，不要省略。

---

## 边界约束

1. **加载后需确认**：使用 new_session_template.md 时，Agent 完成记忆加载后，**必须等待用户确认**是否执行下一步操作，**禁止**在加载记忆后立即自动执行任何工具操作（如扫描代码、编译、测试等）。只有在用户明确说"继续"、"执行"或类似指令后，才能执行操作。
