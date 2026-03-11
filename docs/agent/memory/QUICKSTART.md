# Agent Memory System 快速入门

> 从零开始使用记忆系统的完整示例

---

## 场景

你正在开发 `ammalloc` 内存池的 `thread_cache` 模块，需要跨会话保持上下文。

---

## 第一步：初始化模块记忆（首次）

### 1. 创建目录结构
```bash
mkdir -p docs/agent/memory/modules/ammalloc/submodules
mkdir -p docs/agent/memory/modules/ammalloc/adrs
mkdir -p docs/agent/handoff/workstreams
```

### 2. 创建主模块记忆

**文件**：`docs/agent/memory/modules/ammalloc/module.md`

```markdown
---
scope: module
module: ammalloc
parent: none
depends_on: []
adr_refs: []
last_verified: 2026-03-11
owner: team
status: active
---

# ammalloc 主模块记忆

## 模块范围
### 职责
- 负责：高性能内存分配/释放
- 不负责：业务逻辑、模型推理

### 边界
- 上游：调用方通过 `am_malloc()`/`am_free()`
- 下游：返回指针或归还内存到 OS

### 子模块划分
- `thread_cache`：线程局部缓存

## 已确认事实
- 采用 ThreadCache → CentralCache → PageCache 分层架构
- 单线程快路径目标 < 5ns

## 核心抽象
- `FreeList`：嵌入式 LIFO 链表
- `Span`：连续页区间

## 待办事项
- [ ] 实现 ThreadCache 动态水位线
```

### 3. 创建子模块记忆

**文件**：`docs/agent/memory/modules/ammalloc/submodules/thread_cache.md`

```markdown
---
scope: submodule
module: thread_cache
parent: ammalloc
depends_on: []
adr_refs: []
last_verified: 2026-03-11
owner: team
status: active
---

# thread_cache 子模块记忆

## 模块范围
### 职责
- 线程局部前端缓存
- 无锁快路径分配/释放

## 已确认事实
- 使用 `alignas(CACHE_LINE_SIZE)`
- 快路径 O(1) 且无锁

## 待办事项
- [ ] 实现动态水位线调节
```

### 4. 提交到 git
```bash
git add docs/agent/memory/
git commit -m "memory: init ammalloc thread_cache"
git push
```

---

## 第二步：开始工作（新会话）

### 用户输入
> "继续 ammalloc thread_cache 的工作"

### Agent 自动执行

**1. 解析范围**
- 模块：`ammalloc`
- 子模块：`thread_cache`

**2. 加载记忆**（按顺序）
```
✅ AGENTS.md
✅ docs/agent/memory/README.md
✅ docs/agent/memory/project.md
✅ docs/agent/memory/modules/ammalloc/module.md
✅ docs/agent/memory/modules/ammalloc/submodules/thread_cache.md
⚠️ 无 handoff（首次工作）
```

**3. 输出状态**
```markdown
## 已解析范围
- 模块：ammalloc
- 子模块：thread_cache

## 已加载文件
- ✅ docs/agent/memory/project.md
- ✅ docs/agent/memory/modules/ammalloc/module.md
- ✅ docs/agent/memory/modules/ammalloc/submodules/thread_cache.md
- ⚠️ 无 handoff（从 memory 的待办事项开始）

## 当前接续目标
实现 ThreadCache 动态水位线调节

## 下一步动作
- 修改 ammalloc/src/thread_cache.cpp
- 实现 FetchRange() 动态水位线逻辑
```

---

## 第三步：工作进行中

假设你完成了部分代码，但还没完成全部工作。

### 生成 Handoff（结束会话前）

**用户**：
> "生成 handoff" 或 "今天先到这里"

**Agent 执行**：
1. 按 `docs/agent/prompts/handoff.md` 生成 handoff 文件
2. 展示生成的文件内容供用户确认
3. 提示用户提交到 git

**生成的文件**：`docs/agent/handoff/workstreams/ammalloc__thread_cache/20260311T103000Z--ses_001--sisyphus.md`

```markdown
---
kind: handoff
schema_version: "1.1"
created_at: 2026-03-11T10:30:00Z
session_id: ses_001
task_id: task_tc_001
module: ammalloc
submodule: thread_cache
agent: sisyphus
status: active
memory_status: pending
supersedes: null
closed_at: null
closed_reason: null
---

# ThreadCache 动态水位线实现

## 目标
实现 FetchRange() 的动态水位线调节机制

## 当前状态
- 已完成：
  - 基础 ThreadCache 架构
  - FreeList LIFO 结构
- 未完成：
  - 动态水位线算法
  - 高并发测试

## 涉及文件
- ammalloc/src/thread_cache.cpp

## 阻塞点
需要确认 TransferCache 容量查询接口是否线程安全

## 推荐下一步
1. 添加 CentralCache::GetTransferCacheCapacity() 接口
2. 修改 FetchRange() 动态调节逻辑
3. 运行单测验证

## 验证方式
./build/tests/unit/aethermind_unit_tests --gtest_filter=ThreadCache.*
```

### 提交到 git
```bash
git add docs/agent/handoff/
git commit -m "handoff: ammalloc thread_cache progress"
git push
```

---

## 第四步：跨机器接续工作（新环境）

### 1. 拉取最新状态
```bash
git pull
# 下载了最新的 handoff 文件
```

### 2. 新会话启动

**用户输入**：
> "继续 ammalloc thread_cache 的工作"

**Agent 执行**：

```
✅ AGENTS.md
✅ docs/agent/memory/README.md
✅ docs/agent/memory/project.md
✅ docs/agent/memory/modules/ammalloc/module.md
✅ docs/agent/memory/modules/ammalloc/submodules/thread_cache.md
✅ docs/agent/handoff/workstreams/ammalloc__thread_cache/20260311T103000Z--ses_001--sisyphus.md
```

**Agent 输出**：
```markdown
## 已解析范围
- 模块：ammalloc
- 子模块：thread_cache

## 已加载文件
- ✅ docs/agent/memory/modules/ammalloc/module.md
- ✅ docs/agent/memory/modules/ammalloc/submodules/thread_cache.md
- ✅ docs/agent/handoff/workstreams/ammalloc__thread_cache/20260311T103000Z--ses_001--sisyphus.md

## 当前接续目标
从昨天的阻塞点继续：确认 TransferCache 容量查询接口线程安全性

## 阻塞点
- TransferCache 容量查询接口是否线程安全

## 推荐下一步
1. 检查 central_cache.h 的 TransferCache 实现
2. 确认容量查询是否需要加锁
3. 实现 GetTransferCacheCapacity() 接口
```

---

## 第五步：完成工作并关闭 Handoff

工作完成后，你可以选择：

### 选项 A：回写 Memory 并关闭 Handoff

如果形成了稳定结论，先回写 memory：

**更新**：`docs/agent/memory/modules/ammalloc/submodules/thread_cache.md`
```yaml
# 在已确认事实中添加：
- 动态水位线算法：根据 TransferCache 剩余容量的 1/4 调节
- 高并发场景下吞吐量提升 15%
```

然后关闭 handoff：

**更新 handoff**：
```yaml
status: closed
memory_status: applied
closed_at: 2026-03-11T18:00:00Z
closed_reason: completed
```

```bash
git add docs/agent/memory/ docs/agent/handoff/
git commit -m "memory: thread_cache 动态水位线设计; handoff: completed"
git push
```

### 选项 B：生成新的 Handoff（工作未结束）

如果还有剩余工作，生成新的 handoff 取代旧的：

**新文件**：`20260311T140000Z--ses_002--sisyphus.md`
```yaml
status: active
memory_status: pending
supersedes: 20260311T103000Z--ses_001--sisyphus.md
```

**旧文件更新**：
```yaml
status: superseded
```

```bash
git add docs/agent/handoff/
git commit -m "handoff: new progress, supersede old"
git push
```

---

## 第六步：长期维护

### 定期更新 Memory
```bash
# 每周或每个里程碑后
git add docs/agent/memory/
git commit -m "memory: update verification date and progress"
git push
```

### 清理旧的 Handoff

本地清理策略（建议手动执行，与 README 一致）：
```bash
# 删除超过 7 天的 closed/superseded handoff
# 始终保留最近 3 个文件（无论状态）
# 永远不要删除 status: active 的 handoff

# 1. 查看文件状态
# grep -l "status: closed" docs/agent/handoff/workstreams/ammalloc__thread_cache/*.md
# grep -l "status: superseded" docs/agent/handoff/workstreams/ammalloc__thread_cache/*.md
# 2. 手动删除超过 7 天的 closed/superseded 文件，同时保留最近 3 个文件
```

**原则**：宁可保留过多，也不要误删 `active` handoff。

Git 历史保留所有记录用于审计。

---

## 常见问题

### Q: 没有 handoff 时能工作吗？
**A**: 可以。Agent 会从 memory 的"待办事项"开始。

### Q: 多个 active handoff 怎么办？
**A**: 选择 `created_at` 最新的一个，其他的标记为 `superseded`。

### Q: 如何查看历史 handoff？
**A**: `git log docs/agent/handoff/` 查看所有历史提交。

### Q: 可以删除 closed/superseded 的 handoff 吗？
**A**: 本地可以删除，但 git 历史会保留。建议按 README 的手动清理策略处理。

---

## 完整命令速查

```bash
# 初始化
git clone <repo>
cd AetherMind

# 每天开始
git pull
# 用户说："继续 <module> <submodule> 的工作"

# 每天结束
# 用户说："生成 handoff" → Agent 生成文件 → 用户确认
git add docs/agent/handoff/
git commit -m "handoff: <module> <submodule> progress"
git push

# 里程碑完成
# 用户说："工作完成，回写 memory" → Agent 更新文件 → 用户确认
git add docs/agent/memory/ docs/agent/handoff/
git commit -m "memory: <updates>; handoff: completed"
git push
```

---

> 更多细节见：
> - `docs/agent/memory/README.md` - 完整规范
> - `docs/agent/prompts/quick_resume.md` - 快捷恢复流程
> - `docs/agent_memory_system.md` - 架构设计
