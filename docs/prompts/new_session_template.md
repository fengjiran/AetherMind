开始新会话时，先按以下约束收敛上下文，再继续实现。

> **本模板适用于**：
> - 首次建档新模块
> - 跨模块协调任务
> - 需要明确预填目标/ADR/回写项的正式启动
> 
> **日常快速接续**：使用 [`quick_resume.md`](./quick_resume.md)，一句话即可恢复工作。

## 前置检查（按此顺序执行）：

按 `docs/memory/README.md` 的规范读取顺序：
1. 阅读 `AGENTS.md`
2. 阅读 `docs/memory/README.md`（操作规范）
3. 阅读 `docs/memory/project.md`；不存在时明确记为 `无项目级 memory`
4. 检查 `docs/memory/modules/[MODULE_NAME]/module.md`；存在则加载，不存在时明确记为 `无主模块 memory`
5. 若 `[SUBMODULE_NAME|无]` 不为 `无`：
   - 先加载父模块 `docs/memory/modules/[MODULE_NAME]/module.md`（如第4步未加载）
   - 再检查 `docs/memory/modules/[MODULE_NAME]/submodules/[SUBMODULE_NAME].md`；不存在时明确记为 `无子模块 memory`
6. 获取 handoff（临时状态）：
   - 从任务系统/对话上下文中获取（优先）
   - 若不可用，从 `docs/handoff/workstreams/<workstream_key>/` 读取最新 handoff 文件（fallback）
7. 参考 `docs/prompts/handoff.md` 的输出结构规范
8. 如本轮形成稳定结论，按 `docs/prompts/memory_update_and_adr.md` 准备 memory 增量；若 `[是否有 ADR 增量: 是/否]` 为 `是`，同步准备 ADR 草案

**Workstream 键规则**：详见 `docs/memory/README.md` "Handoff 存储规范"章节

## 本轮输入：

- 主模块：`[MODULE_NAME]`
- 子模块：`[SUBMODULE_NAME|无]`
- 目标：`[本轮目标]`
- 需要回写到 memory 的内容：`[需要回写到 memory 的内容]`
- 是否有 ADR 增量：`[是否有 ADR 增量: 是/否]`
- 构建/测试/基准状态：`[构建/测试/基准状态]`

## 输出要求：

### 目标
- 直接写本轮要完成的工作：`[本轮目标]`
- 若目标拆分为多步，只保留当前最小可执行范围。

### 当前状态
- 已完成：[已落地内容；没有则写 `无`]
- 未完成：[剩余工作；没有则写 `无`]
- 已加载记忆：[ `docs/memory/project.md` / `docs/memory/modules/[MODULE_NAME]/module.md` / `docs/memory/modules/[MODULE_NAME]/submodules/[SUBMODULE_NAME].md` 中实际存在的文件 ]
- 所有权与生命周期约束：[资源由谁拥有，谁只能借用，何时释放]
- 线程安全预期：[单线程 / 多线程、可重入性、锁或原子要求]
- 热路径注意事项：[哪些路径必须避免额外分配、复制、锁竞争或隐藏 O(N^2)]
- 构建/测试/基准状态：`[构建/测试/基准状态]`

### 涉及文件
- `docs/memory/project.md`：[已存在 / 不存在 / 本轮未修改 / 本轮已修改]
- `docs/memory/modules/[MODULE_NAME]/module.md`：[已存在 / 不存在 / 本轮未修改 / 本轮已修改]
- `docs/memory/modules/[MODULE_NAME]/submodules/[SUBMODULE_NAME].md`：[已存在 / 不存在 / 不适用 / 本轮未修改 / 本轮已修改]
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
