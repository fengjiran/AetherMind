# AetherMind 文档

本目录只保留四类内容：系统现状、模块设计、过程规范、已知问题。构建、测试与运行命令以根目录 [README.md](../README.md) 为准，本页不重复。

## 文档地图

| 入口 | 回答的问题 | 权威性 |
|---|---|---|
| [architecture.md](architecture.md) | 系统怎么分层、依赖方向、跨层数据流、关键冻结决策 | 架构唯一权威 |
| [modules/](modules/) | 每个模块负责什么，接口、所有权、并发与错误边界 | 模块设计唯一权威 |
| [product.md](product.md) | 产品范围、验收标准、非目标、演进方向 | 产品唯一权威 |
| [guides/](guides/) | 怎么编码、注释、测试、评审、做算子 | 过程规范 |
| [api/](api/) | 对外 API 的生命周期与错误语义 | 与头文件 Doxygen 同源 |
| [operators/](operators/) | 算子契约、reference/optimized 证据 | 算子文档唯一位置 |
| [issues.md](issues.md) | 已知缺陷与待办 | 短生命周期 |
| [archive/](archive/) | 历史资料 | 不权威、不维护 |

> `docs/agent/` 仅保存代理上下文（`memory/project.md`、`handoff.md`），不属于人类文档权威体系。

## 权威顺序

1. 用户当前明确指令。
2. 已验证的代码、CMake 与测试事实。
3. `architecture.md`、`modules/`、`product.md`。
4. `guides/`、`api/`、`operators/`。
5. `issues.md`。
6. `archive/`：仅作历史参考，不作为事实来源。

## 写作规则

1. 新建文档前先问：能否并入现有入口？不能则不建。
2. 现状写 `architecture.md` / `modules/`；在研方向写 `roadmap.md`（Phase 3 建立）或 `issues.md`；不新建计划、评审、验证、日志目录。
3. 单篇文档不超过 600 行；按模块拆分，不按日期或版本拆分。
4. 不写状态、日期、版本头；不使用 `NN-` / `NNNN-` 编号；不使用模板目录。
5. 文档与代码同一 PR 修改；过时文档直接删除或合并，历史由 git 保留。
6. 模块文档固定六节：职责与边界（含"不得负责什么"）、关键数据结构与所有权、接口与代码位置、主流程、关键取舍（选择/原因/被否方案）、测试与验证入口。
7. 决策写进对应模块文档的"关键取舍"；跨模块且不可逆的决策写进 `architecture.md` 附录，每条一行并链接 PR。
8. 验证结论写进 `CHANGELOG.md` 或 `operators/<op>/benchmarks/`；不建带日期的验证报告。
9. 自动化只保留一种：相对链接有效性检查。
10. 术语随文定义；只有跨模块稳定术语才进入下表。

## 术语表

| 术语 | 定义 |
|---|---|
| ModelLoader | HF 目录 I/O、配置校验与权重解析，返回 `LoadedModel` |
| LoadedModel | 模型加载产物：config + resolved raw weights + backing storage，构造后只读 |
| BuildModelGraph | HF → 语义图的唯一转换入口，按模型家族分发到 per-family builder |
| ModelGraph | backend-independent 的语义 DAG |
| ModelCompiler | 编译阶段编排：`BuildModelGraph` → `OptimizeModelGraph` → `LowerModelGraph` |
| LoweredGraph | 不可变、结构验证过的编译产物 |
| ExecutionPlan | 不可变执行计划：kernel 已 resolve、权重已绑定、workspace 需求已冻结 |
| ExecutableModel | 模型准备产物：拥有编译产物、packed artifacts、权重绑定与 phase plans；入口为 `Load` / `Prepare` |
| InferenceSession | 同步贪婪生成编排：Prefill → Decode → Argmax/stop |
| KVCacheManager | 静态 KV 存储、单 session reservation 与提交水位管理 |
| W16/A32 | 保留 16-bit 矩阵权重、FP32 activation 的执行策略 |
| KV16 | 16-bit KV 存储；仅在显式策略下准入 |
| ammalloc | 自研用户态分配器（ThreadCache / CentralCache / PageCache） |
| Argmax | 当前产品唯一的贪婪采样策略 |
