# 项目上下文

> 本文件只记录跨模块稳定事实。执行约束以根目录 `AGENTS.md` 为准，当前工作状态见 `docs/agent/handoff.md`。

## 项目目标与边界

- AetherMind 是 C++20 桌面/服务器 CPU 本地大模型推理引擎。
- 当前产品：CPU-only、单进程、同步单请求、Token IDs 输入输出、静态 KV Cache、贪婪采样、Llama Dense 模型。
- 长期方向（非当前承诺）：服务化、异构计算、分布式推理、其他模型家族与量化执行。

## 目录与构建

- `include/`：公共头文件；`src/`：核心实现；`apps/`：应用；`tests/`：测试；`docs/`：文档。
- 构建、测试与 benchmark 命令以根 `README.md` 为准。
- 模块职责与依赖方向以 `AGENTS.md` §2.1 为准。
- 核心库目标 `AetherMind`；CLI `aethermind_infer`；单测 `aethermind_unit_tests`。

## 文档入口

- 架构：`docs/architecture.md`
- 模块设计：`docs/modules/`
- 产品范围与验收：`docs/product.md`
- 开发规范：`docs/guides/`
- 算子契约与证据：`docs/operators/`
- 已知问题：`docs/issues.md`

## 不变量

- 已验证代码、CMake 与测试事实优先于任何文档。
- 文档与代码同一 PR 修改；过时文档直接删除或合并，历史由 git 保留。
- 稳定事实只写一处，其他文档链接引用。
