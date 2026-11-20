# AetherMind 文档

[项目手册](handbook/README.md) 是实现文档的主入口，覆盖架构、接口、构建运行和验证。

## 文档地图

| 我想了解 | 从这里开始 |
|---|---|
| 使用与部署 | [构建](../README.md#build)、[运行](../README.md#run-inference)、[产品范围与精度矩阵](product.md#14-当前能力与验证状态)、[Runtime 配置](guides/runtime-configuration.md) |
| API 集成 | [C++ API](api/public-api.md)、[Runtime C ABI](api/runtime-c-abi.md)、[资源预算](guides/resource-budgeting.md) |
| 架构与开发 | [系统架构](architecture.md)、[核心模块](handbook/核心模块设计/核心模块设计.md)、[仓库开发规则](../AGENTS.md)、[算子开发流程](guides/operator-development-workflow.md) |
| 测试与证据 | [产品验收矩阵](product.md#5-验收标准与测试矩阵)、[集成验证](guides/final-validation.md)、[算子契约与基准](operators/README.md) |
| 规划与历史 | [项目路线图](roadmap/README.md)、[待解决问题](issues.md#未解决)、[历史归档](archive/README.md) |

## 写作规则

产品范围与验收写在 `product.md`，跨模块约束写在 `architecture.md`。[公开 API 与调用契约](api/public-api.md) 和 [Runtime C ABI](api/runtime-c-abi.md) 由 `docs/api/` 维护，[单算子契约与验证证据](operators/README.md) 由 `docs/operators/` 维护。手册专题记录实现、工作流和阅读示例，涉及这些契约时链接到维护入口。按已验证的代码、CMake 和测试事实核对实现描述。

行为变更与主文档放在同一 PR。用户可见变化记入 [CHANGELOG](../CHANGELOG.md)，性能结论链接 [算子证据](operators/README.md)。

提交前运行 `python3 tools/check_docs_links.py` 检查根入口和活跃文档的相对路径及 Markdown 标题锚点。
