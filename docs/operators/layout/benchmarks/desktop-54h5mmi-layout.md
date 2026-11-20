# 动态布局算子验证记录 — desktop-54h5mmi

| 机器 | 构建 |
|---|---|
| Intel Core Ultra 9 285H，WSL2 6.6.87.2 | GCC 14.2，Release 与 ASan/UBSan |

## CPU-06 动态执行

源码身份为 `b26b8d6c` 加当前工作树。Release 先运行 `CpuDynamicOperators.MaterializesDynamicGraphThroughPreparedPlan`，再扩大至布局、Cast、Softmax、图与执行层，以及 P4-B 和 W16 回归。原始日志与 XML 保存于 `benchmark-results/operators/layout/20261120T014843Z_b26b8d6c_desktop-54h5mmi_cpu06-dynamic/`。

| 验收 | 结果 |
|---|---|
| 动态符号图 | 同一计划绑定两种长度，实际执行六步。核对转置后的坐标和独立概率公式。另验收外部输出承接 Reshape 的 unknown 推导维。 |
| 聚焦 Release | 动态图、Cast 位合同和 Softmax 推导，19/19。 |
| 扩展 Release | 图、lowering、执行绑定、P4-B 至 2048 token 和 W16，463/463。 |
| ASan/UBSan | 聚焦 19/19，LSan 关闭。 |

**Accepted（低层 graph/ExecutionPlan 动态执行）**。验证覆盖有界布局和对应浮点 dtype 的 reference 路径。模型级回归以已有 P4-B/W16 路径为主。新图的性能测量由真实 consumer 负载触发。
