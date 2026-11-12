# Packed artifact dtype 边界修复验证

- 日期：2026-10-29。
- 基线：`4f5c0cef` 加当前既有未提交变更；本轮不暂存、不提交。
- 范围：`PackedWeightCollection::Insert` 对 backend artifact 的 logical dtype 校验。

## 修复合同

`LogicalByteSize` 先将 `logical_dtype()` 读取到局部值。Undefined 继续表示无 logical-size premise；其他 dtype 在任何 `nbytes()` 查询前检查 raw lane 编码，scalable 与保留的 `0xffff` 返回 `InvalidArgument`。已验证的 element byte size 只查询一次，用于后续 checked size multiplication。

原有零字节 logical dtype、合法 zero-element shape、rank-0 scalar 与 fixed-vector 的语义保持。拒绝发生在 collection 插入或绑定 source 之前，失败不会发布 artifact；参数中的最后一个 owner 可正常释放，collection 可随后接受另一个 source 的合法插入。这里没有修改 RawWeightView、kernel tuple、packing 布局或数值算法。

## 运行证据

先运行单项 `PackedWeightCollectionOwnership.InsertRejectsScalableLogicalDTypesWithoutBindingSource`，再执行四个新增用例和扩大集合：

| 配置 | 新增集合 | 扩大回归集合（包含新增四项） | 结果 |
|---|---:|---:|---|
| Release，AVX2 编译关闭 | 4 GTest | 197 GTest | 全通过 |
| Debug，AVX2 编译开启 | 4 GTest | 197 GTest | 全通过 |
| ASAN / UBSAN / LSan | 4 CTest | 183 CTest（182 GTest + 1 fixture） | 全通过，无 sanitizer 报告 |

所有结果均无 failed、disabled、skipped、not-run。ASAN 使用 `halt_on_error=1:detect_leaks=1`，UBSAN 使用 `halt_on_error=1:print_stacktrace=1`；既有 interposer 专项从 sanitizer 集合排除。CTest inventory 与 JUnit 由 `verify_ctest_report.verify_run` 精确核对。

新增测试覆盖 scalable F16/F32/BF16（普通与 zero-element shape）、reserved lane、拒绝后的 owner 释放与 source/entries 状态保持及合法重试；也显式验证 Undefined/零字节兼容行为，以及 fixed-vector 的 31/32 字节 storage 边界。扩大集合覆盖原有 collection 零尺寸、溢出、ownership、真实 CPU packing、模型准备与生成路径。

审核时的独立 Release 程序重新编译后，scalable 和 reserved 两条 `SIGABRT` 路径均变为 `InvalidArgument`；正常 Float32、Undefined 与零字节样例继续成功。修复前后源码/日志保留在产物目录中。

## 原始产物与范围

产物目录：`build/validation/artifact-dtype-20261029/`。`run_checks.py` / `validation_runner.py` 保存 filter 和执行方式，`*.command.json` 保存精确命令与环境；JSON、inventory、JUnit 和 verified 报告记录实际运行结果。`summary.json` 保存 source/binary/config/policy hashes，`source.patch` 和 `untracked/` 保存 dirty source 快照。

冻结 policy 仅增加四项新 validation 用例：baseline/AVX 515→519、ASAN 417→421、TSAN 339→343。前三者的 live inventory 减去新增项后，必须匹配原集合的完整 count/SHA256。TSAN 仅更新已核对的预期名字，本轮未构建或运行，没有同步实现变更；具体差量在 `policy-delta.json`。

AVX 注册 inventory 来自本轮 Debug AVX-enabled 构建，不代表完整 Release AVX profile 重跑。测试分类、clang-format 与 `git diff --check` 通过；文档检查为 0 problems、29 条既有 warnings。

本轮没有运行全量单测、完整七 profile 矩阵或 benchmark，不作性能或新 dtype 执行能力声明。
