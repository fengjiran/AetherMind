# Raw weight dtype 边界修复验证

- 日期：2026-10-29。
- 基线：`4f5c0cef` 加既有未提交的资源预算、FP32 compatibility 变更；本轮继续不暂存、不提交。
- 范围：通用 raw-weight metadata 校验与 packing / model preparation 的错误传播。

## 修复合同

`RawWeightView::IsValid()` 在任何 dtype equality 或 `nbytes()` 查询前检查非零 bits 和固定正 lane 编码（1 至 `INT16_MAX`）。它拒绝 scalable、保留的 `0xffff` 编码和零 lane。`ValidateRawWeightView()` 再检查非零 element byte size，然后沿用 shape product 与 byte size 的 checked arithmetic。

合法 zero-element tensor、rank-0 scalar 和 fixed-vector element 保持原合同；不以 `bytes == 0` 一概拒绝空 tensor。`BuildWeightPackingRequests` 对存在但无效的 raw view 返回 `InvalidArgument`，无法解析的 binding 保留 `Internal`。Compiler 不因此承担完整 payload/shape materialization。

## 运行结果

先运行单项 `RawWeightValidation.RejectsScalableAndReservedLaneEncodings`，随后运行新增集合和受影响的既有集合：

| 配置 | 新增 | 相关回归 | 结果 |
|---|---:|---:|---|
| Release，AVX2 编译关闭 | 9 GTest | 442 GTest | 全通过 |
| Debug，AVX2 编译开启 | 9 GTest | 442 GTest | 全通过 |
| ASAN / UBSAN / LSan | 9 CTest | 428 CTest（426 GTest + 2 fixture） | 全通过，无 sanitizer 报告 |

结果均无 failed、disabled、skipped、not-run。ASAN 排除既有 interposer 专项，使用 `halt_on_error=1:detect_leaks=1`；UBSAN 使用 `halt_on_error=1:print_stacktrace=1`。CTest inventory 与 JUnit 通过 `verify_ctest_report.verify_run` 精确核对。

6 个直接合同用例覆盖 scalable F16/F32/BF16、reserved lane、零 lane/零字节 dtype、undefined/void、空 tensor、scalar 和 fixed-vector 字节计数。3 个消费者用例使用真实 CpuBackend，覆盖 bounded/unlimited prepack、默认无限制 plain Prepare 和 packed Prepare；异常输入均返回 `InvalidArgument` 且无 payload allocation，并在同一 Runtime/storage 上合法重试成功。此分配断言不排除 STL/诊断 metadata allocation。

审核时的独立 Release 复现也已重新编译运行：scalable 与 `Handle(64, 0)` 在 raw validation、prepack 和默认 Prepare 的六条路径，现在全部返回 `InvalidArgument`，不再触发 `SIGABRT` 或错误返回 OK。源码与修复前后日志保存在原始产物目录中。

## 原始产物与边界

产物目录：`build/validation/raw-weight-dtype-20261029/`。`run_checks.py` 保存精确新增/回归 filter，`*.command.json` 保存命令与环境；`*-final.json`、`*.inventory.json`、`*.junit.xml` 和 `*.verified.json` 保存实际结果。`summary.json` 记录 source/binary/config/policy hashes，`source.patch` 与 `untracked/` 保存 dirty source 快照。

冻结 policy 仅加入这 9 个无条件注册的 validation 用例：baseline/AVX 506→515，ASAN 408→417，TSAN 330→339；其余集合不变。baseline、Debug AVX-enabled 与 ASAN 的 live inventory 减去新增项后，必须匹配原 policy 的完整 count/SHA256。TSAN 仅由已验证的旧 inventory 加本轮无条件新增名字推导，记录于 `policy-delta.json`；本轮未构建或运行 TSAN，没有同步/所有权实现变更。

本轮没有运行全量单测、完整七 profile 矩阵或 benchmark，也不修改 dtype 构造规则、kernel dtype 支持范围、packing 物理布局或推理数值语义。
