# RMSNorm 基准记录 — desktop-54h5mmi

| 项 | 值 |
|---|---|
| CPU | Intel Core Ultra 9 285H；客体可见 16 logical CPU；AVX2/FMA/F16C，无 AVX-512 |
| OS / kernel | Ubuntu 24.04.1 LTS / 6.6.87.2-microsoft-standard-WSL2 |
| Compiler / build | GCC 14.2.0；build，Release，C++20；sanitizer=none |
| 备注 | WSL2；默认 CPU feature policy；本条只记录实际选核与数值回归，不含性能采样 |

## A16 实际选核身份与验证边界核对

- commit：`cab0c145`，dirty（产品/路线图及证据说明修订，无 kernel 或 priority 变更）。
- 命令：`cmake --build build --target aethermind_unit_tests -j 4`；实际选核探针 `/tmp/aethermind-a16-dispatch-probe`，源码及编译命令保留于 raw；聚焦回归 `./build/tests/unit/aethermind_unit_tests --gtest_filter='A16LoadAdmission.*:LinearA16Dispatch.*:LinearA16Avx2Fma.*:CPUKernelRmsNormA16.*:A16ModelAcceptance.*'`。
- 配置：真实 `DetectCpuCapabilities` 返回 `{SSE4.1, AVX, AVX2, FMA, F16C, AVXVNNI}`。探针只调用正式 `PrepareKernel` / `GetPackingRecipe`，不执行 kernel。
- 结果：Prefill/Decode 的 8 个 RMSNorm selector 全部 resolve 为 AVX2/FMA；weight=F32、output=activation。plain recipe 为 `none`/alignment 0，packed 为 `cpu_identity`/alignment 64，packing query 与 resolved recipe 一致。各 phase 的实际名称相同：

| activation / output | plain kernel | identity-packed kernel |
|---|---|---|
| BF16 | `cpu::rmsnorm_bf16_w32_avx2_fma` | `cpu::rmsnorm_bf16_w32_packed_identity_avx2_fma` |
| F16 | `cpu::rmsnorm_f16_w32_avx2_fma` | `cpu::rmsnorm_f16_w32_packed_identity_avx2_fma` |

- 回归：27 项通过、0 skip，其中 RMSNorm A16 9 项，含独立 oracle、AVX2/tail、BF16 underflow/normal 边界、alignment 与 packed identity 校验；另有 BF16 O1/O2 模型路径与准入回归。本轮不重跑 2048 near-limit 校准。
- raw：`benchmark-results/operators/rmsnorm/20261117T013309Z_cab0c145_desktop-54h5mmi_a16-dispatch-audit/`（逐 selector TSV、探针源码、metadata、测试 XML/log；本地 gitignored 留存，无远端 retention）。
- 结论：**Accepted（本机选核身份与限定回归）**。native BF16/FP16 的实际执行、数值与性能均为 **not-run**；本次不调整 priority，不推导加速收益。
