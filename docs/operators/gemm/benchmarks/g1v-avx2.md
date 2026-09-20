# GEMM G1V AVX2 工作文件

- **算子索引**: [GEMM 实验记录与验证报告索引](../README.md)
- **状态**: In Progress / Implemented for opt-in diagnostics / Performance Not Run
- **门禁状态**: correctness [x] / production-path [ ] / 可追溯 [ ]

## 1. 目标与合同不变式

- 目标：为 plain FP32 Linear 的 Decode `M=1` direct-weight K-contiguous 路径提供 AVX2+FMA candidate，并保留 scalar-compatible fallback。
- 合同不变式：不改变 OperatorSchema、dtype、shape、alias、zero-size 或 overflow 语义。`RunGemmF32Reference` 保持 double-accumulation correctness oracle；AVX entry 仅直接处理 `M=1`、`K>0`、lhs-K/rhs-K/output-N unit stride。其他所有 reference-legal input 委托 `RunGemmF32ScalarOptimized`，其余 fallback 继续委托 reference。
- 关联代码 / 关联测试：`gemm_f32_avx2.cpp`、`gemm_internal.h`、`linear_entry.cpp`、`test_cpu_gemm_avx2.cpp`、`test_cpu_linear_kernel.cpp`。

## 2. 实验记录（追加式，失败保留）

### 2026-09-21 — EXP-001：G1V-A opt-in implementation

- 假设 / 预期机制：K-direction AVX2+FMA、four-output block 与 two independent 16-float accumulators can reduce Decode direct-weight GEMV instruction overhead relative to the scalar candidate.
- 验证命令（benchmark/test）与核心元数据（commit、dirty、CPU、OS/kernel、compiler+flags、raw artifact）：correctness build/test commands are recorded by this change; formal benchmark command and raw artifact are **NOT RUN**.
- Correctness：`direct AVX2 / feature-gated Linear dispatch / prepared Executor | PASS | 84 focused tests in scalar+AVX2 Release configuration; tests are not performance evidence`。
- 结果摘要：`Case | Baseline | Candidate | Delta | Raw artifact` — performance Not Run.
- 分析与决定：Implemented for opt-in diagnostics. No acceptance, priority promotion, or performance conclusion is authorized until paired scalar-to-AVX2 hot/streaming data and target-machine artifacts exist.

## 3. 正式结论（冻结区，按 commit@date 追加，不覆盖）

### uncommitted@2026-09-21 — G1V-A implementation boundary

- 判定：Needs More Data.
- 环境快照（核心 6 项；production 结论附全档）：performance environment and raw artifact are Not Run.
- correctness / numerical error 摘要：direct K/N tails, padded/unaligned buffers, output guards, zero dimensions, overwrite, scalar fallback, feature policy, prepared Linear, and Executor integration passed in focused validation.
- production-path benchmark 摘要：Not Run; registrations exist for direct AVX2 and prepared Linear AVX2/scalar-fallback hot and streaming paths.
- layout/alias/fallback、workspace/ownership、dispatch、并发（适用时）：descriptor requires `{AVX2, FMA}` and priority 20; scalar descriptor remains priority 10. The AVX entry does not add packing, workspace, allocations, threading, prefetch, or an execution-plan specialization layer.
- 未覆盖项与结论边界：no performance result, assembly inspection, PMU data, paired A/B, streaming measurement, or bare-metal evidence.

## 4. 门禁判定

- [x] correctness 与 safety 测试通过
- [ ] production-path benchmark 已运行且附核心元数据
- [ ] 结果可追溯（commit/机器/命令/raw artifact）
- [x] 未运行或未证明的内容已标注

## 5. 相关链接

- [CPU GEMM 优化方案](../cpu-gemm-optimization.md)
- [算子开发与优化工作流](../../../guides/operator-development-workflow.md)
