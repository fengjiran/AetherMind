# CPU Cast 与 FP16 A16

CPU-06 首段为 CPU O1/plain 和 O2/identity-packed 打通 Llama Dense 的 FP16 计算、FP16 KV 与 FP32 跨层 residual。C++ 调用选择 `WeightPolicy::kKeepFp16WeightsFp32Activation`，同时指定 `ModelLoadOptions::activation_dtype = DataType::Float(16)` 和同型 `kv_dtype`。Runtime 的物理 KV dtype 必须与之相同。矩阵保留 FP16，Norm 权重拓宽到 FP32。

| 边界 | 合同 |
|---|---|
| Embedding | FP16 权重直接拓宽到 FP32 residual |
| Attention、MLP 和最终 Norm | 读取 FP32 residual，完成 FP32 RMSNorm 后显式 Cast 到 FP16 |
| Q/K/V、RoPE、Attention、SiluMul | 复用 FP16 端口合同。Q/K 保留 Linear 和 RoPE 两处窄化，KVUpdate 同型写入 |
| Attention 输出投影、MLP down projection | FP16 输出完成舍入，再拓宽到 FP32，与跨层 residual 执行 FP32 Add |
| LM head | FP16 输入与权重，直接输出 FP32 logits |

Cast 动态 CPU entry 支持 F32→F16 和 F16→F32。保留形状、固定 RNE、gradual underflow、带符号零和 NaN quiet 合同。连续视图走原有 batch conversion reference，支持合法非连续、rank-zero 和空视图。不同 dtype 的输入输出存储须互斥，输出布局须可证 injective，校验在 prepared params 写入前完成。其他 Cast 方向在语义层和常量折叠中可表达。

A16 图沿用 unfused residual 路径。验证使用独立 FP32 residual oracle，对比 logits、KV、greedy token 与 commit 水位。见 [P4-B 验证范围](../../improvement-plan/11-fp16-bf16-execution-support.md#46-p4-bfp16-计算与-fp32-residual)。
