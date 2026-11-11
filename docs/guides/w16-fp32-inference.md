# CPU W16 权重与 FP32 执行

状态为 Current。C++ `ExecutableModel::Load` 提供 CPU Llama Dense 的 F16/BF16 矩阵执行，支持 O1/plain 和 O2/identity-packed。矩阵、Embedding 和 LM head 保留选定的 16-bit 格式，Final/Input/PostAttention Norm 在准备期转为 F32，activation、residual、KV 和 logits 为 F32。

Runtime 使用模型几何和 F32 KV 创建，并保持稳定地址。通过显式 policy 选择权重精度。

```cpp
ModelLoadOptions options{
        .preset = CompilePreset::kO1Plain,
        .target = DeviceType::kCPU,
        .weights = WeightPolicy::kKeepBf16WeightsFp32Activation,
        .limits = {.max_model_payload_bytes = 512U * 1024U * 1024U},
};
auto model = ExecutableModel::Load(runtime, model_dir, options);
```

F16 模式选择 `kKeepFp16WeightsFp32Activation`。`kO2Packed` 使用 W16 identity artifacts 和 QKV/GateUp composite，布局保留两字节元素。已有 F32 B-panel 的布局身份仍描述其 F32 物理格式。

## 准备与所有权

所有矩阵的实际 dtype 必须匹配 policy，Norm 源接受 scalar F32/F16/BF16。缺省、auto 和已识别 F32/F16/BF16 hint 作为来源信息保留，实际矩阵决定准入。未知 hint、非 scalar 或不匹配的矩阵在转换前拒绝。独立 LM head 保留其 source view，缺少独立 head 时复用 embedding binding。

inference 冷路径取得 Backend 的两个 widening callback，调用 `ConvertRawModelWeightsToFp32` 的 Norm-only scope。所有 metadata、checked bytes、retained owner 账本和 cap 先检查，随后分配独立 Norm Buffer。相同 Norm source 由 owner identity、地址、bytes、dtype、shape 去重。矩阵与 Norm 共用 shard 时，结果矩阵继续保活整个 shard，Norm 使用新的 F32 owner。

`max_model_payload_bytes` 计入 retained owned backing、转换 Norm、inline constants 和 packed artifacts。Mapped/external 分别报告，scope 转换使用的峰值指标是 source owned backing 与新 Norm payload 的合计。未知 retained backing 遇有限 cap 时返回 FailedPrecondition。实际 RSS 和映射驻留页由独立测量观察。

## Kernel 与数值

Embedding 使用显式 F32 output spec，字节范围和 alias 在准备期检查，token vocabulary bounds 在执行时检查，每个 token 读取一行原始 W16 bytes。默认 reference 可在全部 CPU feature 禁用时运行，effective AVX+F16C 或 AVX+AVX2 分别选择 F16/BF16 行级 SIMD widening，短尾部复用 scalar。

W16 Linear/LM head reference 支持预验证的元素单位 row/column strides，原始 weight 通过 byte-safe 读取拓宽。累加按 K 升序执行 F32 product 与 F32 add，源文件禁用 FMA contraction。QKV/GateUp 的 identity-packed reference 复用相同 reduction。Norm、RoPE、Attention 和 KV 采用现有 F32 执行合同。

模型回归以同一 reduced checkpoint 的 P1 FP32 compatibility 为对照，tiny fixture 使用 `abs_error <= 1e-5 + 1e-4 * abs(reference)`。top-1 margin 大于该步 logits 最大绝对误差的两倍时要求 argmax 一致。算子额外使用独立 double oracle 与 gamma 误差界。P1 的 double 累加 reference 与 W16 F32 累加分别按各自合同验证。

O1/O2 的测试覆盖 Prefill、多次 Decode、Generate、MHA/GQA、tied/untied、Norm source dtype、EOS、容量、失败重试、owner lifetime 和 binding 完成后的 Decode malloc-family 零分配。当前实现与验证矩阵见 [P2 提案](../improvement-plan/11-fp16-bf16-execution-support.md)，单机性能记录见 [Linear](../operators/linear/benchmarks/desktop-54h5mmi-linear.md) 和 [Embedding](../operators/embedding/benchmarks/desktop-54h5mmi-embedding.md)。

Linear 当前交付 reference 路径，吞吐与准备/资源指标按记录中的具体 workload 解释。C ABI v1 与独立 inference CLI 使用其既有 strict FP32 准入。
