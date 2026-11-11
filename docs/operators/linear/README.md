# CPU Linear

当前实现包含已有 F32 plain/packed kernels，以及 W16/A32 plain / identity-packed reference。W16 的 Linear、LM head、QKV 和 GateUp consumer 使用相同的 F32 reduction 合同，支持 CPU O1/plain 与 O2/identity-packed Llama Dense。

input 为 rank 至少 1 的 F32 activation，weight 为 `[N,K]` scalar F16/BF16，output 最后轴为 N，前导轴跟随 input。prepared args 使用 byte weight pointer、元素单位 strides 和已校验的 dtype。合法 rowwise strides、rank flatten、地址范围与 alias 由 binding-time helpers 验证。零 M/N 为无操作，零 K 写正零。

reference 以 `memcpy` 读取 W16，按 K 升序执行独立的 F32 product 和 add，TU 禁用 FMA contraction。输入 activation 的 F32 精度参与乘法。F32 reference 的 double 累加与 W16 F32 累加分别通过自己的数值合同和独立 oracle 验证。

O2 identity/composite artifacts 保留 W16 元素，QKV/GateUp 复用同一 reduction。当前 W16 路径属于 reference 实现，prepared/kernel 与模型准备/执行的性能观察见 [本机记录](benchmarks/desktop-54h5mmi-linear.md)。新 B-panel 的布局评估也在该记录中。

实现见 [Linear reference](../../../src/backend/cpu/kernels/linear/linear_w16_reference.cpp)，测试见 [Linear regression](../../../tests/unit/backend/cpu/kernels/test_cpu_linear_kernel.cpp)，模型使用见 [W16 指南](../../guides/w16-fp32-inference.md)。遵循 [算子工作流](../../guides/operator-development-workflow.md)。
