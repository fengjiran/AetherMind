# SiluMul AVX2/FMA

AVX2 descriptor 复用原有 builder 的 broadcast、别名、地址范围和 dtype 校验。Exact in-place 可用，因为每组的 gate 和 up 均先读取，再写该组输出。非连续或发生 broadcast 的 invocation 保留通用 scalar 路径。

FP32 body 每组处理 8 个元素。对 `exp(-abs(gate))` 做二进制范围缩减，余项落在约 ±ln(2)/2，使用七阶 Taylor 多项式和精确的 2^n 缩放。正 gate 使用 `gate / (1 + exp)`，负 gate 先乘 exp 再除。最后与 up 单独相乘，避免改变 SiLU 到 Mul 的舍入位置。

向量域要求 `abs(gate)` 位于 [2^-32, 16]，`abs(up)` 位于 [2^-32, 2^100]，两者都允许带符号零。上述范围内非零乘积为 normal FP32，并避开 overflow。超出域的组、非有限值、不到 8 个元素的尾部和非 round-to-nearest 的调用都使用原有 scalar 公式。ISA 翻译单元关闭 fast-math 和隐式 contraction，显式 FMA 用于 exp 多项式。

BF16 用整数扩展读取 16 位 payload，走同一 FP32 算术，再用整数 RNE 写回。若计算结果距 BF16 舍入中点不超过 32 个 FP32 尾数单位，整组交给 scalar 重新计算。该保护在当前 libm 和主机上用全部 BF16 gate 编码及 up 的 128 个尾数位置对拍。保留 NaN 分类与带符号零。该实现无需持久表、附加 workspace 或 per-call allocation。

测试用独立 FP64 公式检查 FP32 快速域，误差预算为 6 个 FP32 epsilon 的相对误差。特殊值和 BF16 窄化与独立 scalar kernel 对照。模型级 O1/O2、轨迹与时延证据见 [benchmarks](benchmarks/)。
