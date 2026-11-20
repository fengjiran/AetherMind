# CPU Softmax

独立 Softmax 已接通 CPU FP32、FP16 和 BF16 prepared reference。保留输入 dtype 和形状。带 rank 的输入在算子推导期检查 axis，绑定时按实际 rank 复验。负轴相对 rank，输入至少一维，空张量产生空输出。

每个轴切片先求最大值，使用 FP64 计算指数及其和，概率经 FP32 写回目标精度。`-Inf` 与有限输入同处一片时产生零概率。包含 NaN、`+Inf` 或全 `-Inf` 的切片产生 NaN。支持可验证的步长。输出须 injective，允许与输入完全相同的映射。执行无需额外 scratch。

Llama Dense 的 Attention 继续使用内部归约。此处的执行证据来自独立符号图，详见 [CPU-06 动态算子验收](../layout/README.md)。
