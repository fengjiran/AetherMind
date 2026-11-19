# Cast 验证记录 — desktop-54h5mmi

| 项 | 值 |
|---|---|
| CPU | Intel Core Ultra 9 285H，WSL2，实际检测到 AVX2/FMA/F16C |
| Build | GCC 14.2。Release、AVX2/F16C-off 和 Debug ASan/UBSan |
| 来源 | `4a048033` 加 CPU-05/06 工作树 |

## CPU-06 首段：P4-B 生产执行

- 独立校准含 MHA untied、MHA tied、GQA、MQA 和 MHA near-limit 五个 FP16 checkpoint。两个独立 oracle 进程生成的 manifest 和文件身份一致。生产在 O1/plain 与 O2/identity-packed 对照 logits、K/V、固定排名和 greedy token，near-limit 为 2046-token Prefill 加两步 Decode，并通过实时 W16/A32/KV32 质量比较。
- 生产与独立 oracle 的最大归一化误差（以各自容差为单位）为 logits **0.648**、K **0.359**、V **0.252**，均小于 1。选核检查覆盖两种 Cast 方向、FP32 残差 Add 与 RMSNorm、同型 FP16 KV，以及 FP16 Linear 到 F32 logits。
- Release 聚焦测试 **334/334**，含 Cast 位合同、P4-B 长上下文、W16/KV16/BF16 A16、FP32 HF 轨迹和 C ABI。AVX2/F16C-off 和 ASan/UBSan 的短专项分别 **10/10**，包含完整短模型矩阵（LSan 关闭）。Release glibc interposer 在两种 preset 下均观察到 prepared Decode 的 malloc-family 调用为零。
- 官方 GQA 模型 runner 使用 `--weights keep-fp16 --activation-dtype f16 --kv-dtype f16 --mode allocations`。两种 preset 生成 `[8, 12, 8]`，一轮 Prefill 与两步 Decode 的 Runtime allocator 调用为零。
- 重建独立 CLI 后，strict FP32 CLI/Session 对照 **48/48**。CTest 的 P4-B fixture 检查与短模型验收入口均通过。

**Accepted（当前 CPU 模型路径和限定数值验收）**。Cast 使用 reference 与现有转换合同，本批记录数值和分配证据。AVX-512 原生执行和代表性大模型另行采集。原始资料保存在 `benchmark-results/operators/cast/20261119T033819Z_4a048033_desktop-54h5mmi_cpu06-p4b/`。
