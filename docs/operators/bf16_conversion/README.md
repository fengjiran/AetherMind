# BF16 / FP32 批量转换

这是 dtypes/backend 数值 primitive，遵循 [算子开发与优化工作流](../../guides/operator-development-workflow.md)。当前提供双向 portable scalar、AVX2 双方向实现与 P1 准备路径接入；不新增 GraphOp、Cast pass、kernel registry 或 P2 W16/A32 kernel。

## 数值与范围合同（O0/O1）

- BF16→FP32 精确保留所有有限值、subnormal 和 signed zero；FP32→BF16 固定 RNE + gradual underflow，舍入溢出到同号 Infinity。
- 两个方向均保留 NaN sign 和可容纳的 payload，设置 quiet bit。单值 `BFloat16::operator float()` 与批量 widening 位结果一致（同样 quiet signaling NaN）；包装类型 narrowing 与批量 narrowing 位结果一致。
- 原始数据用 `memcpy` 读取和写入，允许任意字节对齐，不假设存在 `BFloat16` / `float` 数组对象。只做整数位运算，不依赖或改变 rounding mode、FTZ/DAZ 与异常标志。
- checked byte-span API 要求完整源元素、精确目标长度、无重叠，检查乘法溢出与非空函数；错误发生在写入前。空范围允许空指针/函数；成功路径不分配 storage。

接口定义见 [bf16_conversion.h](../../../include/aethermind/dtypes/bf16_conversion.h) 和 [公共 API](../../api/public-api.md)。直接调用 trusted scalar/callback 时，调用方负责有效内存、可表示字节数和无重叠。

源码组织与 FP16 对齐。[bfloat16.cpp](../../../src/dtypes/bfloat16.cpp) 集中实现单值转换和两个 scalar batch 函数，共享本文件私有的整数转换 helper，使 NaN 和舍入规则由同一处实现。[bf16_conversion.cpp](../../../src/dtypes/bf16_conversion.cpp) 承担 checked byte-span 入口及范围校验。

## 函数选择与生产调用路径

`Bf16ConversionFuncs { to_fp32, from_fp32 }` 是无状态函数表，不借用 backend。默认 Backend 返回 scalar；CpuBackend 用自身的 effective features 进入 `SelectBf16Conversions`：AVX policy 基座 + AVX2 时两个方向选择 AVX2 整数实现，否则 scalar。不重新检测硬件，也不要求 BF16 ISA。

```text
ExecutableModel::Load
  → Backend::GetHostBf16Conversions
  → RawWeightFp32ConversionOptions::bf16_to_fp32
  → ConvertRawModelWeightsToFp32
  → 每个唯一 BF16 tensor 调用一次 prepared widening
  → 现有 FP32 Compile / Prepare / Session
```

model 继续承担 validation、budget preflight、分配、去重和 owner；不依赖 CPU 实现，删除其私有 BF16 按元素转换。F16/BF16 callback 独立，两个配置项均须非空。P1 的最终 retained backing 统计、tied storage 和错误回滚合同保持。

## 验收与后续边界

数值回归覆盖全部 65,536 个 BF16 pattern、有限 midpoint/邻点、确定性随机 FP32 位模式、NaN payload、signed zero 与 underflow/overflow；范围回归覆盖非对齐、零长度、长度错误、重叠、canary 与保护页。浮点环境回归覆盖 rounding mode、既有异常标志与 x86 FTZ/DAZ；模型与 Load 测试覆盖独立 callback、去重、分配前拒绝、scalar fallback 和原有 FP32 生成轨迹。

验证命令与本机结果记入 [证据记录](benchmarks/desktop-54h5mmi-bf16-conversion.md)。AVX2 双方向实现已落地并通过与 scalar 的位级对拍（O2 正确性）；本机 batch benchmark 已记录，但尚未建立性能门禁或推导模型/生成吞吐结论（O3/O4 后续独立验收）；未来优化必须保持本合同并独立验证 prepared/真实准备路径，不能仅凭 native BF16 指令名称替换 reference。
