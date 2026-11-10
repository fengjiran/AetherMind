# FP16 批量转换方案

## 合同与职责

保持 Half 单值转换作为数值权威：widening 精确，narrowing 固定 RNE + gradual underflow；NaN quiet 且保留 sign/可容纳的 payload；不改 rounding mode、FTZ/DAZ 或异常标志。

`dtypes/fp16_conversion.h` 定义 checked byte-span API 和无状态函数表，scalar batch 放在 Half 实现旁以复用同一整数算法。checked API 在写入前核对完整元素、输出长度、乘法溢出、非空函数和不重叠。直接调用 prepared 函数时，调用方负责这些前置条件。成功调用不分配 storage，源数据不要求 C++ Half 对象或类型对齐。

CpuBackend 按自身 effective features 选择两个方向的函数，inference 只传递选中的 widening，model 继续负责预算、分配、去重和 owner。默认 Backend 返回 scalar，不要求自定义 backend 实现新能力。函数表没有 backend/data 指针；不存在 per-element dispatch、全局 policy cache 或新 registry。

## ISA 实现

- F16C：16-bit 整数 mask 识别 NaN，先 OR `0x0200`，再 `_mm256_cvtph_ps`。有限值（含 half subnormal）均精确，Infinity 不 quiet。主循环 8 个元素，短尾部 scalar。
- AVX2 narrowing：按 FP32 exponent 对 normal/subnormal/tiny/overflow/NaN 分类。normal 保留 13 个 discarded bits；subnormal 使用完整 24-bit significand 一次 RNE。所有 lane 的移位量先限制到 14…24；允许结果进位到 `0x0400`。unsigned pack 后重排 qwords，恢复元素顺序和 sign bits。
- 普通 F16C narrowing 会报告 inexact/overflow 等，`_MM_FROUND_NO_EXC` 的立即数高位并不提供 VEX SAE，不能满足当前合同。AVX-512 的 SAE 转换保留为未来独立候选，不临时修改 MXCSR。

独立 TU 使用 `-mavx -mf16c` 或 `-mavx2`，不继承 FMA 要求；`-fno-lto` 和 `SKIP_UNITY_BUILD_INCLUSION` 保证 ISA 隔离。CPU policy 必须同时允许 AVX 和该方向的 ISA。构建开关与 feature 检测独立，未编译或不满足有效特性时使用 scalar。

## 验收与证据

测试覆盖每个 lane 中的全部 binary16 位模式、有限值 midpoint、随机 FP32 位模式、非对齐和各短尾部、保护页、空/错误/重叠范围、四种 rounding mode、FTZ/DAZ 和既有异常标志；权重与 Load 测试验证 callback 的调用位置、去重、预算与自定义 backend fallback。

scalar、仅 F16C、仅 AVX2 由 [最小构建工程](../../../tests/cmake/fp16_conversion_project/CMakeLists.txt) 重用真实源码与 ISA helpers 验证。ASan/UBSan 检查地址和移位安全；环境限制须在证据中单列。现有 compile-command gate 同时验证新 ISA sources、availability macros、Unity 和 LTO 边界。

性能分为 checked batch、含 allocation/去重的权重准备、实际 HF checkpoint 的 Load→Compile→Prepare 三层。同机 A/B 保存全部 repetitions，并补充同进程随机交错以观察进程间噪声；结论仅适用于记录中的 CPU、构建、输入和 cache 条件，见 [记录](benchmarks/desktop-54h5mmi-fp16-conversion.md)。
