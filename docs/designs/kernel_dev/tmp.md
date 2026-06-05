必须修正
1. ISA / fallback 契约与当前实现不一致
文档 2.4.4 / 2.5 说必须有 scalar fallback、runtime ISA dispatch、不能在不支持 ISA 的机器执行高级指令。但当前 CpuRmsNormKernel 注册名是：
cpu::rmsnorm_f32_scalar
实现却直接使用 AVX2/FMA intrinsic：
__m256
_mm256_fmadd_ps
_mm256_loadu_ps
所以现在它不是 scalar kernel，也没有 runtime dispatch。建议文档明确标注：
- 当前实现：AVX2+FMA-only baseline
- 待补：scalar fallback、AVX2 dispatch、正确的 KernelSelector::isa 注册
2. Reference double 精度契约与测试/benchmark 实现不一致
文档 2.3.1 要求：
double(X) * double(X)
但当前 reference 代码实际是 float 乘法后再累加到 double：
mean_square += row_in[i] * row_in[i];
这不是严格 double reference。应改为：
const double x = static_cast<double>(row_in[i]);
mean_square += x * x;
并且 inv_rms 也应保持 double 到最终 cast。
3. epsilon 校验边界已移动到 entry
Operator 层校验 `params_.epsilon_ <= 0.0F`，`CpuRmsNormKernelEntry` 负责从 attrs 解码并校验 `std::isfinite(epsilon) && epsilon > 0`。
`CpuRmsNormKernel` 是已验证 `RmsNormFp32KernelArgs` 上的 typed compute primitive；直接调用方必须保证 `epsilon_` 有效。
4. benchmark correctness 契约与当前 benchmark 不一致
文档 2.3.3 / 2.7.2 要求 benchmark 记录：
max_abs_diff, max_rel_diff
当前 benchmark 只记录：
items/s, bytes/s, GFLOP/s
建议不要把 correctness 放进 benchmark 主循环。更合理写法是：benchmark 启动前做一次 reference 校验，或把这条改成“benchmark 可选记录 correctness，单元测试必须覆盖”。
建议修正
5. in-place 契约尚未测试
文档允许：
output == input 精确 in-place
当前实现看起来是安全的，因为 Phase 1 先完整读完 input，Phase 3 每个元素 load 后立刻 store，不会读回已覆盖数据。但目前单测没有覆盖，应补 output 与 input 同 buffer 的 case。
6. shape 覆盖要求比现有测试更强
文档要求覆盖 4096 / 8192 / 11008、非 2 的幂、质数 hidden、tail。当前 unit test 主要是 hidden=4；benchmark 覆盖 4096/8192，但没有 11008，也没有质数 hidden。建议把这部分标为“验收清单”，不要让读者误以为已经满足。
7. Operator shape 校验缺少 seq_len > 0
RmsNormOp::CheckShapes() 校验了 hidden_size > 0，但没有显式校验 seq_len > 0。Kernel 层有校验。若契约要求 Operator 层负责 shape 校验，应补上 input.shape_[0] > 0。
8. “Phase 1” 用词容易混淆
文档里 “Phase 1 约束” 指项目阶段，但实现里 Phase1 是 sum of squares。建议改成“当前阶段约束”或“AetherMind Phase 1 约束”。
