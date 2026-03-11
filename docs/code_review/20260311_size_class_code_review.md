# 代码审查报告

## 基本信息
- 审查日期: 2026-03-11
- 审查对象: `ammalloc/include/ammalloc/size_class.h`
- 风险级别: 🔴 Deep
- 审查依据: `docs/code_review/code_review_guide.md`

## 快速门禁结果
- [x] 聚焦单测通过: `./build/tests/unit/aethermind_unit_tests --gtest_filter="*SizeClass*"` (12/12)
- [x] 相关联动测试通过: `./build/tests/unit/aethermind_unit_tests --gtest_filter="*ThreadCache*:*CentralCache*"` (32/32)
- [x] LSP 诊断: `ammalloc/include/ammalloc/size_class.h` 无诊断
- [ ] 格式化检查: 本次未修改源码，未触发
- [ ] 静态分析: 本次未执行 clang-tidy

## 审查范围与证据
- 头文件实现: `ammalloc/include/ammalloc/size_class.h`
- 调用点: `ammalloc/include/ammalloc/thread_cache.h`, `ammalloc/src/thread_cache.cpp`, `ammalloc/src/central_cache.cpp`, `ammalloc/src/ammalloc.cpp`
- 单测/基准: `tests/unit/test_size_class.cpp`, `tests/benchmark/benchmark_size_class.cpp`
- 宏语义: `include/utils/logging.h`

## 维度审查结果

### P0 严重问题（必须修复）
- 无。

### P1 中等问题（建议修复）
1. `SafeSize()` 注释语义与实现行为不一致。
   - 位置: `ammalloc/include/ammalloc/size_class.h:185`, `ammalloc/include/ammalloc/size_class.h:190`
   - 证据: 注释描述越界返回 `0`；实现在越界时先执行 `AM_CHECK(false, ...)`。
   - 关联证据: `include/utils/logging.h:35` 的 `AM_CHECK` 失败会 `std::abort()`。
   - 影响: API 名称和注释给出“安全回退”预期，但实际为致命检查，易导致调用方误用。
   - 建议: 二选一保持一致
     - 方案 A: 明确文档为“调试/契约检查接口，越界即终止”；
     - 方案 B: 改为真实非致命回退（越界不终止，返回 0）。

### P2 轻微问题（后续优化）
1. `CalculateBatchSize()`/`GetMovePageNum()` 的输入前置条件未显式声明。
   - 位置: `ammalloc/include/ammalloc/size_class.h:224`, `ammalloc/include/ammalloc/size_class.h:257`
   - 证据: 当前调用链以 class-aligned size 传入（`ammalloc/include/ammalloc/thread_cache.h:45`, `ammalloc/src/central_cache.cpp:302`），但接口注释未明确。
   - 建议: 在头文件补充前置条件（例如 `size == RoundUp(size)`）或加 debug 断言。

2. 无效输入契约测试覆盖不足。
   - 位置: `tests/unit/test_size_class.cpp`
   - 已覆盖: 主要覆盖有效输入与边界（含 round-trip、碎片率、batch/page 上下界）。
   - 缺口: 未显式固定 `RoundUp(MAX_TC_SIZE + 1)`、`CalculateBatchSize(0)`、`GetMovePageNum(0)`、`SafeSize(out_of_range)` 的行为契约。
   - 建议: 增加 3-4 个窄用例锁定行为，防止后续回归。

## 正向结论
- `Index`/`Size`/`RoundUp` 映射在当前配置下逻辑一致，`Size(Index(s)) >= s` 由单测大范围验证。
- Batch 和 Page 计算路径满足当前调用约束，未发现立即性 correctness bug。
- 现有基准覆盖 `Index/Size/RoundUp/CalculateBatchSize/GetMovePageNum` 热路径。

## 结论
- 状态: 🟡 有条件通过（建议先收敛 P1 契约一致性，再补 P2 契约测试）
- 建议优先级:
  1. 先统一 `SafeSize()` 的注释与行为语义。
  2. 再补无效输入契约测试，固化边界行为。
