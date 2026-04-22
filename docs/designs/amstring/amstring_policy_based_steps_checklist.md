# amstring 详细开发步骤与 Checklist（Policy-Based Design 版本）

> 范围：基于 `basic_string_core + LayoutPolicy + GrowthPolicy` 的 policy-based 设计。  
> 目标：用统一的 core 承载算法与生命周期，用不同 layout policy 实现 stable generic 布局与 compact `char` 优化布局。  
> 开发方式：严格 TDD，按 milestone 推进。

---

# 0. 总体开发原则

## 0.1 TDD 基本循环

每个功能点都按以下顺序推进：

- [ ] 先写测试，明确预期行为。
- [ ] 写与 `std::basic_string` 的差分测试。
- [ ] 写边界测试。
- [ ] 写 invariant 测试。
- [ ] 写异常安全测试，必要时使用 failing allocator。
- [ ] 实现最小代码。
- [ ] 跑单元测试。
- [ ] 跑 ASan / UBSan / LSan。
- [ ] 跑随机差分测试。
- [ ] 功能稳定后再做 benchmark。
- [ ] benchmark 证明必要后再优化。

## 0.2 不提前做的事情

MVP 阶段禁止提前引入：

- [ ] COW / 引用计数 large string。
- [ ] SIMD 查找。
- [ ] safe over-read。
- [ ] branchless size 复杂技巧。
- [ ] 完整 allocator propagation。
- [ ] 所有 `std::basic_string` overload。
- [ ] stream / formatter。
- [ ] `char16_t / char32_t / wchar_t` 的极限性能优化。

## 0.3 核心 invariant

任意阶段、任意 API 修改后都必须满足：

- [ ] `data() != nullptr`
- [ ] `data()[size()] == CharT{}`
- [ ] `size() <= capacity()`
- [ ] `begin() == data()`
- [ ] `end() == data() + size()`
- [ ] moved-from object 是合法空字符串
- [ ] small / heap 状态可 O(1) 判断
- [ ] heap 状态下 `data()` 指向有效分配区域
- [ ] heap 状态下 `capacity()` 返回字符数，不是字节数
- [ ] capacity 不包含 category bits

## 0.4 policy-based 设计约束

- [ ] `basic_string_core` 只负责算法、生命周期、异常安全和 allocator 调度。
- [ ] `LayoutPolicy` 只负责底层存储布局与 metadata/category 编码。
- [ ] `GrowthPolicy` 只负责容量增长规则。
- [ ] 不允许在 core 中写死 small metadata 的 bit 布局。
- [ ] 不允许在 core 中写死端序相关逻辑。
- [ ] 所有端序优化仅能存在于 `compact_layout_policy<char>` 内部。

## 0.5 命名空间与目录结构约定

本项目采用以下约定：

**命名空间**：
- 使用 `aethermind::` 命名空间（与项目整体命名风格一致）

**目录结构**：
- 所有 amstring 相关头文件直接放在 `include/amstring/` 目录下
- 文件命名直观反映其内容

**目录布局**：
```text
include/amstring/
├── basic_string.hpp       # 对外 API 层
├── string.hpp             # 类型别名
├── string_fwd.hpp         # 前置声明
├── config.hpp             # 配置常量
├── core.hpp               # basic_string_core
├── layout_policy.hpp      # LayoutPolicy 接口
├── stable_layout_policy.hpp   # stable layout
├── compact_layout_policy.hpp  # compact layout（预留）
├── growth_policy.hpp      # GrowthPolicy
├── allocator_support.hpp  # allocator 辅助
├── invariant.hpp          # invariant 检查
└── char_algorithms.hpp    # Traits 辅助算法
```

---

# 1. Milestone 0：项目骨架与规格冻结

## 1.1 目标

建立工程骨架、测试框架和设计文档，冻结第一版实现边界。

## 1.2 目录结构

- [ ] 创建 `include/amstring/`
- [ ] 创建 `include/amstring/basic_string.hpp`
- [ ] 创建 `include/amstring/string.hpp`
- [ ] 创建 `include/amstring/config.hpp`
- [ ] 创建 `include/amstring/string_fwd.hpp`
- [ ] 创建 `include/amstring/core.hpp`
- [ ] 创建 `include/amstring/layout_policy.hpp`
- [ ] 创建 `include/amstring/stable_layout_policy.hpp`
- [ ] 创建 `include/amstring/compact_layout_policy.hpp`
- [ ] 创建 `include/amstring/growth_policy.hpp`
- [ ] 创建 `include/amstring/allocator_support.hpp`
- [ ] 创建 `include/amstring/invariant.hpp`
- [ ] 创建 `tests/unit/amstring/`
- [ ] 创建 `tests/differential/`
- [ ] 创建 `tests/fuzz/`
- [ ] 创建 `tests/allocator/`
- [ ] 创建 `tests/regression/`
- [ ] 创建 `tests/benchmark/amstring/`
- [ ] 创建 `docs/`

## 1.3 文档冻结

创建并完成以下文档：

- [ ] `docs/storage_layout.md`
- [ ] `docs/api_compatibility.md`
- [ ] `docs/tdd_plan.md`
- [ ] `docs/exception_safety.md`
- [ ] `docs/iterator_invalidation.md`
- [ ] `docs/benchmark_policy.md`

## 1.4 第一版设计结论

在文档中明确冻结：

- [ ] 第一版使用 policy-based 设计。
- [ ] `basic_string_core` 是统一算法与生命周期骨架。
- [ ] `stable_layout_policy<CharT>` 是 generic 主布局。
- [ ] 第一版先让所有 `CharT` 走 `stable_layout_policy`。
- [ ] `compact_layout_policy<char>` 在后续性能阶段引入。
- [ ] 第一版不做 COW。
- [ ] 第一版不做 safe over-read。
- [ ] 第一版不做 SIMD。
- [ ] 第一版先实现 `char`。
- [ ] 后续按顺序支持 `char8_t`、`char16_t`、`char32_t`、`wchar_t`。
- [ ] allocator 模板参数保留，MVP 只完整测试 `std::allocator`。

## 1.5 工具链

- [ ] 配置 CMake。
- [ ] 配置 C++20。
- [ ] 配置 GoogleTest 或 Catch2。
- [ ] 配置 Google Benchmark。
- [ ] 配置 ASan。
- [ ] 配置 UBSan。
- [ ] 配置 LSan。
- [ ] 配置 Debug build。
- [ ] 配置 Release build。
- [ ] 配置 RelWithDebInfo build。
- [ ] 配置 benchmark build。
- [ ] 配置 CI 或本地脚本。

## 1.6 验收标准

- [ ] 空项目可编译。
- [ ] 单元测试框架可运行。
- [ ] benchmark 框架可运行。
- [ ] sanitizer build 可运行。
- [ ] 文档中明确第一版边界。
- [ ] 没有开始写复杂字符串 API。

---

# 2. Milestone 1：定义 Policy 接口与默认选择机制

## 2.1 目标

先把 policy-based 设计的“骨架关系”固定下来，不急着写全部实现。

## 2.2 需要完成的类型

- [ ] 定义 `default_layout_policy<CharT>`
- [ ] 定义 `default_growth_policy`
- [ ] 定义 `basic_string_core<CharT, Traits, Allocator, LayoutPolicy, GrowthPolicy>`
- [ ] 定义 `stable_layout_policy<CharT>`
- [ ] 为后续预留 `compact_layout_policy<char>`

## 2.3 默认选择规则

第一阶段固定：

- [ ] `default_layout_policy<char>::type = stable_layout_policy<char>`
- [ ] `default_layout_policy<char8_t>::type = stable_layout_policy<char8_t>`
- [ ] `default_layout_policy<char16_t>::type = stable_layout_policy<char16_t>`
- [ ] `default_layout_policy<char32_t>::type = stable_layout_policy<char32_t>`
- [ ] `default_layout_policy<wchar_t>::type = stable_layout_policy<wchar_t>`

## 2.4 GrowthPolicy 接口

- [ ] 定义 `min_heap_capacity(required)`
- [ ] 定义 `next_capacity(old_cap, required)`

## 2.5 `LayoutPolicy` 接口清单

为 `stable_layout_policy<CharT>` 先定义统一接口：

- [ ] `storage_type`
- [ ] `init_empty(storage)`
- [ ] `is_small(storage)`
- [ ] `is_heap(storage)`
- [ ] `data(storage)`
- [ ] `size(storage)`
- [ ] `capacity(storage)`
- [ ] `set_size(storage, n)`
- [ ] `set_capacity(storage, n)`
- [ ] `init_small(storage, src, n)`
- [ ] `init_heap(storage, ptr, size, cap)`
- [ ] `heap_ptr(storage)`
- [ ] `destroy_heap(storage)`
- [ ] `check_invariants(storage)`

## 2.6 验收标准

- [ ] `basic_string_core` 能以 policy 参数形式实例化。
- [ ] `basic_string<CharT>` 能通过默认 policy 选择生成具体 core。
- [ ] 还没有引入 `compact_layout_policy<char>` 的复杂细节。
- [ ] 项目结构中责任边界已经清晰。

---

# 3. Milestone 2：实现 `stable_layout_policy<CharT>`

## 3.1 目标

实现 generic 主布局，使其适配多字符类型。

## 3.2 设计要求

- [ ] 采用 `heap_rep` 作为总存储预算。
- [ ] `kStorageBytes = sizeof(heap_rep)`。
- [ ] `kSmallArraySize = kStorageBytes / sizeof(CharT)`。
- [ ] `kSmallCapacity = kSmallArraySize - 1`。
- [ ] `small[kSmallCapacity]` 用作 metadata。
- [ ] metadata 编码为 `kSmallCapacity - size`。
- [ ] 不依赖最后一个 byte。
- [ ] 不依赖端序。

## 3.3 `storage_type`

- [ ] 定义 `heap_rep { CharT* data; size_t size; size_t capacity_with_tag; }`
- [ ] 定义 `union storage_type { CharT small[kSmallArraySize]; heap_rep heap; std::byte raw[kStorageBytes]; }`

## 3.4 状态判定与编码

- [ ] 定义 small / heap category 编码。
- [ ] `is_small()` 正确。
- [ ] `is_heap()` 正确。
- [ ] `size()` 能正确提取。
- [ ] `capacity()` 能正确提取。
- [ ] `data()` 能正确返回 small 或 heap 地址。
- [ ] `data()[size()] == CharT{}`。

## 3.5 初始化与销毁

- [ ] `init_empty()` 正确。
- [ ] `init_small()` 正确。
- [ ] `init_heap()` 正确。
- [ ] `destroy_heap()` 正确。
- [ ] moved-from 后可 reset 为 empty small。

## 3.6 Unit Tests

- [ ] `stable_layout_default_is_empty`
- [ ] `stable_layout_small_one_char`
- [ ] `stable_layout_small_literal`
- [ ] `stable_layout_small_max_size`
- [ ] `stable_layout_small_embedded_null`
- [ ] `stable_layout_heap_sso_plus_one`
- [ ] `stable_layout_heap_large_literal`
- [ ] `stable_layout_heap_embedded_null`
- [ ] `stable_layout_size_capacity_data`
- [ ] `stable_layout_invariants_small`
- [ ] `stable_layout_invariants_heap`

## 3.7 多 CharT 测试

以下类型都跑基础布局测试：

- [ ] `char`
- [ ] `char8_t`
- [ ] `char16_t`
- [ ] `char32_t`
- [ ] `wchar_t`

## 3.8 验收标准

- [ ] stable layout 对多 CharT 基础正确。
- [ ] 不依赖 endian trick。
- [ ] ASan / UBSan / LSan 通过。
- [ ] `LayoutPolicy` 接口完整可用。

---

# 4. Milestone 3：实现 `basic_string_core` 的基础读接口

## 4.1 目标

先让 core 成为一个可构造、可查询状态的最小骨架。

## 4.2 实现内容

- [ ] `basic_string_core() noexcept`
- [ ] `basic_string_core(const allocator_type&) noexcept`
- [ ] `basic_string_core(const CharT* s, size_type n, const allocator_type&)`
- [ ] `data()`
- [ ] `c_str()`
- [ ] `size()`
- [ ] `capacity()`
- [ ] `empty()`
- [ ] `check_invariants()`

## 4.3 约束

- [ ] core 中所有 size/capacity/data 的实现都必须通过 `LayoutPolicy` 调用。
- [ ] core 中不允许直接操作 `storage_` 的内部布局字段。
- [ ] `check_invariants()` 需要转调 `LayoutPolicy::check_invariants()` 并补充 core 层 invariant。

## 4.4 Unit Tests

- [ ] `core_default_construct`
- [ ] `core_construct_from_pointer_length_small`
- [ ] `core_construct_from_pointer_length_heap`
- [ ] `core_data_size_capacity`
- [ ] `core_empty`
- [ ] `core_c_str_null_terminated`
- [ ] `core_invariant_default`
- [ ] `core_invariant_small`
- [ ] `core_invariant_heap`

## 4.5 验收标准

- [ ] core 可作为独立底层对象使用。
- [ ] 所有基础读接口正确。
- [ ] 与 stable layout 结合运行正常。
- [ ] 仍未引入复杂生命周期语义。

---

# 5. Milestone 4：实现 `basic_string_core` 生命周期语义

## 5.1 目标

让 core 成为完整 RAII 类型。

## 5.2 实现内容

- [ ] `~basic_string_core()`
- [ ] `basic_string_core(const basic_string_core&)`
- [ ] `basic_string_core(basic_string_core&&) noexcept`
- [ ] `operator=(const basic_string_core&)`
- [ ] `operator=(basic_string_core&&) noexcept`
- [ ] `swap(basic_string_core&) noexcept`
- [ ] `reset_empty() noexcept`
- [ ] `destroy_heap_if_needed() noexcept`

## 5.3 语义要求

- [ ] small copy 复制整个 storage。
- [ ] heap copy 深拷贝。
- [ ] small move 等价于复制后 reset 源对象。
- [ ] heap move 窃取缓冲区。
- [ ] moved-from 对象必须合法。
- [ ] copy assignment 提供 strong exception guarantee。
- [ ] move assignment 尽量 `noexcept`。

## 5.4 Unit Tests

- [ ] `core_copy_construct_small`
- [ ] `core_copy_construct_heap`
- [ ] `core_move_construct_small`
- [ ] `core_move_construct_heap`
- [ ] `core_copy_assign_small_to_small`
- [ ] `core_copy_assign_small_to_heap`
- [ ] `core_copy_assign_heap_to_small`
- [ ] `core_copy_assign_heap_to_heap`
- [ ] `core_move_assign_small`
- [ ] `core_move_assign_heap`
- [ ] `core_swap_small_small`
- [ ] `core_swap_small_heap`
- [ ] `core_swap_heap_heap`
- [ ] `core_moved_from_is_valid`
- [ ] `core_destroy_heap_no_leak`

## 5.5 验收标准

- [ ] 无泄漏。
- [ ] 无 double free。
- [ ] 无 use-after-free。
- [ ] 生命周期操作全部通过 invariant 检查。
- [ ] allocator 默认路径下语义正确。

---

# 6. Milestone 5：封装对外 `basic_string` MVP

## 6.1 目标

构建用户可用的 `aethermind::basic_string<CharT>`。

## 6.2 类型定义

- [ ] `value_type`
- [ ] `traits_type`
- [ ] `allocator_type`
- [ ] `size_type`
- [ ] `difference_type`
- [ ] `reference`
- [ ] `const_reference`
- [ ] `pointer`
- [ ] `const_pointer`
- [ ] `iterator`
- [ ] `const_iterator`

## 6.3 MVP 构造与访问 API

- [ ] `basic_string()`
- [ ] `basic_string(const CharT*)`
- [ ] `basic_string(const CharT*, size_type)`
- [ ] `basic_string(std::basic_string_view<CharT, Traits>)`
- [ ] `basic_string(size_type count, CharT ch)`
- [ ] `basic_string(const basic_string&)`
- [ ] `basic_string(basic_string&&) noexcept`
- [ ] `~basic_string()`

- [ ] `operator=(const basic_string&)`
- [ ] `operator=(basic_string&&) noexcept`
- [ ] `operator=(const CharT*)`
- [ ] `operator=(CharT)`

- [ ] `size()`
- [ ] `length()`
- [ ] `capacity()`
- [ ] `empty()`
- [ ] `data()`
- [ ] `c_str()`
- [ ] `begin()/end()`
- [ ] `cbegin()/cend()`
- [ ] `operator[]`
- [ ] `at()`
- [ ] `front()/back()`

## 6.4 Unit Tests

- [ ] `string_default_construct`
- [ ] `string_construct_from_cstr`
- [ ] `string_construct_from_pointer_length`
- [ ] `string_construct_from_string_view`
- [ ] `string_construct_count_char`
- [ ] `string_construct_embedded_null`
- [ ] `string_copy_construct`
- [ ] `string_move_construct`
- [ ] `string_copy_assign`
- [ ] `string_move_assign`
- [ ] `string_assign_cstr`
- [ ] `string_assign_char`
- [ ] `string_data_c_str`
- [ ] `string_size_length_capacity`
- [ ] `string_begin_end`
- [ ] `string_operator_index`
- [ ] `string_at_out_of_range`
- [ ] `string_front_back`

## 6.5 差分测试

- [ ] `std::basic_string<CharT>` 构造/访问对照
- [ ] `at()` 异常行为对照
- [ ] embedded null 对照

## 6.6 验收标准

- [ ] `aethermind::basic_string<char>` 可用。
- [ ] `aethermind::string` 别名可用。
- [ ] 与 `std::string` 基础行为一致。
- [ ] 所有路径依然走 stable layout。

---

# 7. Milestone 6：实现 `GrowthPolicy` 与 Capacity API

## 7.1 目标

为所有修改类 API 打基础。

## 7.2 `GrowthPolicy`

- [ ] 实现 `min_heap_capacity(required)`
- [ ] 实现 `next_capacity(old_cap, required)`

## 7.3 core 容量接口

- [ ] `recommend_capacity(required)`
- [ ] `reallocate_and_copy(new_cap)`
- [ ] `reserve(new_cap)`
- [ ] `resize(count)`
- [ ] `resize(count, ch)`
- [ ] `clear()`
- [ ] `shrink_to_fit()`

## 7.4 行为要求

- [ ] `reserve()` 不改变 size。
- [ ] `resize(smaller)` 截断内容。
- [ ] `resize(larger)` 填充字符。
- [ ] `clear()` 后 size 为 0，保持合法 null terminator。
- [ ] `shrink_to_fit()` 允许 heap -> small。
- [ ] capacity 始终以字符数为单位。

## 7.5 Unit Tests

- [ ] `reserve_zero`
- [ ] `reserve_noop`
- [ ] `reserve_cross_sso_boundary`
- [ ] `reserve_keep_content`
- [ ] `resize_smaller_small`
- [ ] `resize_smaller_heap`
- [ ] `resize_larger_small`
- [ ] `resize_larger_heap`
- [ ] `resize_larger_fill_char`
- [ ] `clear_small`
- [ ] `clear_heap_keep_capacity`
- [ ] `shrink_small_noop`
- [ ] `shrink_heap_to_heap`
- [ ] `shrink_heap_to_small`

## 7.6 差分测试

- [ ] reserve sequence
- [ ] resize sequence
- [ ] clear sequence
- [ ] shrink sequence

## 7.7 验收标准

- [ ] capacity API 全部可用。
- [ ] 与 `std::basic_string` 的结果一致。
- [ ] 不要求 capacity 增长曲线与标准库完全相同。
- [ ] invariant 全部通过。

---

# 8. Milestone 7：实现 Append / Assign

## 8.1 目标

完成最核心的增长与重置路径。

## 8.2 实现内容

- [ ] `assign(const CharT* s, size_type n)`
- [ ] `append(const CharT* s, size_type n)`
- [ ] `append_fill(size_type count, CharT ch)`
- [ ] `source_overlaps(s, n)`
- [ ] `ensure_capacity_for_append(n)`

对外 API：

- [ ] `push_back()`
- [ ] `pop_back()`
- [ ] `append(const basic_string&)`
- [ ] `append(const CharT*)`
- [ ] `append(const CharT*, size_type)`
- [ ] `append(size_type count, CharT ch)`
- [ ] `append(string_view)`
- [ ] `operator+=`

## 8.3 自引用处理

- [ ] `s.append(s)`
- [ ] `s.append(s.data(), s.size())`
- [ ] `s.append(s.data() + offset, n)`
- [ ] `assign(self)` / `assign(self_subrange)` 正确

## 8.4 Unit Tests

- [ ] `assign_empty`
- [ ] `assign_small`
- [ ] `assign_heap`
- [ ] `assign_embedded_null`
- [ ] `assign_self`
- [ ] `append_empty`
- [ ] `append_small_to_small`
- [ ] `append_small_to_heap`
- [ ] `append_heap_to_heap`
- [ ] `append_embedded_null`
- [ ] `append_self_small`
- [ ] `append_self_heap`
- [ ] `append_self_subrange`
- [ ] `push_back_until_sso_boundary`
- [ ] `push_back_cross_sso_boundary`
- [ ] `pop_back_small`
- [ ] `pop_back_heap`

## 8.5 差分测试

- [ ] random assign
- [ ] random append
- [ ] random push_back / pop_back
- [ ] embedded null append
- [ ] self append

## 8.6 验收标准

- [ ] append / assign 结果正确。
- [ ] 自引用路径无 use-after-free。
- [ ] 所有路径通过 sanitizer。
- [ ] stable layout 下功能完整。

---

# 9. Milestone 8：实现 Erase / Insert / Replace

## 9.1 目标

完成主要修改 API。

## 9.2 实现顺序

- [ ] `erase`
- [ ] `insert`
- [ ] `replace`

## 9.3 core 实现

- [ ] `erase(pos, count)`
- [ ] `insert(pos, s, n)`
- [ ] `replace(pos, count, s, n)`

## 9.4 对外 API

- [ ] `erase(size_type pos = 0, size_type count = npos)`
- [ ] `erase(iterator pos)`
- [ ] `erase(iterator first, iterator last)`
- [ ] `insert(size_type pos, const CharT*)`
- [ ] `insert(size_type pos, const CharT*, size_type)`
- [ ] `insert(size_type pos, size_type count, CharT ch)`
- [ ] `insert(size_type pos, const basic_string&)`
- [ ] `replace(size_type pos, size_type count, const CharT*, size_type)`
- [ ] `replace(size_type pos, size_type count, const basic_string&)`
- [ ] `replace(size_type pos, size_type count, size_type count2, CharT ch)`

## 9.5 Unit Tests

- [ ] `erase_prefix`
- [ ] `erase_middle`
- [ ] `erase_suffix`
- [ ] `erase_all`
- [ ] `erase_empty_range`
- [ ] `insert_front`
- [ ] `insert_middle`
- [ ] `insert_back`
- [ ] `insert_embedded_null`
- [ ] `insert_self`
- [ ] `insert_self_subrange`
- [ ] `replace_same_length`
- [ ] `replace_shorter`
- [ ] `replace_longer`
- [ ] `replace_all`
- [ ] `replace_self`
- [ ] `replace_self_subrange`

## 9.6 差分测试

- [ ] random erase
- [ ] random insert
- [ ] random replace
- [ ] random mixed mutation
- [ ] self-reference mutation

## 9.7 验收标准

- [ ] 主要 mutation API 全部可用。
- [ ] 与 `std::basic_string` 行为一致。
- [ ] 所有操作后 invariant 通过。
- [ ] sanitizer 通过。

---

# 10. Milestone 9：实现 Compare / Find / 常用查询 API

## 10.1 目标

完成常用查询能力。

## 10.2 实现内容

- [ ] `compare()`
- [ ] `find(CharT ch)`
- [ ] `find(const CharT* s, pos, count)`
- [ ] `find(string_view, pos)`
- [ ] `rfind()`
- [ ] `starts_with()`
- [ ] `ends_with()`
- [ ] `contains()`

## 10.3 算法要求

第一版：

- [ ] 单字符查找使用 `Traits::find`
- [ ] compare 使用 `Traits::compare`
- [ ] substring find 使用朴素算法
- [ ] 不做 SIMD
- [ ] 不做 BMH
- [ ] 不做 memmem 特化

## 10.4 Unit Tests

- [ ] `compare_equal`
- [ ] `compare_less`
- [ ] `compare_greater`
- [ ] `compare_embedded_null`
- [ ] `find_char_exists`
- [ ] `find_char_missing`
- [ ] `find_string_exists`
- [ ] `find_string_missing`
- [ ] `find_empty_needle`
- [ ] `rfind_exists`
- [ ] `starts_with_true_false`
- [ ] `ends_with_true_false`
- [ ] `contains_true_false`

## 10.5 差分测试

- [ ] random compare
- [ ] random find char
- [ ] random find substring
- [ ] random rfind
- [ ] embedded null find

## 10.6 验收标准

- [ ] 查询结果与 `std::basic_string` 一致。
- [ ] `npos` 行为一致。
- [ ] 所有路径通过 stable layout 测试。

---

# 11. Milestone 10：放开多 CharT 支持

## 11.1 目标

在 stable layout 下支持全部目标字符类型。

## 11.2 顺序

- [ ] `char8_t`
- [ ] `char16_t`
- [ ] `char32_t`
- [ ] `wchar_t`

## 11.3 每种 CharT 必测项

- [ ] default construct
- [ ] pointer-length construct
- [ ] count-char construct
- [ ] embedded null
- [ ] small / heap 边界
- [ ] copy / move
- [ ] reserve / resize
- [ ] append / assign
- [ ] erase / insert / replace
- [ ] find / compare
- [ ] shrink_to_fit

## 11.4 重点检查

- [ ] size / capacity 按字符数计算
- [ ] null terminator 为 `CharT{}`
- [ ] `Traits::copy/move/compare` 的长度单位正确
- [ ] allocator 分配大小为 `(capacity + 1) * sizeof(CharT)`

## 11.5 差分测试

- [ ] `std::u8string`
- [ ] `std::u16string`
- [ ] `std::u32string`
- [ ] `std::wstring`

## 11.6 验收标准

- [ ] stable layout 支持多 CharT。
- [ ] 不依赖 endian trick。
- [ ] 多字符类型基础功能完整。

---

# 12. Milestone 11：Allocator 支持完善

## 12.1 目标

从默认 `std::allocator` 扩展到更完整的 allocator-aware 行为。

## 12.2 阶段一：默认 allocator

- [ ] `std::allocator` 路径全覆盖。
- [ ] small string 不进行 heap allocation。

## 12.3 阶段二：failing allocator

- [ ] constructor allocation failure
- [ ] copy constructor allocation failure
- [ ] reserve allocation failure
- [ ] append allocation failure
- [ ] assign allocation failure
- [ ] insert allocation failure
- [ ] replace allocation failure

## 12.4 阶段三：stateful allocator

- [ ] `select_on_container_copy_construction`
- [ ] `propagate_on_container_copy_assignment`
- [ ] `propagate_on_container_move_assignment`
- [ ] `propagate_on_container_swap`
- [ ] `is_always_equal`

## 12.5 验收标准

- [ ] allocator 行为清晰。
- [ ] failing allocator 测试通过。
- [ ] copy/move assignment 的 allocator 路径正确。
- [ ] 文档更新。

---

# 13. Milestone 12：引入 `compact_layout_policy<char>`

## 13.1 目标

在不破坏 public API 和 core 流程的前提下，为 `char` 引入 fbstring-like 紧凑优化布局。

## 13.2 前置条件

- [ ] stable layout 的 `char` 版本已经全部稳定。
- [ ] `aethermind::string` 的功能、差分测试、随机测试全部通过。
- [ ] benchmark 基线已建立。
- [ ] 已明确性能瓶颈。

## 13.3 需要完成的内容

- [ ] 定义 `compact_layout_policy<char>::storage_type`
- [ ] 定义 category 编码
- [ ] 定义 endian-aware codec
- [ ] 实现 `init_empty`
- [ ] 实现 `is_small` / `is_heap`
- [ ] 实现 `data` / `size` / `capacity`
- [ ] 实现 `init_small`
- [ ] 实现 `init_heap`
- [ ] 实现 `destroy_heap`
- [ ] 实现 `check_invariants`

## 13.4 约束

- [ ] 端序相关逻辑只能存在于 compact layout 内部。
- [ ] core 层零改动或最小改动。
- [ ] public API 零改动。
- [ ] sanitizer 构建不得启用危险 fast path。
- [ ] 不得引入 safe over-read，除非单独宏控并延后。

## 13.5 Tests

- [ ] 与 stable layout 同一套 layout 单测全部跑通。
- [ ] `basic_string_core<char, ..., compact_layout_policy<char>>` 生命周期测试全部通过。
- [ ] `aethermind::string` public API 全部通过。
- [ ] differential test 全部通过。
- [ ] random test 全部通过。
- [ ] fuzz regression case 全部通过。

## 13.6 验收标准

- [ ] `char` 默认 layout 可切换到 compact。
- [ ] 功能零回归。
- [ ] policy-based 设计证明有效：只换 layout，不改 core 流程。

---

# 14. Milestone 13：性能优化与基准测试

## 14.1 目标

对比：

- [ ] `std::string`
- [ ] `folly::fbstring`
- [ ] `aethermind::string`（stable）
- [ ] `aethermind::string`（compact）

## 14.2 Benchmark 分类

构造：

- [ ] default construct
- [ ] construct small literal
- [ ] construct heap
- [ ] construct embedded null

拷贝移动：

- [ ] copy small
- [ ] copy heap
- [ ] move small
- [ ] move heap

追加：

- [ ] push_back loop
- [ ] append small
- [ ] append heap
- [ ] append self

查询：

- [ ] find char
- [ ] find substring
- [ ] compare

真实 workload：

- [ ] 日志拼接
- [ ] 路径拼接
- [ ] HTTP header 解析
- [ ] JSON key/value 处理
- [ ] unordered_map key

## 14.3 记录指标

- [ ] ns/op
- [ ] allocations/op
- [ ] bytes allocated/op
- [ ] cycles/op
- [ ] instructions/op
- [ ] branch misses
- [ ] cache misses

## 14.4 优化项目

在 compact layout 基础上，逐项评估：

- [ ] small size fast path
- [ ] append fast path
- [ ] `memcmp` compare
- [ ] `memchr` find
- [ ] branchless size
- [ ] capacity growth 调整
- [ ] safe over-read（仅实验性，默认关闭）

## 14.5 验收标准

- [ ] `char` 小字符串场景明显优于 stable layout。
- [ ] 常见场景接近或不明显落后 `fbstring`。
- [ ] 优化都有 benchmark 依据。
- [ ] 无 correctness 回归。

---

# 15. Milestone 14：标准库集成与完善

## 15.1 运算符与工具

- [ ] `operator==`
- [ ] `operator!=`
- [ ] `operator<`
- [ ] `operator<=`
- [ ] `operator>`
- [ ] `operator>=`
- [ ] `operator<=>`
- [ ] `operator+`
- [ ] `std::hash`
- [ ] `std::swap`

## 15.2 可选

- [ ] `operator<<`
- [ ] `operator>>`
- [ ] `std::formatter`

## 15.3 验收标准

- [ ] 可作为 `unordered_map` key。
- [ ] 可排序。
- [ ] 可与 `std::basic_string_view` 配合使用。
- [ ] 文档标明已实现/未实现接口。

---

# 16. Release Checklist

## 16.1 正确性

- [ ] 全部 unit test 通过。
- [ ] 全部 differential test 通过。
- [ ] 全部 allocator test 通过。
- [ ] 全部 regression test 通过。
- [ ] fuzz 无已知 crash。
- [ ] ASan 通过。
- [ ] UBSan 通过。
- [ ] LSan 通过。
- [ ] Debug invariant 无失败。

## 16.2 API

- [ ] `aethermind::string` 可用。
- [ ] `aethermind::u8string` 可用。
- [ ] `aethermind::u16string` 可用。
- [ ] `aethermind::u32string` 可用。
- [ ] `aethermind::wstring` 可用。
- [ ] API compatibility matrix 已更新。
- [ ] 未实现 API 在文档中明确标注。

## 16.3 性能

- [ ] benchmark 已运行。
- [ ] benchmark 数据已保存。
- [ ] 对比 `std::string`。
- [ ] 对比 `fbstring`。
- [ ] 主要性能退化点已记录。
- [ ] 优化项有数据支持。

## 16.4 文档

- [ ] storage layout 文档更新。
- [ ] exception safety 文档更新。
- [ ] iterator invalidation 文档更新。
- [ ] API compatibility 文档更新。
- [ ] benchmark policy 文档更新。
- [ ] README 更新。
- [ ] 使用示例更新。

## 16.5 工程质量

- [ ] 编译无 warning。
- [ ] clang-tidy 主要检查通过。
- [ ] clang-format 已应用。
- [ ] public header 可独立 include。
- [ ] 不依赖未声明宏。
- [ ] 不泄露 detail API。
- [ ] CI 或本地脚本可一键验证。

---

# 17. 每日开发工作流 Checklist

每次开发一个小功能时：

- [ ] 明确本次只做一个功能点。
- [ ] 明确本次修改发生在 core / layout / growth 哪一层。
- [ ] 写对应单元测试。
- [ ] 写至少一个边界测试。
- [ ] 必要时写差分测试。
- [ ] 必要时写异常测试。
- [ ] 实现最小代码。
- [ ] 跑当前测试文件。
- [ ] 跑相关测试集合。
- [ ] 跑 sanitizer。
- [ ] 检查 invariant。
- [ ] 更新 API compatibility matrix。
- [ ] 提交前确认没有把 layout 细节泄漏到 core。

每次完成一个 milestone 时：

- [ ] 跑全部 unit test。
- [ ] 跑全部 differential test。
- [ ] 跑 sanitizer 全量。
- [ ] 更新文档。
- [ ] 记录风险点。
- [ ] 必要时跑 benchmark。
- [ ] 归档 regression case。

---

# 18. 优先级总览

## P0：必须先完成

- [ ] 项目骨架。
- [ ] policy 接口与默认选择机制。
- [ ] `stable_layout_policy<CharT>`。
- [ ] `basic_string_core` 基础读接口。
- [ ] `basic_string_core` 生命周期。
- [ ] `aethermind::string` MVP。
- [ ] `std::string` 差分测试。
- [ ] sanitizer 全量可跑。

## P1：核心功能

- [ ] reserve / resize / clear / shrink_to_fit。
- [ ] assign / append / push_back / pop_back。
- [ ] erase / insert / replace。
- [ ] compare / find。
- [ ] 多 CharT 的 stable layout 支持。

## P2：标准兼容与 allocator

- [ ] allocator 完整化。
- [ ] 运算符。
- [ ] hash。
- [ ] swap。
- [ ] 其他常用 overload。

## P3：性能冲刺

- [ ] `compact_layout_policy<char>`。
- [ ] benchmark 基线。
- [ ] char 专用优化。
- [ ] `memchr` / `memcmp` 快路径。
- [ ] branchless size（可选）。
- [ ] safe over-read（实验性）。

---

# 19. 建议的第一周执行计划

## Day 1
- [ ] 建立目录结构。
- [ ] 配置 CMake。
- [ ] 接入测试框架。
- [ ] 接入 sanitizer。
- [ ] 编写 `storage_layout.md` 初版。
- [ ] 编写 `api_compatibility.md` 初版。

## Day 2
- [ ] 定义 `default_layout_policy`。
- [ ] 定义 `default_growth_policy`。
- [ ] 定义 `basic_string_core` 模板骨架。
- [ ] 写 policy 接口测试桩。

## Day 3
- [ ] 实现 `stable_layout_policy<CharT>::storage_type`。
- [ ] 实现 `init_empty / is_small / data / size / capacity`。
- [ ] 写 empty/small 单测。

## Day 4
- [ ] 实现 `init_small / init_heap / destroy_heap`。
- [ ] 写 heap 单测。
- [ ] 加入 `char16_t / char32_t / wchar_t` 的基础 layout 测试。

## Day 5
- [ ] 实现 `basic_string_core` 基础读接口。
- [ ] 实现 `check_invariants()`。
- [ ] 跑 sanitizer。

## Day 6
- [ ] 实现 core 析构、copy/move 构造。
- [ ] 实现 core copy/move assignment。
- [ ] 写生命周期测试。

## Day 7
- [ ] 封装 `aethermind::basic_string<char>` MVP。
- [ ] 实现基础构造与访问 API。
- [ ] 写与 `std::string` 的基础差分测试。
- [ ] 整理第一周问题清单。

---

# 20. 最小可交付版本定义

MVP 完成条件：

- [ ] `aethermind::string` 支持 default/cstr/pointer-length/string_view/count-char 构造。
- [ ] 支持 copy/move/destructor。
- [ ] 支持 data/c_str/size/capacity/empty。
- [ ] 支持 begin/end/operator[]/at/front/back。
- [ ] 支持 reserve/resize/clear/shrink_to_fit。
- [ ] 支持 push_back/pop_back/append/operator+=。
- [ ] 支持 compare/find/starts_with/ends_with。
- [ ] 通过 ASan / UBSan / LSan。
- [ ] 通过基础差分测试。
- [ ] 有初步 benchmark 对比 std::string 和 fbstring。
- [ ] 文档中明确未实现 API。

MVP 不要求：

- [ ] `compact_layout_policy<char>` 已启用。
- [ ] 完整 allocator propagation。
- [ ] 完整 iterator overload。
- [ ] 完整 replace overload。
- [ ] COW。
- [ ] SIMD。
- [ ] safe over-read。
- [ ] stream / formatter。
- [ ] 所有 CharT 全部达到高性能。

---

# 21. 最终目标版本定义

最终稳定版应满足：

- [ ] 功能覆盖大部分 `std::basic_string` 常用 API。
- [ ] 支持 `char / char8_t / char16_t / char32_t / wchar_t`。
- [ ] `char` 版本默认走 `compact_layout_policy<char>`。
- [ ] 宽字符版本默认走 `stable_layout_policy<CharT>`。
- [ ] 小字符串性能接近或优于 `std::string`。
- [ ] 常见 `char` 场景性能接近 `fbstring`。
- [ ] 所有修改 API 通过随机差分测试。
- [ ] allocator 行为明确且测试覆盖。
- [ ] 文档完整。
- [ ] benchmark 可复现。
