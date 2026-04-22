# amstring 详细开发步骤与 Checklist

> 目标：从零实现一个 `fbstring-like` 的高性能字符串库 `amstring`。  
> 范围：支持多 `CharT`，功能逐步对齐 `std::basic_string`，`char` 版本性能对标 `folly::fbstring`，开发模式采用 TDD。  
> 原则：先正确性，再完整性，再性能；先 generic core，再 `char` 专用优化。

---

## 0. 总体开发原则

### 0.1 TDD 基本循环

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

### 0.2 不提前做的事情

MVP 阶段禁止提前引入：

- [ ] COW / 引用计数 large string。
- [ ] SIMD 查找。
- [ ] safe over-read。
- [ ] branchless size 复杂技巧。
- [ ] 完整 allocator propagation。
- [ ] 所有 `std::basic_string` overload。
- [ ] stream / formatter。
- [ ] `char16_t / char32_t / wchar_t` 的极限性能优化。

### 0.3 核心 invariant

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

---

## 1. Milestone 0：项目骨架与规格冻结

### 1.1 目标

建立工程骨架、测试框架和设计文档，冻结第一版实现边界。  
融入 AetherMind 项目现有结构，amstring 测试文件单独子目录管理。

### 1.2 目录结构

头文件（扁平结构，公共/内部通过命名约定区分）：

- [x] 创建 `include/amstring/`
- [x] 创建 `include/amstring/basic_string.hpp`（公共 API）
- [x] 创建 `include/amstring/string.hpp`（公共 API）
- [x] 创建 `include/amstring/string_fwd.hpp`（公共 API）
- [x] 创建 `include/amstring/config.hpp`（内部）
- [x] 创建 `include/amstring/core.hpp`（内部）
- [x] 创建 `include/amstring/layout.hpp`（内部）
- [x] 创建 `include/amstring/category.hpp`（内部）
- [x] 创建 `include/amstring/growth_policy.hpp`（内部）
- [x] 创建 `include/amstring/char_algorithms.hpp`（内部）
- [x] 创建 `include/amstring/invariant.hpp`（内部）

单元测试（融入现有 `tests/unit/`，单独子目录）：

- [x] 创建 `tests/unit/amstring/`
- [x] 创建 `tests/unit/amstring/test_core_layout.cpp`（空测试，验证框架可编译）
- [x] 创建 `tests/unit/amstring/test_core_lifecycle.cpp`（空测试，验证框架可编译）
- [ ] 后续：`test_api_*.cpp`, `test_differential.cpp`

Benchmark（融入现有 `tests/benchmark/`，单独子目录）：

- [x] 创建 `tests/benchmark/amstring/`
- [x] 创建 `tests/benchmark/amstring/benchmark_construct.cpp`（空 benchmark，验证框架可编译）
- [x] 创建 `tests/benchmark/amstring/benchmark_vs_std_string.cpp`（空 benchmark）
- [ ] 后续：`benchmark_copy_move.cpp`, `benchmark_append.cpp`

设计文档（补充到 `docs/designs/amstring/`）：

- [ ] 补充 `docs/designs/amstring/storage_layout.md`
- [ ] 补充 `docs/designs/amstring/api_compatibility.md`
- [ ] 补充 `docs/designs/amstring/tdd_plan.md`
- [ ] 补充 `docs/designs/amstring/exception_safety.md`
- [ ] 补充 `docs/designs/amstring/iterator_invalidation.md`
- [ ] 补充 `docs/designs/amstring/benchmark_policy.md`

### 1.3 文档冻结

创建并完成以下文档：

- [ ] `storage_layout.md`：对象大小目标、Small/Heap 布局图、Category 编码方案
- [ ] `api_compatibility.md`：std::basic_string API 覆盖矩阵、已实现/未实现标记
- [ ] `tdd_plan.md`：测试策略、差分测试框架设计、invariant 定义
- [ ] `exception_safety.md`：strong/weak guarantee 定义、failing allocator 测试策略

### 1.4 第一版设计结论

在文档中明确冻结：

- [ ] 第一版采用 Small + Heap exclusive ownership。
- [ ] 第一版预留 Large category，但不启用特殊 Large 策略。
- [ ] 第一版不做 COW。
- [ ] 第一版不做 safe over-read。
- [ ] 第一版不做 SIMD。
- [ ] 第一版先实现 `char`。
- [ ] 后续按顺序支持 `char8_t`、`char16_t`、`char32_t`、`wchar_t`。
- [ ] `char` 性能优化后续通过 core 特化完成。
- [ ] generic core 优先保证多 `CharT` 正确性。
- [ ] allocator 模板参数保留，MVP 只完整测试 `std::allocator`。
- [ ] Small metadata 用 CharT 存储，不用 byte-level trick（第一版简化）。

### 1.5 工具链（利用现有项目配置）

AetherMind 项目已有：

- [x] CMake >= 3.28
- [x] C++20
- [x] GoogleTest（系统安装）
- [x] Google Benchmark（FetchContent）
- [x] ASan/UBSan/LSan 配置

amstring 直接复用：

- [ ] 验证 `tests/unit/amstring/*.cpp` 被 GLOB_RECURSE 收集
- [ ] 验证 `tests/benchmark/amstring/*.cpp` 被 GLOB_RECURSE 收集
- [ ] 验证 sanitizer build 可运行 amstring 测试

### 1.6 验收标准

- [ ] `include/amstring/` 目录存在，含骨架头文件。
- [ ] `tests/unit/amstring/` 目录存在，含空测试文件。
- [ ] `tests/benchmark/amstring/` 目录存在，含空 benchmark 文件。
- [ ] `storage_layout.md`, `api_compatibility.md`, `tdd_plan.md` 完成。
- [ ] 第一版设计决策在文档中明确记录。
- [ ] `cmake --build build --target aethermind_unit_tests` 通过。
- [ ] `cmake --build build --target aethermind_benchmark` 通过。
- [ ] 没有开始写复杂字符串 API（只骨架）。

---

## 2. Milestone 1：Generic Core Layout Prototype

### 2.1 目标

实现 `basic_string_core<CharT, Traits, Allocator>` 的基础 small / heap 布局。  
此阶段只要求 core 层可测试，不要求对外 `basic_string` 完整可用。

### 2.2 类型设计

- [ ] 定义 `detail::Category`
- [ ] 定义 `Category::Small`
- [ ] 定义 `Category::Medium`
- [ ] 定义 `Category::Large`
- [ ] 定义 `detail::MediumLarge<CharT>`
- [ ] 定义 `detail::basic_string_core<CharT, Traits, Allocator>`
- [ ] 定义 `size_type`
- [ ] 定义 `value_type`
- [ ] 定义 `allocator_type`
- [ ] 定义 `traits_type`

### 2.3 Small layout

推荐第一版 generic small metadata 策略：

```cpp
static constexpr size_t kStorageBytes = sizeof(MediumLarge<CharT>);
static constexpr size_t kSmallArraySize = kStorageBytes / sizeof(CharT);
static constexpr size_t kSmallCapacity = kSmallArraySize - 1;
```

Checklist：

- [ ] `kStorageBytes` 正确。
- [ ] `kSmallArraySize` 正确。
- [ ] `kSmallCapacity` 正确。
- [ ] small buffer 有 `kSmallCapacity + 1` 个 `CharT` 存储位。
- [ ] 最后一个 `CharT` 作为 metadata。
- [ ] small 有效字符区间为 `[0, kSmallCapacity)`。
- [ ] small 下 `data()[size()] == CharT{}`。
- [ ] metadata 不会覆盖有效字符。

### 2.4 Heap layout

- [ ] heap 状态保存 `CharT* data`
- [ ] heap 状态保存 `size_t size`
- [ ] heap 状态保存 `size_t capacity_with_category`
- [ ] `capacity()` 能正确剥离 category。
- [ ] `set_capacity()` 能正确写入 category。
- [ ] heap 分配 `capacity + 1` 个 `CharT`。
- [ ] heap 保留 null terminator 空间。
- [ ] heap 释放使用 allocator traits。

### 2.5 Core 基础接口

实现：

- [ ] `basic_string_core() noexcept`
- [ ] `init_small(const CharT* src, size_type n)`
- [ ] `init_heap(const CharT* src, size_type n, size_type capacity)`
- [ ] `destroy_heap() noexcept`
- [ ] `data() noexcept`
- [ ] `data() const noexcept`
- [ ] `size() const noexcept`
- [ ] `capacity() const noexcept`
- [ ] `empty() const noexcept`
- [ ] `is_small() const noexcept`
- [ ] `is_medium() const noexcept`
- [ ] `is_large() const noexcept`
- [ ] `category() const noexcept`
- [ ] `check_invariants() const noexcept`

### 2.6 Unit Tests

Core layout 测试：

- [ ] `core_default_is_empty`
- [ ] `core_default_data_not_null`
- [ ] `core_default_size_is_zero`
- [ ] `core_default_capacity_is_small_capacity`
- [ ] `core_default_null_terminated`
- [ ] `core_default_category_is_small`
- [ ] `core_small_one_char`
- [ ] `core_small_literal`
- [ ] `core_small_max_size`
- [ ] `core_small_embedded_null`
- [ ] `core_small_data_points_to_inline_buffer`
- [ ] `core_heap_sso_plus_one`
- [ ] `core_heap_large_literal`
- [ ] `core_heap_embedded_null`
- [ ] `core_heap_size_is_correct`
- [ ] `core_heap_capacity_at_least_size`
- [ ] `core_heap_null_terminated`
- [ ] `core_heap_category_is_medium`

### 2.7 验收标准

- [ ] `char` 版本 core layout 测试通过。
- [ ] `check_invariants()` 在所有测试中通过。
- [ ] ASan 通过。
- [ ] UBSan 通过。
- [ ] LSan 通过。
- [ ] 没有实现 append / insert / replace。
- [ ] 没有实现 safe over-read。
- [ ] 没有实现 branchless size。

---

## 3. Milestone 2：Core Lifecycle

### 3.1 目标

让 core 成为完整 RAII 类型，支持安全拷贝、移动、赋值和析构。

### 3.2 实现内容

- [ ] `~basic_string_core()`
- [ ] `basic_string_core(const basic_string_core&)`
- [ ] `basic_string_core(basic_string_core&&) noexcept`
- [ ] `basic_string_core& operator=(const basic_string_core&)`
- [ ] `basic_string_core& operator=(basic_string_core&&) noexcept`
- [ ] `swap(basic_string_core&) noexcept`
- [ ] `clear() noexcept`
- [ ] `reset_to_empty() noexcept`

### 3.3 生命周期语义

Small：

- [ ] small copy 复制 inline buffer。
- [ ] small move 等价于复制，然后源对象 reset。
- [ ] small destroy 无操作。

Heap：

- [ ] heap copy 深拷贝。
- [ ] heap move 窃取指针。
- [ ] heap move 后源对象变为空 small。
- [ ] heap destroy 释放分配区域。
- [ ] copy assignment 提供 strong exception guarantee。
- [ ] move assignment 尽量 `noexcept`。

### 3.4 Unit Tests

- [ ] `core_copy_construct_empty`
- [ ] `core_copy_construct_small`
- [ ] `core_copy_construct_heap`
- [ ] `core_copy_after_mutation_independent`
- [ ] `core_move_construct_empty`
- [ ] `core_move_construct_small`
- [ ] `core_move_construct_heap`
- [ ] `core_moved_from_is_empty_valid_string`
- [ ] `core_copy_assign_small_to_small`
- [ ] `core_copy_assign_small_to_heap`
- [ ] `core_copy_assign_heap_to_small`
- [ ] `core_copy_assign_heap_to_heap`
- [ ] `core_copy_assign_self`
- [ ] `core_move_assign_small`
- [ ] `core_move_assign_heap`
- [ ] `core_move_assign_self`
- [ ] `core_swap_small_small`
- [ ] `core_swap_small_heap`
- [ ] `core_swap_heap_heap`
- [ ] `core_destroy_small_noop`
- [ ] `core_destroy_heap_no_leak`

### 3.5 验收标准

- [ ] 无内存泄漏。
- [ ] 无 double free。
- [ ] 无 use-after-free。
- [ ] move 后源对象可析构、可重新赋值。
- [ ] copy 后两个对象互不影响。
- [ ] 所有生命周期操作后 invariant 通过。
- [ ] ASan / UBSan / LSan 全部通过。

---

## 4. Milestone 3：对外 basic_string MVP

### 4.1 目标

封装 `basic_string_core`，形成最小可用 `aethermind::basic_string<char>`。

### 4.2 类型定义

- [ ] 定义 `aethermind::basic_string<CharT, Traits, Allocator>`
- [ ] 定义 `aethermind::string`
- [ ] 暂时不开放其他别名，或开放但不进入主测试。
- [ ] public API 不暴露 core。
- [ ] public API 风格对齐 `std::basic_string`。

### 4.3 类型别名

实现：

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

### 4.4 构造与析构

实现：

- [ ] `basic_string()`
- [ ] `basic_string(const CharT*)`
- [ ] `basic_string(const CharT*, size_type)`
- [ ] `basic_string(std::basic_string_view<CharT, Traits>)`
- [ ] `basic_string(size_type count, CharT ch)`
- [ ] `basic_string(const basic_string&)`
- [ ] `basic_string(basic_string&&) noexcept`
- [ ] `~basic_string()`

暂缓：

- [ ] iterator range constructor
- [ ] initializer_list constructor
- [ ] allocator-aware constructor 全量重载

### 4.5 赋值

实现：

- [ ] `operator=(const basic_string&)`
- [ ] `operator=(basic_string&&) noexcept`
- [ ] `operator=(const CharT*)`
- [ ] `operator=(CharT)`

### 4.6 访问 API

实现：

- [ ] `size() const noexcept`
- [ ] `length() const noexcept`
- [ ] `capacity() const noexcept`
- [ ] `empty() const noexcept`
- [ ] `data() noexcept`
- [ ] `data() const noexcept`
- [ ] `c_str() const noexcept`
- [ ] `begin() noexcept`
- [ ] `begin() const noexcept`
- [ ] `end() noexcept`
- [ ] `end() const noexcept`
- [ ] `cbegin() const noexcept`
- [ ] `cend() const noexcept`
- [ ] `operator[](size_type)`
- [ ] `operator[](size_type) const`
- [ ] `at(size_type)`
- [ ] `at(size_type) const`
- [ ] `front()`
- [ ] `front() const`
- [ ] `back()`
- [ ] `back() const`

### 4.7 Unit Tests

- [ ] `string_default_construct`
- [ ] `string_construct_from_empty_cstr`
- [ ] `string_construct_from_cstr`
- [ ] `string_construct_from_pointer_length`
- [ ] `string_construct_from_string_view`
- [ ] `string_construct_count_char`
- [ ] `string_construct_with_embedded_null`
- [ ] `string_copy_construct_small`
- [ ] `string_copy_construct_heap`
- [ ] `string_move_construct_small`
- [ ] `string_move_construct_heap`
- [ ] `string_copy_assign`
- [ ] `string_move_assign`
- [ ] `string_assign_cstr`
- [ ] `string_assign_char`
- [ ] `string_data_not_null`
- [ ] `string_c_str_null_terminated`
- [ ] `string_size_length_equal`
- [ ] `string_capacity_valid`
- [ ] `string_empty`
- [ ] `string_begin_end`
- [ ] `string_operator_index`
- [ ] `string_at_valid`
- [ ] `string_at_out_of_range`
- [ ] `string_front_back`

### 4.8 差分测试

与 `std::string` 对照：

- [ ] default construct
- [ ] cstr construct
- [ ] pointer-length construct
- [ ] count-char construct
- [ ] copy construct
- [ ] move construct
- [ ] copy assign
- [ ] move assign
- [ ] index access
- [ ] `at()` exception

### 4.9 验收标准

- [ ] `aethermind::string` MVP 可独立使用。
- [ ] 所有 API 后 invariant 通过。
- [ ] 与 `std::string` 基础行为一致。
- [ ] ASan / UBSan / LSan 通过。
- [ ] 没有实现复杂修改 API。

---

## 5. Milestone 4：Capacity API

### 5.1 目标

实现容量管理能力，为 append / insert / replace 做准备。

### 5.2 API

实现：

- [ ] `reserve(size_type new_cap)`
- [ ] `resize(size_type count)`
- [ ] `resize(size_type count, CharT ch)`
- [ ] `clear() noexcept`
- [ ] `shrink_to_fit()`
- [ ] `max_size() const noexcept`

### 5.3 行为规则

- [ ] `reserve(0)` 不破坏对象。
- [ ] `reserve(n <= capacity())` 不改变内容。
- [ ] `reserve(n > capacity())` 保留原内容。
- [ ] `reserve()` 不改变 `size()`。
- [ ] `resize(smaller)` 截断内容。
- [ ] `resize(larger)` 使用 `CharT{}` 填充。
- [ ] `resize(larger, ch)` 使用 `ch` 填充。
- [ ] `clear()` 只设置 size 为 0。
- [ ] `clear()` 后 `data()[0] == CharT{}`
- [ ] `clear()` 不强制释放 heap。
- [ ] `shrink_to_fit()` 保留内容。
- [ ] `shrink_to_fit()` 允许 heap 降级为 small。
- [ ] 所有操作保持 null terminator。

### 5.4 Unit Tests

- [ ] `reserve_zero`
- [ ] `reserve_less_than_capacity_noop`
- [ ] `reserve_equal_capacity_noop`
- [ ] `reserve_greater_than_capacity`
- [ ] `reserve_cross_sso_boundary`
- [ ] `reserve_keep_content`
- [ ] `resize_zero`
- [ ] `resize_smaller_small`
- [ ] `resize_smaller_heap`
- [ ] `resize_larger_small_to_small`
- [ ] `resize_larger_small_to_heap`
- [ ] `resize_larger_heap`
- [ ] `resize_larger_with_char`
- [ ] `clear_small`
- [ ] `clear_heap_keep_capacity`
- [ ] `shrink_small_noop`
- [ ] `shrink_heap_to_exact_heap`
- [ ] `shrink_heap_to_small`
- [ ] `max_size_nonzero`

### 5.5 差分测试

- [ ] reserve sequence
- [ ] resize sequence
- [ ] clear sequence
- [ ] shrink sequence
- [ ] mixed reserve/resize/clear

### 5.6 验收标准

- [ ] 与 `std::string` 行为基本一致。
- [ ] capacity 变化策略允许和 `std::string` 不同，但内容与 size 必须一致。
- [ ] 所有操作后 invariant 通过。
- [ ] ASan / UBSan / LSan 通过。

---

## 6. Milestone 5：Append API

### 6.1 目标

实现最核心的字符串增长路径。

### 6.2 API

实现：

- [ ] `push_back(CharT ch)`
- [ ] `pop_back() noexcept`
- [ ] `append(const basic_string&)`
- [ ] `append(const CharT*)`
- [ ] `append(const CharT*, size_type)`
- [ ] `append(size_type count, CharT ch)`
- [ ] `append(std::basic_string_view<CharT, Traits>)`
- [ ] `operator+=(CharT)`
- [ ] `operator+=(const CharT*)`
- [ ] `operator+=(const basic_string&)`
- [ ] `operator+=(std::basic_string_view<CharT, Traits>)`

### 6.3 实现步骤

- [ ] 实现 `core.append(const CharT* src, size_type n)`
- [ ] 实现 `core.append_fill(size_type count, CharT ch)`
- [ ] 实现 `grow_to(required_capacity)`
- [ ] 实现增长策略。
- [ ] 实现 small -> small append。
- [ ] 实现 small -> heap append。
- [ ] 实现 heap -> heap 无扩容 append。
- [ ] 实现 heap -> heap 扩容 append。
- [ ] 处理 `n == 0`。
- [ ] 处理 self append。
- [ ] 处理 self subrange append。
- [ ] 每次 append 后写 null terminator。

### 6.4 自引用规则

必须识别这些情况：

- [ ] `s.append(s)`
- [ ] `s.append(s.data(), s.size())`
- [ ] `s.append(s.data() + offset, n)`
- [ ] `s += s`

处理策略：

- [ ] 如果 src 在当前 `[data(), data() + size()]` 范围内，先记录 offset。
- [ ] 如果不扩容，可直接 `Traits::move / copy`。
- [ ] 如果扩容，先基于 offset 在新 buffer 中重定位 src。
- [ ] 对复杂 overlap，必要时使用临时 buffer。

### 6.5 Unit Tests

- [ ] `push_back_empty`
- [ ] `push_back_small`
- [ ] `push_back_until_sso_full`
- [ ] `push_back_cross_sso_boundary`
- [ ] `push_back_heap`
- [ ] `pop_back_small`
- [ ] `pop_back_heap`
- [ ] `pop_back_to_empty`
- [ ] `append_empty`
- [ ] `append_cstr`
- [ ] `append_pointer_length`
- [ ] `append_string_view`
- [ ] `append_basic_string`
- [ ] `append_count_char`
- [ ] `append_embedded_null`
- [ ] `append_small_to_small`
- [ ] `append_small_to_heap`
- [ ] `append_heap_to_heap_no_realloc`
- [ ] `append_heap_to_heap_realloc`
- [ ] `append_self_small`
- [ ] `append_self_heap`
- [ ] `append_self_subrange_small`
- [ ] `append_self_subrange_heap`
- [ ] `operator_plus_equal_char`
- [ ] `operator_plus_equal_cstr`
- [ ] `operator_plus_equal_string`
- [ ] `operator_plus_equal_string_view`

### 6.6 差分测试

- [ ] random push_back
- [ ] random pop_back
- [ ] random append literal
- [ ] random append pointer-length
- [ ] random append embedded null
- [ ] random append self

### 6.7 验收标准

- [ ] 自引用 append 无 use-after-free。
- [ ] append 后 null terminator 正确。
- [ ] append 后 size 正确。
- [ ] append 后内容与 `std::string` 一致。
- [ ] ASan / UBSan / LSan 通过。

---

## 7. Milestone 6：Assign / Erase / Insert / Replace

### 7.1 目标

实现字符串主要修改 API。

### 7.2 推荐顺序

- [ ] assign
- [ ] erase
- [ ] insert
- [ ] replace

### 7.3 Assign API

实现：

- [ ] `assign(const basic_string&)`
- [ ] `assign(const CharT*)`
- [ ] `assign(const CharT*, size_type)`
- [ ] `assign(size_type count, CharT ch)`
- [ ] `assign(std::basic_string_view<CharT, Traits>)`

测试：

- [ ] `assign_empty`
- [ ] `assign_small`
- [ ] `assign_heap`
- [ ] `assign_embedded_null`
- [ ] `assign_count_char`
- [ ] `assign_self`
- [ ] `assign_self_subrange`

### 7.4 Erase API

实现：

- [ ] `erase(size_type pos = 0, size_type count = npos)`
- [ ] `erase(const_iterator pos)`
- [ ] `erase(const_iterator first, const_iterator last)`

测试：

- [ ] `erase_empty_range`
- [ ] `erase_prefix`
- [ ] `erase_middle`
- [ ] `erase_suffix`
- [ ] `erase_all`
- [ ] `erase_pos_out_of_range`
- [ ] `erase_count_too_large`
- [ ] `erase_iterator_one`
- [ ] `erase_iterator_range`
- [ ] `erase_heap_to_small_no_auto_shrink`

### 7.5 Insert API

实现第一批：

- [ ] `insert(size_type pos, const CharT*)`
- [ ] `insert(size_type pos, const CharT*, size_type)`
- [ ] `insert(size_type pos, size_type count, CharT ch)`
- [ ] `insert(size_type pos, const basic_string&)`
- [ ] `insert(size_type pos, std::basic_string_view<CharT, Traits>)`

后续补：

- [ ] iterator insert one char
- [ ] iterator insert count
- [ ] iterator insert range

测试：

- [ ] `insert_front`
- [ ] `insert_middle`
- [ ] `insert_back`
- [ ] `insert_empty`
- [ ] `insert_count_char`
- [ ] `insert_cross_sso_boundary`
- [ ] `insert_heap_no_realloc`
- [ ] `insert_heap_realloc`
- [ ] `insert_pos_out_of_range`
- [ ] `insert_embedded_null`
- [ ] `insert_self`
- [ ] `insert_self_subrange`

### 7.6 Replace API

实现第一批：

- [ ] `replace(size_type pos, size_type count, const CharT*)`
- [ ] `replace(size_type pos, size_type count, const CharT*, size_type)`
- [ ] `replace(size_type pos, size_type count, const basic_string&)`
- [ ] `replace(size_type pos, size_type count, std::basic_string_view<CharT, Traits>)`
- [ ] `replace(size_type pos, size_type count, size_type count2, CharT ch)`

测试：

- [ ] `replace_empty_range`
- [ ] `replace_same_length`
- [ ] `replace_shorter`
- [ ] `replace_longer`
- [ ] `replace_prefix`
- [ ] `replace_middle`
- [ ] `replace_suffix`
- [ ] `replace_all`
- [ ] `replace_with_empty`
- [ ] `replace_pos_out_of_range`
- [ ] `replace_embedded_null`
- [ ] `replace_self`
- [ ] `replace_self_subrange`

### 7.7 差分测试

- [ ] random assign
- [ ] random erase
- [ ] random insert
- [ ] random replace
- [ ] random mixed mutation
- [ ] random mutation with embedded null
- [ ] random mutation with self-reference

### 7.8 验收标准

- [ ] 修改后内容与 `std::string` 一致。
- [ ] 越界异常行为与 `std::string` 对齐。
- [ ] 自引用场景无 use-after-free。
- [ ] 所有操作后 invariant 通过。
- [ ] ASan / UBSan / LSan 通过。

---

## 8. Milestone 7：Find / Compare API

### 8.1 目标

实现查询和比较能力。

### 8.2 Compare API

实现：

- [ ] `compare(const basic_string&) const noexcept`
- [ ] `compare(std::basic_string_view<CharT, Traits>) const noexcept`
- [ ] `compare(size_type pos, size_type count, std::basic_string_view<CharT, Traits>) const`

测试：

- [ ] `compare_equal`
- [ ] `compare_less`
- [ ] `compare_greater`
- [ ] `compare_prefix_shorter`
- [ ] `compare_prefix_longer`
- [ ] `compare_empty`
- [ ] `compare_embedded_null`
- [ ] `compare_pos_count`

### 8.3 Find API

实现：

- [ ] `find(CharT ch, size_type pos = 0) const noexcept`
- [ ] `find(const CharT* s, size_type pos = 0) const`
- [ ] `find(const CharT* s, size_type pos, size_type count) const`
- [ ] `find(std::basic_string_view<CharT, Traits> s, size_type pos = 0) const noexcept`
- [ ] `rfind(CharT ch, size_type pos = npos) const noexcept`
- [ ] `rfind(std::basic_string_view<CharT, Traits> s, size_type pos = npos) const noexcept`

后续：

- [ ] `find_first_of`
- [ ] `find_last_of`
- [ ] `find_first_not_of`
- [ ] `find_last_not_of`

C++20/23 风格：

- [ ] `starts_with`
- [ ] `ends_with`
- [ ] `contains`

### 8.4 算法策略

第一版：

- [ ] 单字符查找使用 `Traits::find`。
- [ ] substring 查找使用朴素算法。
- [ ] compare 使用 `Traits::compare`。
- [ ] starts / ends 使用 `Traits::compare`。
- [ ] 不做 SIMD。
- [ ] 不做 Boyer-Moore-Horspool。
- [ ] 不做 memmem 专用优化。

### 8.5 Unit Tests

- [ ] `find_char_exists`
- [ ] `find_char_missing`
- [ ] `find_char_from_pos`
- [ ] `find_char_pos_equal_size`
- [ ] `find_char_pos_greater_size`
- [ ] `find_string_exists`
- [ ] `find_string_missing`
- [ ] `find_string_empty_needle`
- [ ] `find_string_from_pos`
- [ ] `find_embedded_null`
- [ ] `rfind_char_exists`
- [ ] `rfind_char_missing`
- [ ] `rfind_string_exists`
- [ ] `rfind_string_missing`
- [ ] `starts_with_true`
- [ ] `starts_with_false`
- [ ] `ends_with_true`
- [ ] `ends_with_false`
- [ ] `contains_true`
- [ ] `contains_false`

### 8.6 差分测试

- [ ] random compare
- [ ] random find char
- [ ] random find substring
- [ ] random rfind
- [ ] random starts_with / ends_with
- [ ] random embedded null find

### 8.7 验收标准

- [ ] 查询结果与 `std::string` 一致。
- [ ] `npos` 行为一致。
- [ ] 空 needle 行为一致。
- [ ] embedded null 行为一致。
- [ ] ASan / UBSan / LSan 通过。

---

## 9. Milestone 8：多 CharT 支持

### 9.1 目标

逐步开放：

- [ ] `char8_t`
- [ ] `char16_t`
- [ ] `char32_t`
- [ ] `wchar_t`

### 9.2 类型别名

实现：

- [ ] `using string = basic_string<char>;`
- [ ] `using u8string = basic_string<char8_t>;`
- [ ] `using u16string = basic_string<char16_t>;`
- [ ] `using u32string = basic_string<char32_t>;`
- [ ] `using wstring = basic_string<wchar_t>;`

### 9.3 每种 CharT 必测项

对每个 `CharT` 跑：

- [ ] default construct
- [ ] pointer-length construct
- [ ] count-char construct
- [ ] embedded null construct
- [ ] small max size
- [ ] small -> heap
- [ ] heap construct
- [ ] copy
- [ ] move
- [ ] reserve
- [ ] resize
- [ ] append
- [ ] erase
- [ ] insert
- [ ] replace
- [ ] find
- [ ] compare
- [ ] shrink_to_fit
- [ ] null terminator
- [ ] invariant

### 9.4 重点检查

- [ ] capacity 按字符数计算，不按字节数计算。
- [ ] 分配大小为 `(capacity + 1) * sizeof(CharT)`。
- [ ] `Traits::copy` 参数是字符数。
- [ ] `Traits::move` 参数是字符数。
- [ ] `Traits::compare` 参数是字符数。
- [ ] null terminator 是 `CharT{}`。
- [ ] metadata 编码不破坏有效字符。
- [ ] `wchar_t` 在不同平台大小下测试通过。
- [ ] `char8_t` 不和 `char` 指针隐式混用。

### 9.5 差分测试

与标准类型对照：

- [ ] `std::string`
- [ ] `std::u8string`
- [ ] `std::u16string`
- [ ] `std::u32string`
- [ ] `std::wstring`

### 9.6 验收标准

- [ ] 每个 CharT 的基础测试通过。
- [ ] 每个 CharT 的差分测试通过。
- [ ] ASan / UBSan / LSan 通过。
- [ ] generic core 不依赖 endian trick。
- [ ] generic core 不依赖 `sizeof(CharT) == 1`。

---

## 10. Milestone 9：Allocator 支持

### 10.1 目标

逐步补齐 allocator-aware 行为。

### 10.2 第一阶段：std::allocator

- [ ] 默认 allocator 可用。
- [ ] 所有已有功能在 `std::allocator` 下通过。
- [ ] allocator 参数存在但暂不完整测试 stateful 行为。

### 10.3 第二阶段：stateless custom allocator

- [ ] 自定义 allocator 可分配。
- [ ] 自定义 allocator 可释放。
- [ ] 记录 allocation count。
- [ ] 记录 deallocation count。
- [ ] 验证无泄漏。
- [ ] 验证 small string 不分配。

### 10.4 第三阶段：failing allocator

测试：

- [ ] constructor allocation failure
- [ ] copy constructor allocation failure
- [ ] reserve allocation failure
- [ ] resize allocation failure
- [ ] append allocation failure
- [ ] assign allocation failure
- [ ] insert allocation failure
- [ ] replace allocation failure

验收：

- [ ] 构造失败无泄漏。
- [ ] append 失败后原对象不变。
- [ ] insert 失败后原对象不变。
- [ ] replace 失败后原对象不变。
- [ ] assign 失败后原对象不变或处于合法状态。
- [ ] 所有失败路径 invariant 通过。

### 10.5 第四阶段：stateful allocator

实现并测试：

- [ ] `select_on_container_copy_construction`
- [ ] `propagate_on_container_copy_assignment`
- [ ] `propagate_on_container_move_assignment`
- [ ] `propagate_on_container_swap`
- [ ] `is_always_equal`
- [ ] allocator 相等时 move assignment 窃取 buffer。
- [ ] allocator 不等时 move assignment 执行深拷贝或重新分配。
- [ ] swap 时根据 propagation traits 决定是否交换 allocator。
- [ ] copy assignment 根据 propagation traits 处理 allocator。

### 10.6 验收标准

- [ ] allocator 测试全部通过。
- [ ] failing allocator 测试全部通过。
- [ ] stateful allocator 语义明确。
- [ ] 异常安全文档更新。
- [ ] 与 `std::basic_string` 行为尽量对齐。

---

## 11. Milestone 10：标准库集成

### 11.1 比较运算符

实现：

- [ ] `operator==`
- [ ] `operator!=`
- [ ] `operator<`
- [ ] `operator<=`
- [ ] `operator>`
- [ ] `operator>=`
- [ ] `operator<=>`

测试：

- [ ] equal
- [ ] not equal
- [ ] less
- [ ] greater
- [ ] prefix comparison
- [ ] embedded null comparison
- [ ] compare with `const CharT*`
- [ ] compare with `std::basic_string_view`

### 11.2 operator+

实现：

- [ ] `string + string`
- [ ] `string + const CharT*`
- [ ] `const CharT* + string`
- [ ] `string + CharT`
- [ ] `CharT + string`

测试：

- [ ] plus small small
- [ ] plus small heap
- [ ] plus heap heap
- [ ] plus empty
- [ ] plus embedded null

### 11.3 hash

实现：

- [ ] `std::hash<aethermind::basic_string<CharT, Traits, Allocator>>`

测试：

- [ ] equal strings have equal hash
- [ ] embedded null participates in hash
- [ ] different content likely different hash
- [ ] works in `std::unordered_map`

### 11.4 swap

实现：

- [ ] `std::swap`
- [ ] `aethermind::swap`

测试：

- [ ] swap small small
- [ ] swap small heap
- [ ] swap heap heap
- [ ] swap with allocator constraints

### 11.5 可选集成

后续实现：

- [ ] `operator<<`
- [ ] `operator>>`
- [ ] `std::formatter`
- [ ] `from_chars` / `to_chars` 辅助接口，不作为 string 核心职责

### 11.6 验收标准

- [ ] 可作为 `std::unordered_map` key。
- [ ] 可排序。
- [ ] 可与 `std::basic_string_view` 交互。
- [ ] 运算符行为与 `std::string` 一致。

---

## 12. Milestone 11：Differential / Random / Fuzz 测试体系

### 12.1 差分测试框架

建立测试工具：

- [ ] 定义 `StringPair<CharT>`，同时持有 `std::basic_string<CharT>` 和 `aethermind::basic_string<CharT>`。
- [ ] 定义 `CheckEqual()`。
- [ ] 定义 `CheckInvariant()`。
- [ ] 每次操作后自动检查。

每步检查：

- [ ] size 一致。
- [ ] 内容一致。
- [ ] `data()[size()] == CharT{}`
- [ ] empty 一致。
- [ ] find 结果一致。
- [ ] compare 结果一致。
- [ ] 异常行为尽量一致。
- [ ] amstring invariant 通过。

### 12.2 随机操作

实现随机操作：

- [ ] construct
- [ ] assign
- [ ] clear
- [ ] reserve
- [ ] resize
- [ ] push_back
- [ ] pop_back
- [ ] append
- [ ] erase
- [ ] insert
- [ ] replace
- [ ] find
- [ ] rfind
- [ ] compare
- [ ] shrink_to_fit

### 12.3 随机输入

覆盖：

- [ ] 空字符串
- [ ] 单字符
- [ ] small 范围
- [ ] small capacity 边界
- [ ] small capacity + 1
- [ ] 中等字符串
- [ ] 大字符串
- [ ] embedded null
- [ ] 重复字符
- [ ] 随机二进制内容
- [ ] Unicode code unit 样本

### 12.4 Fuzz

- [ ] 使用 libFuzzer。
- [ ] 将输入字节解析为操作序列。
- [ ] 同时驱动 std string 和 amstring。
- [ ] 每步检查一致性。
- [ ] 保存 crash regression case。

### 12.5 验收标准

- [ ] 每次提交可跑短随机测试。
- [ ] 每晚或手动跑长随机测试。
- [ ] fuzz 发现的问题全部转 regression test。
- [ ] mutation API 在随机测试中稳定。

---

## 13. Milestone 12：Benchmark 与性能基线

### 13.1 目标

建立性能基线，对比：

- [ ] `std::string`
- [ ] `folly::fbstring`
- [ ] `aethermind::string`

### 13.2 Benchmark 分类

构造：

- [ ] default construct
- [ ] construct small literal
- [ ] construct small pointer-length
- [ ] construct heap
- [ ] construct embedded null

拷贝移动：

- [ ] copy small
- [ ] copy heap
- [ ] move small
- [ ] move heap
- [ ] vector push_back small
- [ ] vector push_back heap

容量：

- [ ] reserve
- [ ] resize grow
- [ ] resize shrink
- [ ] shrink_to_fit

追加：

- [ ] push_back loop
- [ ] append small
- [ ] append heap
- [ ] append self
- [ ] operator+=

修改：

- [ ] insert front
- [ ] insert middle
- [ ] insert back
- [ ] erase front
- [ ] erase middle
- [ ] replace same length
- [ ] replace shorter
- [ ] replace longer

查询：

- [ ] find char exists
- [ ] find char missing
- [ ] find substring exists
- [ ] find substring missing
- [ ] compare equal
- [ ] compare different
- [ ] starts_with
- [ ] ends_with

真实 workload：

- [ ] 日志拼接
- [ ] HTTP header 解析
- [ ] JSON key/value 处理
- [ ] 路径拼接
- [ ] unordered_map string key
- [ ] 编译器 token 文本处理

### 13.3 指标

记录：

- [ ] ns/op
- [ ] allocations/op
- [ ] bytes allocated/op
- [ ] cycles/op
- [ ] instructions/op
- [ ] branch misses
- [ ] cache misses
- [ ] peak RSS，可选

### 13.4 工具

- [ ] Google Benchmark
- [ ] perf
- [ ] heaptrack
- [ ] cachegrind
- [ ] massif，可选

### 13.5 验收标准

- [ ] 有稳定 benchmark 脚本。
- [ ] 有 std::string / fbstring / amstring 对照数据。
- [ ] benchmark 数据保存到 `benchmarks/results/`。
- [ ] 每次优化前后有可比较数据。
- [ ] 不基于感觉优化。

---

## 14. Milestone 13：char 专用优化 Core

### 14.1 目标

在 generic core 稳定后，为 `char` 做 fbstring-like 专用优化，使 `aethermind::string` 性能对标 fbstring。

### 14.2 前置条件

必须满足：

- [ ] generic core 全部测试通过。
- [ ] `aethermind::string` 差分测试通过。
- [ ] mutation random test 稳定。
- [ ] benchmark 基线已建立。
- [ ] 已明确性能瓶颈。

### 14.3 优化方向

- [ ] `basic_string_core<char>` 专用特化。
- [ ] 24-byte object layout。
- [ ] 23-char SSO capacity。
- [ ] 优化 small copy。
- [ ] 优化 size()。
- [ ] 优化 category encoding。
- [ ] 优化 append fast path。
- [ ] 优化 compare 使用 `memcmp`。
- [ ] 优化 find char 使用 `memchr`。
- [ ] 可选启用 safe over-read，但必须宏控。
- [ ] 可选 branchless size，但必须 benchmark 证明收益。

### 14.4 禁止事项

- [ ] 不得破坏 public API。
- [ ] 不得破坏 generic CharT。
- [ ] 不得让 sanitizer build 使用 safe over-read。
- [ ] 不得让优化绕过 invariant。
- [ ] 不得在无 benchmark 依据时引入复杂技巧。

### 14.5 Tests

对 char optimized core 重新跑：

- [ ] 所有 core tests
- [ ] 所有 string tests
- [ ] 所有 mutation tests
- [ ] 所有 differential tests
- [ ] all sanitizer tests
- [ ] random tests
- [ ] fuzz regression tests

### 14.6 Benchmark 验收

与 fbstring 对比：

- [ ] small construct 接近 fbstring。
- [ ] small copy 接近 fbstring。
- [ ] small move 接近 fbstring。
- [ ] append 场景接近 fbstring。
- [ ] find/compare 不明显落后。
- [ ] 真实 workload 有收益或不明显退化。

---

## 15. Release Checklist

### 15.1 正确性

- [ ] 全部 unit test 通过。
- [ ] 全部 differential test 通过。
- [ ] 全部 allocator test 通过。
- [ ] 全部 regression test 通过。
- [ ] fuzz 无已知 crash。
- [ ] ASan 通过。
- [ ] UBSan 通过。
- [ ] LSan 通过。
- [ ] Debug invariant 无失败。

### 15.2 API

- [ ] `aethermind::string` 可用。
- [ ] `aethermind::u8string` 可用。
- [ ] `aethermind::u16string` 可用。
- [ ] `aethermind::u32string` 可用。
- [ ] `aethermind::wstring` 可用。
- [ ] API compatibility matrix 已更新。
- [ ] 未实现 API 在文档中明确标注。

### 15.3 性能

- [ ] benchmark 已运行。
- [ ] benchmark 数据已保存。
- [ ] 对比 std::string。
- [ ] 对比 fbstring。
- [ ] 主要性能退化点已记录。
- [ ] 优化项有数据支持。

### 15.4 文档

- [ ] storage layout 文档更新。
- [ ] exception safety 文档更新。
- [ ] iterator invalidation 文档更新。
- [ ] API compatibility 文档更新。
- [ ] benchmark policy 文档更新。
- [ ] README 更新。
- [ ] 使用示例更新。

### 15.5 工程质量

- [ ] 编译无 warning。
- [ ] clang-tidy 主要检查通过。
- [ ] clang-format 已应用。
- [ ] public header 可独立 include。
- [ ] 不依赖未声明宏。
- [ ] 不泄露 detail API。
- [ ] CI 或本地脚本可一键验证。

---

## 16. 每日开发工作流 Checklist

每次开发一个小功能时：

- [ ] 明确本次只做一个功能点。
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
- [ ] 提交前确认没有引入未使用复杂优化。

每次完成一个 milestone 时：

- [ ] 跑全部 unit test。
- [ ] 跑全部 differential test。
- [ ] 跑 sanitizer 全量。
- [ ] 更新文档。
- [ ] 记录风险点。
- [ ] 必要时跑 benchmark。
- [ ] 归档 regression case。

---

## 17. 优先级总览

### P0：必须先完成

- [ ] 工程骨架。
- [ ] generic core small / heap layout。
- [ ] core invariant。
- [ ] core lifecycle。
- [ ] `aethermind::string` MVP。
- [ ] ASan / UBSan / LSan。
- [ ] 与 `std::string` 差分测试。

### P1：核心功能

- [ ] reserve / resize / clear / shrink_to_fit。
- [ ] append / push_back / pop_back。
- [ ] assign / erase / insert / replace。
- [ ] find / compare。
- [ ] 多 CharT 基础支持。

### P2：标准兼容

- [ ] allocator propagation。
- [ ] iterator overload。
- [ ] operator+。
- [ ] comparison operators。
- [ ] hash。
- [ ] swap。
- [ ] stream / formatter。

### P3：性能优化

- [ ] benchmark 基线。
- [ ] char 专用 optimized core。
- [ ] memchr / memcmp 优化。
- [ ] append fast path。
- [ ] branchless size。
- [ ] safe over-read opt-in。
- [ ] SIMD，可选。

---

## 18. 建议的第一周执行计划

### Day 1（M0：骨架与规格冻结）

- [ ] 创建 `include/amstring/` 目录及骨架头文件。
- [ ] 创建 `tests/unit/amstring/` 子目录及空测试文件。
- [ ] 创建 `tests/benchmark/amstring/` 子目录及空 benchmark 文件。
- [ ] 验证 GLOB_RECURSE 自动收集（编译 `aethermind_unit_tests`, `aethermind_benchmark`）。
- [ ] 编写 `docs/designs/amstring/storage_layout.md`。
- [ ] 编写 `docs/designs/amstring/api_compatibility.md`。
- [ ] 编写 `docs/designs/amstring/tdd_plan.md`。
- [ ] 在文档中明确第一版设计决策。

### Day 2（M1：Core Layout - 类型定义）

- [ ] 实现 `Category` enum（Small/Medium/Large）。
- [ ] 实现 `MediumLarge<CharT>` struct。
- [ ] 实现 `basic_string_core<CharT, Traits, Allocator>` 空对象。
- [ ] 实现 default constructor。
- [ ] 实现 `data / size / capacity / empty`。
- [ ] 写 empty core 测试（`tests/unit/amstring/test_core_layout.cpp`）。

### Day 3（M1：Small Layout）

- [ ] 实现 `init_small(const CharT* src, size_type n)`。
- [ ] 实现 small metadata 编码（用 CharT 存储，不用 byte trick）。
- [ ] 实现 `kSmallCapacity`, `kSmallArraySize` 常量。
- [ ] 实现 small invariant。
- [ ] 写 small 测试。
- [ ] 跑 ASan / UBSan。

### Day 4（M1：Heap Layout）

- [ ] 实现 heap allocation（使用 allocator_traits）。
- [ ] 实现 `init_heap(const CharT* src, size_type n, size_type cap)`。
- [ ] 实现 `destroy_heap()`。
- [ ] 实现 heap capacity encoding（category bits）。
- [ ] 实现 heap invariant。
- [ ] 写 heap 测试。
- [ ] 跑 LSan。

### Day 5（M2：Lifecycle - 析构与构造）

- [ ] 实现 `~basic_string_core()`。
- [ ] 实现 copy constructor（small/heap）。
- [ ] 实现 move constructor（small/heap）。
- [ ] 写 lifecycle 测试（`tests/unit/amstring/test_core_lifecycle.cpp`）。

### Day 6（M2：Lifecycle - 赋值与交换）

- [ ] 实现 copy assignment（strong exception guarantee）。
- [ ] 实现 move assignment（noexcept）。
- [ ] 实现 `swap()`。
- [ ] 写 assignment / swap 测试。
- [ ] 验收：无泄漏、无 double free、无 use-after-free。

### Day 7（M3：basic_string MVP）

- [ ] 封装 `aethermind::basic_string<char>` 公共 API。
- [ ] 实现基础 constructor / accessor。
- [ ] 写与 `std::string` 的基础差分测试。
- [ ] 整理第一周问题清单。

---

## 19. 最小可交付版本定义

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

- [ ] 完整 allocator propagation。
- [ ] 完整 iterator overload。
- [ ] 完整 replace overload。
- [ ] COW。
- [ ] SIMD。
- [ ] safe over-read。
- [ ] stream / formatter。
- [ ] 所有 CharT 全部达到高性能。

---

## 20. 最终目标版本定义

最终稳定版应满足：

- [ ] 功能覆盖大部分 `std::basic_string` 常用 API。
- [ ] 支持 `char / char8_t / char16_t / char32_t / wchar_t`。
- [ ] `char` 版本有专用优化 core。
- [ ] 小字符串性能接近或优于 `std::string`。
- [ ] 常见场景性能接近 `fbstring`。
- [ ] 所有修改 API 通过随机差分测试。
- [ ] allocator 行为明确且测试覆盖。
- [ ] 文档完整。
- [ ] benchmark 可复现。
