# amstring 重新设计开发计划草案

> 目标：放弃当前实现，从零开始重新设计 `amstring`；保留类似 fbstring 的内部存储思想；支持多字符类型；功能逐步对齐 `std::basic_string`；性能对标 `folly::fbstring`；开发模式采用 TDD。

---

## 1. 总体目标

`amstring` 的目标不是简单实现一个 `std::string` 替代品，而是实现一个：

```text
功能上逐步对齐 std::basic_string
内部布局参考 fbstring
小字符串性能强
中大字符串分配策略高效
支持多 CharT 类型
可通过差分测试验证标准兼容性
可通过 benchmark 对标 std::string / fbstring
```

推荐最终接口形式：

```cpp
namespace aethermind {

template <
    class CharT,
    class Traits = std::char_traits<CharT>,
    class Allocator = std::allocator<CharT>
>
class basic_string;

using string    = basic_string<char>;
using u8string  = basic_string<char8_t>;
using u16string = basic_string<char16_t>;
using u32string = basic_string<char32_t>;
using wstring   = basic_string<wchar_t>;

} // namespace aethermind
```

其中：

```text
aethermind::string    对标 std::string / folly::fbstring
aethermind::u8string  对标 std::u8string
aethermind::u16string 对标 std::u16string
aethermind::u32string 对标 std::u32string
aethermind::wstring   对标 std::wstring
```

---

## 2. 核心设计原则

### 2.1 先正确，再极限优化

第一版不追求完全复刻 fbstring 的所有技巧。尤其是以下优化要延后：

```text
safe over-read
branchless size()
large string COW
SIMD find / compare
复杂 allocator propagation 优化
```

第一阶段优先保证：

```text
布局正确
生命周期正确
null terminator 正确
异常安全正确
多 CharT 正确
std::string 差分测试通过
```

### 2.2 保留 fbstring-like 三态存储模型

内部存储采用三态：

```text
Small  : 小字符串内联存储，不分配堆内存
Medium : 中等字符串堆分配，独占所有权
Large  : 大字符串特殊策略，第一版可预留，后续优化
```

推荐第一版实际启用：

```text
Small  : 启用
Medium : 启用
Large  : 预留 category，不启用特殊逻辑
```

也就是说，第一版可以把所有非 small 字符串都作为 medium 独占堆字符串处理。等基础功能、测试和 benchmark 稳定之后，再决定 large 是否引入特殊策略。

### 2.3 第一版不建议做 COW

fbstring 的 large string 设计历史上有引用计数共享思路，但 amstring 第一版不建议使用 COW。

原因：

```text
C++11 之后 std::string 不推荐 COW 语义
非 const data() 会让 detach 逻辑复杂
引用计数带来原子开销
多线程语义容易误导
TDD 和异常安全成本明显提高
```

第一版建议：

```text
Small  : inline ownership
Medium : heap exclusive ownership
Large  : heap exclusive ownership，暂不区分
```

后续可以做实验特性：

```cpp
#define AMSTRING_ENABLE_LARGE_REFCOUNT 0
```

但不进入主线版本。

---

## 3. 内部架构设计

### 3.1 模块划分

建议目录结构（融入 AetherMind 项目）：

```text
AetherMind/
├── include/
│   ├── container/              ← 现有目录（保留 string.h, array.h 等）
│   └── amstring/               ← 新建：amstring 子系统头文件（扁平结构）
│       ├── basic_string.hpp    ← 主模板（公共 API）
│       ├── string.hpp          ← 类型别名（公共 API）
│       ├── string_fwd.hpp      ← 前置声明（公共 API）
│       ├── config.hpp          ← 配置宏（内部）
│       ├── core.hpp            ← basic_string_core（内部）
│       ├── layout.hpp          ← MediumLarge, Storage（内部）
│       ├── category.hpp        ← Category enum（内部）
│       ├── growth_policy.hpp   ← 容量增长策略（内部）
│       ├── char_algorithms.hpp ← 字符操作辅助（内部）
│       └── invariant.hpp       ← invariant 检查（内部）
│
├── tests/
│   ├── unit/
│   │   ├── amstring/           ← 新建：amstring 单元测试子目录
│   │   │   ├── test_core_layout.cpp
│   │   │   ├── test_core_lifecycle.cpp
│   │   │   ├── test_api_basic.cpp
│   │   │   ├── test_api_capacity.cpp
│   │   │   ├── test_api_append.cpp
│   │   │   ├── test_api_mutation.cpp
│   │   │   ├── test_api_find_compare.cpp
│   │   │   ├── test_multi_char.cpp
│   │   │   ├── test_allocator.cpp
│   │   │   ├── test_differential.cpp
│   │   │   └── test_regression.cpp
│   │   └── ...现有测试文件...
│   │
│   └── benchmark/
│       ├── amstring/           ← 新建：amstring benchmark 子目录
│       │   ├── benchmark_construct.cpp
│       │   ├── benchmark_copy_move.cpp
│       │   ├── benchmark_append.cpp
│       │   ├── benchmark_insert_erase.cpp
│       │   ├── benchmark_find_compare.cpp
│       │   ├── benchmark_real_world.cpp
│       │   └── benchmark_vs_std_string.cpp
│       └ ...现有 benchmark 文件...
│
├── docs/
│   └── designs/amstring/       ← 设计文档目录
│       ├── amstring_development_plan.md
│       ├── amstring_development_steps_checklist.md
│       ├── storage_layout.md
│       ├── api_compatibility.md
│       ├── exception_safety.md
│       ├── iterator_invalidation.md
│       ├── benchmark_policy.md
│       └── tdd_plan.md
```

说明：
- 头文件 `include/amstring/` 采用扁平结构，公共/内部通过命名约定区分
- 公共 API：`basic_string.hpp`, `string.hpp`, `string_fwd.hpp`
- 内部实现：`core.hpp`, `config.hpp`, `category.hpp` 等（用户不应直接 include）
- 单元测试 `tests/unit/amstring/` 子目录，融入现有测试框架（GLOB_RECURSE 自动收集）
- Benchmark `tests/benchmark/amstring/` 子目录，融入现有 benchmark 框架
- 设计文档 `docs/designs/amstring/` 与其他设计文档统一管理

### 3.2 类型分层

建议分成两层：

```text
basic_string        : 对外 API 层
basic_string_core   : 内部存储层
```

示意：

```cpp
namespace aethermind {

template <class CharT, class Traits, class Allocator>
class basic_string_core;  // 内部存储层

template <
    class CharT,
    class Traits = std::char_traits<CharT>,
    class Allocator = std::allocator<CharT>
>
class basic_string {
private:
    basic_string_core<CharT, Traits, Allocator> core_;
};

} // namespace aethermind
```

这样做的好处：

```text
对外接口保持 std::basic_string 风格
内部可以独立测试 storage layout
后续可替换 storage policy
核心布局问题不会污染 public API
采用扁平目录结构，避免向上引用问题
```

---

## 4. fbstring-like 存储布局重新设计

### 4.1 对象大小目标

推荐目标：

```text
64-bit 平台：
sizeof(aethermind::basic_string<char>) <= 24 或 32 bytes
```

如果追求 fbstring 风格，优先目标是：

```text
sizeof(string) == sizeof(pointer) + 2 * sizeof(size_t)
```

在 64-bit 下即：

```text
24 bytes
```

但需要注意：一旦支持 allocator，尤其是 stateful allocator，24 字节未必总能保持。

因此建议分两种模式：

```text
默认高性能模式：
  allocator 使用 EBO / compressed pair 优化；
  std::allocator 情况下保持 24 字节。

完整 allocator 模式：
  stateful allocator 可能增加对象大小；
  以标准兼容性优先。
```

### 4.2 MediumLarge 布局

推荐内部布局：

```cpp
template <class CharT>
struct MediumLarge {
    CharT* data;
    size_t size;
    size_t capacity_with_category;
};
```

其中 `capacity_with_category` 的低位或高位保存 category。

不过要支持不同 `CharT`，需要非常谨慎。不要假设 `CharT` 一定是 1 字节。

### 4.3 Small 布局

推荐：

```cpp
template <class CharT>
struct Small {
    CharT buffer[kSmallCapacity + 1];
};
```

其中：

```text
kSmallCapacity = 可存储的最大字符数，不含 null terminator
```

如果继续追求 fbstring 的 24 字节布局，可以用 union：

```cpp
union Storage {
    CharT small_[kSmallArraySize];
    MediumLarge<CharT> ml_;
    std::byte bytes_[sizeof(MediumLarge<CharT>)];
};
```

关键是：

```text
Small 状态下，最后一个 byte 用来保存 small size/category 信息
Medium/Large 状态下，capacity_with_category 保存容量和 category
```

但这对 `char16_t / char32_t / wchar_t` 有复杂影响。

---

## 5. 多 CharT 支持策略

### 5.1 支持范围

第一版最终目标支持：

```text
char
char8_t
char16_t
char32_t
wchar_t
```

但实现顺序不要一次性展开。

推荐顺序：

```text
阶段 A：char
阶段 B：char8_t
阶段 C：char16_t
阶段 D：char32_t
阶段 E：wchar_t
```

原因：

```text
char / char8_t 都是 1 byte code unit，布局最简单
char16_t 是 2 byte code unit
char32_t 是 4 byte code unit
wchar_t 平台相关，Windows 通常 2 byte，Linux 通常 4 byte
```

### 5.2 Small capacity 计算

推荐原则：

```cpp
static constexpr size_t kStorageBytes = sizeof(MediumLarge<CharT>);
static constexpr size_t kSmallArraySize = kStorageBytes / sizeof(CharT);
static constexpr size_t kSmallCapacity = kSmallArraySize - 1;
```

也就是 small buffer 必须至少保留一个 `CharT{}` 作为 null terminator。

例如 24 字节布局下：

| CharT | sizeof(CharT) | small array size | small capacity |
|---|---:|---:|---:|
| `char` | 1 | 24 | 23 |
| `char8_t` | 1 | 24 | 23 |
| `char16_t` | 2 | 12 | 11 |
| `char32_t` | 4 | 6 | 5 |
| `wchar_t` | 2 或 4 | 12 或 6 | 11 或 5 |

这个策略比“把最后一个 byte 强行当 marker”更直观，也更容易测试。

### 5.3 Category 标记建议

为了支持不同 `CharT`，建议不要一开始就使用过度复杂的 endian trick。

第一版推荐更稳定的方案：

```text
Small 状态：
  使用最后一个 CharT 存储 encoded small metadata。
  该位置同时不作为有效字符内容使用。

Medium/Large 状态：
  使用 capacity_with_category 保存 category。
```

也就是说，small buffer 的最后一个 `CharT` 不存储用户字符，而是存储 metadata。

优点：

```text
支持不同 CharT 更容易
不依赖最后一个 byte 的 endian 位置
size() 实现简单
测试容易
```

缺点：

```text
相比 fbstring 可能少 1 个 CharT 的 SSO 容量
```

但对于第一版，这是值得的。

### 5.4 推荐 small 编码

```cpp
enum class Category : unsigned char {
    Small  = 0,
    Medium = 1,
    Large  = 2,
};

template <class CharT>
struct SmallMetadata {
    // 存储在 small_[kSmallCapacity] 位置
    // encoded = kSmallCapacity - size
};
```

Small size 计算：

```cpp
size_t small_size() const noexcept {
    return kSmallCapacity - static_cast<size_t>(small_[kSmallCapacity]);
}
```

对于 `char16_t / char32_t`，这个也成立，只是要保证 metadata 值不超过 `CharT` 可表示范围。由于 small capacity 很小，完全没问题。

### 5.5 更激进的 fbstring byte-level marker

后续优化版可以再改成：

```text
最后一个 byte 存 small metadata/category
通过 endian-aware 逻辑支持不同 CharT
```

但不建议第一版这么做。

第一版目标应该是：

```text
多 CharT 正确性 > SSO 极限容量
```

后续目标才是：

```text
char 专用 fast layout 逼近 fbstring
```

---

## 6. Core 层设计

### 6.1 basic_string_core 职责

`basic_string_core` 只负责：

```text
存储
分配
释放
拷贝
移动
扩容
维护 null terminator
维护 small / medium / large 状态
```

不负责复杂字符串算法，例如：

```text
find
replace
format
stream
hash
```

### 6.2 Core 必须维护的不变量

所有状态都必须满足：

```text
1. data() != nullptr
2. data()[size()] == CharT{}
3. size() <= capacity()
4. begin() == data()
5. end() == data() + size()
6. moved-from object 是合法 empty string
7. small / medium / large 状态可 O(1) 判断
8. heap 状态下 data 指向有效分配区域
9. heap 状态下 capacity 不包含 category bits
```

Debug 模式必须提供：

```cpp
void check_invariants() const noexcept;
```

每个 mutating API 测试后调用。

### 6.3 Core API 草案

```cpp
template <class CharT, class Traits, class Allocator>
class basic_string_core {
public:
    using size_type = size_t;
    using allocator_type = Allocator;

    basic_string_core() noexcept;
    basic_string_core(const CharT* data, size_type size, const Allocator& alloc);
    basic_string_core(const basic_string_core&);
    basic_string_core(basic_string_core&&) noexcept;
    ~basic_string_core();

    basic_string_core& operator=(const basic_string_core&);
    basic_string_core& operator=(basic_string_core&&) noexcept;

    const CharT* data() const noexcept;
    CharT* data() noexcept;

    size_type size() const noexcept;
    size_type capacity() const noexcept;
    bool empty() const noexcept;

    void clear() noexcept;
    void reserve(size_type new_cap);
    void resize(size_type new_size, CharT fill);
    void shrink_to_fit();

    void append(const CharT* src, size_type n);
    void assign(const CharT* src, size_type n);

    void swap(basic_string_core&) noexcept;

private:
    bool is_small() const noexcept;
    bool is_medium() const noexcept;
    bool is_large() const noexcept;

    void init_small(const CharT* src, size_type n);
    void init_heap(const CharT* src, size_type n, size_type cap);
    void destroy_heap() noexcept;
    void grow_to(size_type min_capacity);
};
```

---

## 7. basic_string API 逐步对齐 std::basic_string

### 7.1 第一阶段 API：MVP

先实现最小可用版本：

```cpp
basic_string();
basic_string(const CharT*);
basic_string(const CharT*, size_type);
basic_string(std::basic_string_view<CharT, Traits>);
basic_string(size_type count, CharT ch);

basic_string(const basic_string&);
basic_string(basic_string&&) noexcept;
~basic_string();

basic_string& operator=(const basic_string&);
basic_string& operator=(basic_string&&) noexcept;
basic_string& operator=(const CharT*);
basic_string& operator=(CharT);
```

访问：

```cpp
size();
length();
capacity();
empty();

data();
c_str();

begin();
end();
cbegin();
cend();

operator[];
at();
front();
back();
```

### 7.2 第二阶段 API：容量管理

```cpp
reserve();
resize();
clear();
shrink_to_fit();
max_size();
```

策略：

```text
reserve 只增长
clear 不释放 capacity
shrink_to_fit 可以 heap -> small
resize 缩小时不自动降级
resize 增长时填充 CharT{}
```

### 7.3 第三阶段 API：append / push_back

```cpp
push_back();
pop_back();

append(const basic_string&);
append(const CharT*);
append(const CharT*, size_type);
append(size_type count, CharT ch);
append(std::basic_string_view<CharT, Traits>);

operator+=();
```

必须优先支持自引用：

```cpp
s.append(s);
s.append(s.data(), s.size());
s.append(s.data() + 1, 3);
```

### 7.4 第四阶段 API：assign / erase / insert / replace

建议实现顺序：

```text
assign
erase
insert
replace
```

原因：

```text
assign 最接近重新初始化
erase 只涉及内部移动
insert 涉及扩容和内部搬移
replace 同时包含 erase + insert 的复杂情况
```

第一批接口：

```cpp
assign();
erase(size_type pos = 0, size_type count = npos);
insert(size_type pos, const CharT*);
insert(size_type pos, const CharT*, size_type);
insert(size_type pos, size_type count, CharT ch);
replace(size_type pos, size_type count, const CharT*, size_type);
```

后续补 iterator 版本。

### 7.5 第五阶段 API：find / compare

```cpp
compare();

find();
rfind();
find_first_of();
find_last_of();
find_first_not_of();
find_last_not_of();

starts_with();
ends_with();
contains();
```

第一版算法：

```text
单字符 find：Traits::find 或 memchr 类逻辑
字符串 find：朴素算法
compare：Traits::compare
starts_with / ends_with：Traits::compare
```

后续优化：

```text
char 专用 memchr / memmem
Boyer-Moore-Horspool
SIMD
```

### 7.6 第六阶段 API：标准库集成

```cpp
operator==;
operator!=;
operator<;
operator<=;
operator>;
operator>=;
operator<=>;

operator+;
std::hash;
std::swap;
ostream / istream;
formatter;
```

---

## 8. Allocator 策略

### 8.1 不要忽视 allocator

如果目标是“功能逐步对齐 `std::string`”，allocator 最终必须处理。

但是第一阶段可以分层推进：

```text
阶段 1：仅 std::allocator，接口预留 Allocator 模板参数
阶段 2：支持 stateless allocator
阶段 3：支持 stateful allocator
阶段 4：处理 propagation traits
```

### 8.2 allocator 复杂点

必须逐步处理：

```text
propagate_on_container_copy_assignment
propagate_on_container_move_assignment
propagate_on_container_swap
is_always_equal
select_on_container_copy_construction
```

这些不建议在 MVP 阶段全部实现。

MVP 可以先规定：

```text
Allocator 模板参数存在
默认 std::allocator
暂时只完整测试 std::allocator
stateful allocator 进入后续阶段
```

---

## 9. TDD 开发策略

### 9.1 总原则

每个功能开发顺序：

```text
1. 写 std::basic_string 对照测试
2. 写边界测试
3. 写 invariant 测试
4. 写异常测试
5. 实现最小代码
6. 运行 sanitizer
7. 运行差分测试
8. 运行 benchmark
9. 再优化
```

不要先优化再补测试。

### 9.2 测试分层

#### Unit Test

验证单个 API 行为。

例如：

```text
construct_from_literal
copy_construct_small
move_construct_heap
reserve_cross_sso_boundary
append_self
erase_middle
find_embedded_null
```

#### Core Invariant Test

专门测试内部存储。

```text
small_empty_invariant
small_max_capacity_invariant
heap_invariant
move_from_heap_invariant
shrink_heap_to_small_invariant
```

#### Differential Test

和 `std::basic_string<CharT>` 对照。

每一步检查：

```text
size 一致
内容一致
null terminator 正确
find 结果一致
compare 结果一致
异常行为尽量一致
```

#### Random Operation Test

随机生成操作序列，同时作用于：

```text
std::basic_string<CharT>
aethermind::basic_string<CharT>
```

操作包括：

```text
append
assign
resize
reserve
clear
erase
insert
replace
push_back
pop_back
find
compare
```

#### Fuzz Test

用 libFuzzer 把输入字节解析为操作序列。

目标：

```text
发现越界
发现 use-after-free
发现自引用 bug
发现异常安全 bug
发现 embedded null bug
```

#### Sanitizer

必须长期启用：

```text
ASan
UBSan
LSan
TSan：只用于并发读写误用测试，不表示 string 自身线程安全
```

---

## 10. 多 CharT TDD 矩阵

每一类 CharT 都要有基础测试：

```text
char
char8_t
char16_t
char32_t
wchar_t
```

每类至少覆盖：

```text
default construct
construct from pointer + length
construct with embedded null
small max size
small -> heap
copy
move
append
resize
find
compare
```

重点是 `char16_t / char32_t / wchar_t`：

```text
SSO capacity 是否正确
null terminator 是否正确
Traits::copy/move/compare 是否正确
字节数计算是否没有写错
capacity 是否按 CharT 个数而不是字节数计算
```

---

## 11. 开发里程碑

### Milestone 0：规格冻结

产出文档：

```text
storage_layout.md
api_compatibility.md
tdd_plan.md
exception_safety.md
benchmark_policy.md
```

必须冻结的问题：

```text
对象大小目标是多少
Small capacity 如何计算
Category 如何编码
Large 是否启用特殊策略
Allocator 第一版支持到什么程度
是否默认支持 char8_t
是否允许 safe over-read
```

建议冻结结论：

```text
对象布局：fbstring-like 24 字节优先
第一版 Large 不启用特殊策略
第一版不做 COW
第一版不做 safe over-read
第一版不做 SIMD
第一版支持 char，随后支持 char8_t / char16_t / char32_t / wchar_t
```

### Milestone 1：Core Layout Prototype

目标：

```text
basic_string_core<CharT> 的 small / heap 布局可用
```

完成：

```text
default constructor
init_small
init_heap
destroy
data
size
capacity
category
check_invariants
```

测试：

```text
core_empty
core_small
core_small_max
core_heap
core_null_terminated
core_capacity
core_category
```

验收：

```text
char 版本通过
ASan / UBSan 通过
sizeof 目标达成或解释偏差
```

### Milestone 2：Core Lifecycle

目标：

```text
core 成为完整 RAII 类型
```

完成：

```text
destructor
copy constructor
move constructor
copy assignment
move assignment
swap
clear
```

测试：

```text
copy_small
copy_heap
move_small
move_heap
self_assign
swap_small_small
swap_small_heap
swap_heap_heap
moved_from_valid
```

验收：

```text
无泄漏
无 double free
move noexcept
copy 深拷贝
```

### Milestone 3：basic_string MVP

目标：

```text
aethermind::basic_string<char> 可作为简单字符串使用
```

完成：

```text
constructors
destructor
copy/move
data/c_str
size/length/capacity/empty
begin/end
operator[]
at
front/back
```

测试：

```text
与 std::string 基础行为一致
包含 embedded null
small / heap 都覆盖
```

### Milestone 4：Capacity API

完成：

```text
reserve
resize
clear
shrink_to_fit
```

测试：

```text
reserve 不改变 size
resize 扩大填充正确
resize 缩小内容正确
clear 后 data 合法
shrink_to_fit 内容不变
heap -> small 降级正确
```

### Milestone 5：Append API

完成：

```text
push_back
pop_back
append
operator+=
```

重点测试：

```text
small -> small
small -> heap
heap -> heap
append empty
append embedded null
append self
append self subrange
```

验收：

```text
差分测试通过
自引用 append 无 use-after-free
```

### Milestone 6：Mutation API

完成：

```text
assign
erase
insert
replace
```

测试：

```text
头部/中间/尾部
变长/变短/等长
越界异常
自引用
embedded null
```

验收：

```text
随机操作差分测试通过
ASan / UBSan / LSan 通过
```

### Milestone 7：Find / Compare API

完成：

```text
compare
find
rfind
find_first_of
find_last_of
find_first_not_of
find_last_not_of
starts_with
ends_with
contains
```

测试：

```text
空 needle
找不到
从指定 pos 开始
pos > size
embedded null
重复 pattern
不同 CharT
```

### Milestone 8：多 CharT 支持

按顺序放开：

```text
char8_t
char16_t
char32_t
wchar_t
```

每放开一种，都必须跑完整核心测试和差分测试。

特别检查：

```text
small capacity
metadata 编码
null terminator
Traits::copy/move/compare
sizeof 和 alignment
```

### Milestone 9：Allocator 完整化

完成：

```text
stateful allocator
allocator propagation
allocator-aware copy/move/swap
allocation failure tests
```

测试：

```text
custom allocator counts allocation
failing allocator
stateful allocator copy
stateful allocator move
allocator unequal move assignment
allocator swap propagation
```

### Milestone 10：标准库集成

完成：

```text
operator+
comparison operators
operator<=>
std::hash
std::swap
stream IO
formatter，可选
```

### Milestone 11：Benchmark 与性能优化

对标：

```text
std::string
folly::fbstring
aethermind::basic_string<char>
```

核心 benchmark：

```text
default construct
construct small
construct heap
copy small
copy heap
move small
move heap
append small
append heap
push_back loop
find char
find substring
compare
vector push_back
unordered_map key
日志拼接
路径拼接
HTTP header 解析
JSON key/value 处理
```

性能指标：

```text
ns/op
allocation count/op
bytes allocated/op
cycles/op
instructions/op
branch misses
cache misses
```

工具：

```text
Google Benchmark
perf
heaptrack
cachegrind
```

---

## 12. 关键设计选择建议

### 12.1 第一版 Small metadata 用 CharT 存，不用 byte trick

建议第一版这样做：

```text
small_[kSmallCapacity] 保存 metadata
small_[0..kSmallCapacity-1] 保存字符
```

优点：

```text
多 CharT 简单
不依赖 endian
不依赖 byte aliasing
不容易 UB
TDD 容易
```

缺点：

```text
char 场景可能比 fbstring 少 1 个 SSO 字符
```

这是合理牺牲。后续可以给 `char` 特化做极限布局。

### 12.2 对 char 做特化优化

当 generic 版本稳定后，可以给 `char` / `char8_t` 做特化：

```cpp
template <class Traits, class Allocator>
class basic_string_core<char, Traits, Allocator> {
    // optimized fbstring-like layout
};
```

这样可以同时满足：

```text
generic CharT 正确
char 性能逼近 fbstring
```

不要用一套复杂 byte-level 逻辑强行覆盖所有 CharT。

推荐路线：

```text
generic core：稳定、清晰、多 CharT
char core：极限优化、对标 fbstring
```

### 12.3 Large String 不要过早特殊化

第一版：

```text
size > small_capacity 全部 heap exclusive
```

后续根据 benchmark 决定是否区分：

```text
Medium：普通 heap
Large：大容量增长策略 / mmap / refcount / page aligned allocation
```

但除非 benchmark 证明必要，不要引入 COW。

### 12.4 查找算法不要过早复杂化

第一版：

```text
Traits::find
Traits::compare
朴素 substring find
```

第二版：

```text
char 专用 memchr
char 专用 memmem 或 BMH
SIMD
```

---

## 13. 第一阶段测试清单

### Core 测试

```text
core_default_is_empty
core_default_data_not_null
core_default_null_terminated
core_small_size
core_small_capacity
core_small_max_size
core_small_embedded_null
core_heap_size
core_heap_capacity
core_heap_null_terminated
core_heap_embedded_null
core_copy_small
core_copy_heap
core_move_small
core_move_heap
core_clear_small
core_clear_heap
core_shrink_heap_to_small
```

### basic_string 基础测试

```text
default_construct
construct_from_cstr
construct_from_pointer_length
construct_from_string_view
construct_count_char
copy_construct
move_construct
copy_assign
move_assign
assign_cstr
data
c_str
size
capacity
empty
begin_end
operator_index
at_valid
at_out_of_range
front
back
```

### 修改测试

```text
reserve
resize_smaller
resize_larger
clear
shrink_to_fit
push_back
pop_back
append_cstr
append_pointer_length
append_string_view
append_string
append_self
assign
erase
insert
replace
```

### 查询测试

```text
compare_equal
compare_less
compare_greater
find_char
find_string
find_empty
find_missing
rfind
starts_with
ends_with
contains
```

### 多 CharT 测试

```text
char_small
char8_small
char16_small
char32_small
wchar_small

char_heap
char8_heap
char16_heap
char32_heap
wchar_heap

char_embedded_null
char16_embedded_null
char32_embedded_null
```

---

## 14. 风险点与规避方式

### 风险 1：多 CharT 与 fbstring 极限布局冲突

规避：

```text
先 generic 正确布局
后 char 专用优化布局
不要一开始用 byte-level marker 覆盖所有 CharT
```

### 风险 2：std::basic_string API 太多

规避：

```text
建立 API compatibility matrix
按阶段补接口
每个接口绑定测试
不要一次性写完所有 overload
```

### 风险 3：自引用修改操作

高风险 API：

```text
append
insert
replace
assign
```

规避：

```text
任何 src 指针来自当前字符串时，先识别 overlap
必要时先复制到临时 buffer 或先计算偏移
扩容后重新定位
```

### 风险 4：异常安全

规避：

```text
修改操作采用 allocate-copy-commit 模式
commit 前不破坏原对象
failing allocator 测试必须尽早加入
```

### 风险 5：Allocator 规则复杂

规避：

```text
MVP 只完整支持 std::allocator
allocator propagation 单独作为里程碑
不要把 allocator 复杂度混进初始存储布局开发
```

---

## 15. 推荐执行顺序

最合理的重启路线：

```text
1. 写 storage_layout.md，冻结 generic layout。
2. 写 api_compatibility.md，列出 std::basic_string 接口覆盖矩阵。
3. 搭建 GoogleTest + Google Benchmark + Sanitizer。
4. 实现 generic basic_string_core<CharT>。
5. 先只启用 char。
6. 完成 core 生命周期测试。
7. 封装 basic_string<char> MVP。
8. 做 std::string 差分测试。
9. 实现 reserve / resize / append。
10. 实现 erase / insert / replace。
11. 实现 find / compare。
12. 放开 char8_t。
13. 放开 char16_t / char32_t / wchar_t。
14. 完整 allocator 支持。
15. 与 std::string / fbstring benchmark。
16. 针对 char 做 fbstring-like 特化优化。
```

---

## 16. MVP 定义

第一版 MVP 建议定义为：

```text
aethermind::string MVP：

1. 支持 char。
2. 支持 Small + Heap exclusive ownership。
3. 支持 default/cstr/pointer-length/string_view/count 构造。
4. 支持 copy/move/destructor。
5. 支持 data/c_str/size/capacity/empty。
6. 支持 begin/end/operator[]/at/front/back。
7. 支持 reserve/resize/clear/shrink_to_fit。
8. 支持 push_back/pop_back/append/operator+=。
9. 支持 compare/find/starts_with/ends_with。
10. 通过 ASan/UBSan/LSan。
11. 通过 std::string 差分测试。
12. 有 Google Benchmark 对比 std::string/fbstring。
```

MVP 不要求：

```text
完整 allocator propagation
完整 iterator overload
完整 replace overload
COW
SIMD
safe over-read
stream / formatter
所有 CharT
```

---

## 17. 总体建议

这次从头设计时，建议把目标拆成两条线：

```text
主线 A：generic basic_string<CharT>
目标是正确、标准兼容、多 CharT。

主线 B：optimized basic_string<char>
目标是性能对标 fbstring。
```

不要一开始试图用一份极限 fbstring 布局同时覆盖所有 `CharT`。那会让设计过早复杂化。

最稳妥的技术路线是：

```text
先做 generic 正确版本；
再做 char 专用优化版本；
最后用统一 public API 封装二者。
```

也就是：

```cpp
template <class CharT, class Traits, class Allocator>
class basic_string {
private:
    detail::basic_string_core<CharT, Traits, Allocator> core_;
};
```

其中：

```text
basic_string_core<char>    使用 fbstring-like optimized layout
basic_string_core<char8_t> 可以复用 char-like optimized layout
basic_string_core<char16_t / char32_t / wchar_t> 使用 generic stable layout
```

这条路线能同时满足：

```text
功能逐步对齐 std::string
支持多字符类型
TDD 可控
char 性能对标 fbstring
后续优化空间充足
```
