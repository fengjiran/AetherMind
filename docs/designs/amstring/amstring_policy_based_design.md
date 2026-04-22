# amstring 设计方案

> 目标：使用 **policy-based design**，用统一的 `basic_string_core` 作为算法与生命周期骨架，用 `LayoutPolicy` 等编译期策略类决定底层存储布局，构建一个支持多字符类型、功能逐步对齐 `std::basic_string`、并在 `char` 场景对标 `fbstring` 的高性能字符串库。

---

# 1. 设计目标

`amstring` 的目标不是简单重写一个 `std::string`，而是构建一个：

```text
接口风格接近 std::basic_string
内部采用 policy-based 设计
支持多 CharT
对 char 提供接近 fbstring 的性能
可逐步扩展 allocator / traits / find 优化
可通过差分测试验证语义正确性
```

推荐对外类型：

```cpp
namespace aethermind {

template <
    typename CharT,
    typename Traits = std::char_traits<CharT>,
    typename Allocator = std::allocator<CharT>
>
class basic_string;

using string    = basic_string<char>;
using u8string  = basic_string<char8_t>;
using u16string = basic_string<char16_t>;
using u32string = basic_string<char32_t>;
using wstring   = basic_string<wchar_t>;

}
```

---

# 2. 总体架构

核心分成四层：

```text
basic_string                    对外 API 层
basic_string_core               统一算法与生命周期骨架
LayoutPolicy                    底层存储布局策略
GrowthPolicy / Traits helpers   扩容与字符算法策略
```

结构关系：

```cpp
namespace aethermind {

template<typename CharT>
struct default_layout_policy;

struct default_growth_policy;

template<
    typename CharT,
    typename Traits,
    typename Allocator,
    typename LayoutPolicy,
    typename GrowthPolicy
>
class basic_string_core;

}

namespace aethermind {

template<typename CharT, typename Traits, typename Allocator>
class basic_string {
private:
    using layout_policy =
        typename default_layout_policy<CharT>::type;
    using growth_policy = default_growth_policy;

    basic_string_core<
        CharT, Traits, Allocator, layout_policy, growth_policy
    > core_;
};

}
```

这个结构的核心思想是：

```text
字符串语义统一
底层布局可替换
char 可特化优化
宽字符不被 char 的极限布局绑架
```

## 2.1 命名空间与目录结构约定

本项目采用以下约定：

**命名空间**：使用 `aethermind::` 命名空间

**目录结构**：采用平铺结构，所有 amstring 相关头文件直接放在 `include/amstring/` 目录下：

```text
include/amstring/
├── basic_string.hpp       # 对外 API 层
├── string.hpp             # 类型别名 (string, u8string, etc.)
├── string_fwd.hpp         # 前置声明
├── config.hpp             # 配置常量
├── core.hpp               # basic_string_core 实现
├── layout_policy.hpp      # LayoutPolicy 接口定义
├── stable_layout_policy.hpp   # stable_layout_policy 实现
├── compact_layout_policy.hpp  # compact_layout_policy 实现（预留）
├── growth_policy.hpp      # GrowthPolicy 实现
├── allocator_support.hpp  # allocator 辅助工具
├── invariant.hpp          # invariant 检查
└── char_algorithms.hpp    # Traits 辅助算法
```

---

# 3. Policy-Based 设计原则

## 3.1 `basic_string_core` 负责什么

`basic_string_core` 是统一骨架，负责：

- 对象生命周期
- 拷贝/移动/赋值语义
- reserve / resize / clear / shrink_to_fit
- append / assign / erase / insert / replace 的流程控制
- overlap 检测
- allocator 调用
- grow policy 调用
- 提交式异常安全
- invariant 检查

它**不负责**：

- small metadata 的字节级布局
- category 的 bit 编码
- small size 如何压缩
- heap capacity bits 如何嵌入
- endian-aware marker 解码

这些全部交给 `LayoutPolicy`。

## 3.2 `LayoutPolicy` 负责什么

`LayoutPolicy` 负责：

- `storage_type`
- small / heap / large 的物理表示
- `is_small()` / `category()`
- `data()` / `size()` / `capacity()`
- `set_small_size()`
- `set_heap_size()`
- `set_heap_capacity()`
- `init_empty()`
- `init_small()`
- `init_heap()`
- `destroy_heap_if_needed()`
- `check_invariants()`

也就是说：

```text
core 决定“什么时候扩容、怎么拷贝、怎么提交”
layout 决定“数据究竟存在哪、怎么编码、怎么取出来”
```

## 3.3 `GrowthPolicy` 负责什么

`GrowthPolicy` 只负责容量增长决策：

- `next_capacity(old_cap, required_cap)`
- small -> heap 时的最小初始 heap capacity
- heap 扩容倍率
- 大字符串时是否保守增长

不应混进 layout。

---

# 4. 关键 Policy 划分

推荐至少定义两套 layout policy。

## 4.1 `stable_layout_policy<CharT>`

用于：

- `char16_t`
- `char32_t`
- `wchar_t`
- 开发初期的所有 `CharT`
- generic fallback

特点：

- metadata 存在最后一个 `CharT`
- 尽量按 `CharT` 槽位解释，而不是按最后一个 byte 解释
- 逻辑上尽量端序无关
- 适合多字符类型
- 可读性强，测试简单

## 4.2 `compact_layout_policy<char>`

用于：

- `char`
- 后续可选 `char8_t`

特点：

- 24-byte object 目标
- 23-char SSO 目标
- byte-level marker
- packed category bits
- endian-aware 编码/解码
- 尽量接近 fbstring

## 4.3 默认 policy 选择

```cpp
template<typename CharT>
struct default_layout_policy {
    using type = stable_layout_policy<CharT>;
};

template<>
struct default_layout_policy<char> {
    using type = compact_layout_policy<char>;
};

// 可选，第二阶段再放开
template<>
struct default_layout_policy<char8_t> {
    using type = stable_layout_policy<char8_t>;
};
```

建议第一阶段先这样：

```text
char      -> stable_layout_policy<char>
char8_t   -> stable_layout_policy<char8_t>
宽字符     -> stable_layout_policy<...>
```

等 generic 版本全部稳定后，再切换：

```text
char -> compact_layout_policy<char>
```

这是风险最低的路线。

---

# 5. `basic_string_core` 设计方案

## 5.1 类模板骨架

```cpp
namespace aethermind {

template<
    typename CharT,
    typename Traits,
    typename Allocator,
    typename LayoutPolicy,
    typename GrowthPolicy
>
class basic_string_core {
public:
    using value_type      = CharT;
    using traits_type     = Traits;
    using allocator_type  = Allocator;
    using layout_policy   = LayoutPolicy;
    using growth_policy   = GrowthPolicy;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer         = CharT*;
    using const_pointer   = const CharT*;

    using storage_type = typename layout_policy::storage_type;

public:
    basic_string_core() noexcept;
    explicit basic_string_core(const allocator_type& a) noexcept;

    basic_string_core(const CharT* s, size_type n,
                      const allocator_type& a = allocator_type{});

    basic_string_core(const basic_string_core&);
    basic_string_core(basic_string_core&&) noexcept;
    ~basic_string_core();

    basic_string_core& operator=(const basic_string_core&);
    basic_string_core& operator=(basic_string_core&&) noexcept;

    const_pointer data() const noexcept;
    pointer       data() noexcept;
    const_pointer c_str() const noexcept;

    size_type size() const noexcept;
    size_type capacity() const noexcept;
    bool empty() const noexcept;

    void clear() noexcept;
    void reserve(size_type new_cap);
    void resize(size_type count);
    void resize(size_type count, CharT ch);
    void shrink_to_fit();

    void assign(const CharT* s, size_type n);
    void append(const CharT* s, size_type n);
    void append_fill(size_type count, CharT ch);

    void erase(size_type pos, size_type count);
    void insert(size_type pos, const CharT* s, size_type n);
    void replace(size_type pos, size_type count, const CharT* s, size_type n);

    void swap(basic_string_core&) noexcept;

    void check_invariants() const noexcept;

private:
    allocator_type& alloc_ref() noexcept;
    const allocator_type& alloc_ref() const noexcept;

    bool is_small() const noexcept;
    bool is_heap() const noexcept;

    void reset_empty() noexcept;
    void destroy_heap_if_needed() noexcept;

    size_type recommend_capacity(size_type required) const noexcept;
    void reallocate_and_copy(size_type new_cap);
    void ensure_capacity_for_append(size_type append_n);

    bool source_overlaps(const CharT* s, size_type n) const noexcept;

private:
    storage_type storage_;
    [[no_unique_address]] allocator_type alloc_;
};

}
```

## 5.2 `basic_string_core` 的职责边界

核心原则：

### core 里允许写的逻辑
- “若容量不足，先分配新 buffer，再复制旧内容，再 commit”
- “若 source 与当前 data 重叠，先转换成 offset 或临时缓冲”
- “append/insert/replace 完成后补 null terminator”
- “copy assignment 提供 strong exception guarantee”
- “move assignment 根据 allocator traits 决定窃取还是重分配”

### core 里不允许写死的逻辑
- “small size 存在最后一个 byte”
- “category 在 cap 的高两位”
- “小端机器 marker 在哪一位”
- “small capacity 固定是 23”

这些都只能经由 layout policy 访问。

---

# 6. `LayoutPolicy` 接口设计

推荐 `LayoutPolicy` 提供静态接口，而不是实例对象。因为：

- 无状态
- 可内联
- 更适合基础库
- 不增加对象体积

## 6.1 接口骨架

```cpp
template<typename CharT>
struct stable_layout_policy {
    using value_type  = CharT;
    using size_type   = std::size_t;
    using storage_type = ...;

    static constexpr bool kSupportsCompactBytePacking = false;
    static constexpr bool kEndianAware = false;

    static void init_empty(storage_type&) noexcept;

    static bool is_small(const storage_type&) noexcept;
    static bool is_heap(const storage_type&) noexcept;

    static const CharT* data(const storage_type&) noexcept;
    static CharT*       data(storage_type&) noexcept;

    static size_type size(const storage_type&) noexcept;
    static size_type capacity(const storage_type&) noexcept;

    static void set_size(storage_type&, size_type) noexcept;
    static void set_capacity(storage_type&, size_type) noexcept;

    static void init_small(storage_type&, const CharT* src, size_type n) noexcept;
    static void init_heap(storage_type&, CharT* ptr, size_type size, size_type cap) noexcept;

    static CharT* heap_ptr(storage_type&) noexcept;
    static const CharT* heap_ptr(const storage_type&) noexcept;

    static void destroy_heap(storage_type&) noexcept;

    static void check_invariants(const storage_type&) noexcept;
};
```

## 6.2 `LayoutPolicy` 最小职责约束

每个 layout policy 必须保证：

- `data(storage)[size(storage)] == CharT{}`
- `size(storage) <= capacity(storage)`
- empty 状态可合法返回 `data()`
- moved-from 后可恢复为 empty
- heap 状态下 `heap_ptr()` 有效
- `check_invariants()` 可在 debug 中安全运行

---

# 7. `stable_layout_policy<CharT>` 设计

这是 generic 主力布局。

## 7.1 设计原则

- 以 `MediumLarge<CharT>` 大小为总存储预算
- small 区使用 `CharT[]`
- small 的最后一个 `CharT` 保存 metadata
- 不依赖最后一个 byte
- 不显式依赖端序
- 优先保证跨 `CharT` 正确性

## 7.2 推荐布局

```cpp
template<typename CharT>
struct stable_layout_policy {
    struct heap_rep {
        CharT* data;
        std::size_t size;
        std::size_t capacity_with_tag;
    };

    static constexpr std::size_t kStorageBytes = sizeof(heap_rep);
    static constexpr std::size_t kSmallArraySize = kStorageBytes / sizeof(CharT);
    static constexpr std::size_t kSmallCapacity = kSmallArraySize - 1;

    union storage_type {
        CharT small[kSmallArraySize];
        heap_rep heap;
        std::byte raw[kStorageBytes];
    };
};
```

其中：

- `small[0..kSmallCapacity-1]` 存有效字符
- `small[kSmallCapacity]` 存 small metadata
- heap 模式下 `capacity_with_tag` 带状态位

## 7.3 small metadata 编码

建议：

```text
small[kSmallCapacity] = kSmallCapacity - size
```

读取 size：

```text
size = kSmallCapacity - small[kSmallCapacity]
```

这样对不同 `CharT` 都成立，只要 metadata 值不超过 `CharT` 可表示范围。这里一定满足。

## 7.4 适用范围

推荐用于：

- `char16_t`
- `char32_t`
- `wchar_t`
- `char8_t` 开发初期
- `char` 开发初期

---

# 8. `compact_layout_policy<char>` 设计

这是 `char` 的性能冲刺版本。

## 8.1 设计目标

- `sizeof(string)` 尽量 24 bytes
- SSO 尽量 23 chars
- 尽量接近 fbstring 的 small/medium/large 编码方式
- 端序感知，但复杂性限制在 policy 内部

## 8.2 设计原则

- 仅对 `sizeof(CharT) == 1` 启用
- `storage_type` 中可以直接使用 `bytes_`
- small size/category 允许复用最后一个 byte
- heap capacity 可复用高位/低位 tag
- endian-aware codec 仅在 policy 内部使用

## 8.3 结构草案

```cpp
template<>
struct compact_layout_policy<char> {
    using CharT = char;
    using size_type = std::size_t;

    enum class category : unsigned char {
        small,
        medium,
        large
    };

    struct heap_rep {
        char* data;
        std::size_t size;
        std::size_t cap_with_tag;
    };

    union storage_type {
        char small[sizeof(heap_rep)];
        heap_rep heap;
        std::byte raw[sizeof(heap_rep)];
    };

    static constexpr bool kSupportsCompactBytePacking = true;
    static constexpr bool kEndianAware = true;

    // ... data/size/capacity/category 编解码
};
```

## 8.4 端序处理方式

不要把端序作为顶层 public policy 暴露。  
而是在 `compact_layout_policy<char>` 内部做：

```cpp
using codec = compact_layout_codec<std::endian::native>;
```

由 `codec` 提供：

- `encode_small_size()`
- `decode_small_size()`
- `encode_category()`
- `decode_category()`
- `extract_capacity()`
- `store_capacity_and_category()`

这样端序复杂性不会外溢。

---

# 9. `GrowthPolicy` 设计

建议单独 policy 化，但保持很简单。

## 9.1 接口

```cpp
struct default_growth_policy {
    static constexpr std::size_t min_heap_capacity(std::size_t required) noexcept;
    static constexpr std::size_t next_capacity(std::size_t old_cap,
                                               std::size_t required) noexcept;
};
```

## 9.2 默认策略

推荐：

- old_cap 很小时：至少翻倍
- 中等容量：1.5x
- 始终满足 `>= required`
- 留一个 null terminator

例如：

```text
new_cap = max(required, old_cap + old_cap / 2, small_heap_floor)
```

## 9.3 为什么独立成 policy

这样可以后续加入：

- memory-saving policy
- aggressive growth policy
- page-aware growth policy
- allocator-aware growth policy

而不污染 core。

---

# 10. 对外 `basic_string` 设计

`basic_string` 不应把 policy 暴露给用户。  
对外只保留标准风格模板参数：

```cpp
template<typename CharT, typename Traits, typename Allocator>
class basic_string {
public:
    using value_type      = CharT;
    using traits_type     = Traits;
    using allocator_type  = Allocator;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = CharT&;
    using const_reference = const CharT&;
    using pointer         = CharT*;
    using const_pointer   = const CharT*;
    using iterator        = CharT*;
    using const_iterator  = const CharT*;

private:
    using layout_policy =
        typename default_layout_policy<CharT>::type;
    using growth_policy = default_growth_policy;

    basic_string_core<
        CharT, Traits, Allocator, layout_policy, growth_policy
    > core_;
};
```

这样：

- 用户看到的是标准库风格
- 内部仍然是 policy-based design
- 不把实现复杂度暴露到 API 层

---

# 11. 生命周期与异常安全设计

这是 `basic_string_core` 的重点。

## 11.1 构造/析构
- default constructor：初始化为 empty small
- cstr / ptr+len constructor：根据 `n` 选择 small 或 heap
- destructor：若 heap，则释放
- moved-from：reset 成合法 empty

## 11.2 copy constructor
- small：直接复制 storage
- heap：深拷贝新 buffer
- allocator：根据 `select_on_container_copy_construction`

## 11.3 move constructor
- allocator 可兼容时：直接窃取 heap
- small：直接复制 small storage
- moved-from 源对象 reset

## 11.4 copy assignment
推荐采用：

```text
allocate-copy-commit
```

即：
1. 先准备新状态
2. 成功后再覆盖旧状态
3. 提供 strong exception guarantee

## 11.5 move assignment
- allocator 相等或 `is_always_equal`：直接窃取
- 否则回退为深拷贝移动

---

# 12. 算法骨架设计

`basic_string_core` 统一这些流程。

## 12.1 `append`
流程：

1. 检测 `n == 0`
2. 检测 source 是否 overlap
3. 计算新 size
4. 若 capacity 足够，直接写入
5. 若不足，先分配新 buffer
6. 复制旧内容和新内容
7. 写 null terminator
8. commit 新状态

## 12.2 `insert`
流程：

1. 检查 `pos`
2. 检查 overlap
3. 若原地可做，移动尾部腾位置
4. 否则分配新 buffer 后三段复制
5. commit

## 12.3 `replace`
本质是：
- erase + insert 的统一优化版本

## 12.4 `resize`
- 变小：截断并补 null
- 变大：确保容量，填充字符，补 null

这些流程对所有 layout 都一致，所以应该在 core 里实现。

---

# 13. 多字符类型支持策略

## 13.1 推荐支持顺序

### 阶段 1
- `char`

### 阶段 2
- `char8_t`

### 阶段 3
- `char16_t`
- `char32_t`

### 阶段 4
- `wchar_t`

## 13.2 推荐实现顺序

### 第一步
所有类型都先走：

```text
stable_layout_policy<CharT>
```

### 第二步
在 generic 版本全部稳定后，再切换：

```text
char -> compact_layout_policy<char>
```

### 第三步
若 benchmark 证明有意义，再评估：

```text
char8_t -> compact_layout_policy<char8_t>
```

不建议对：

```text
char16_t / char32_t / wchar_t
```

做端序优化布局。

---

# 14. TDD 方案

这个设计必须用 TDD 驱动，否则 policy 组合会失控。

## 14.1 测试分层

### A. Layout policy 单测
分别验证：

- `stable_layout_policy<CharT>`
- `compact_layout_policy<char>`

测试：
- empty
- small max
- heap
- size/capacity/data 提取
- metadata 编码解码
- invariant

### B. core 单测
对 `basic_string_core<..., stable_layout_policy<...>>`
以及 `basic_string_core<..., compact_layout_policy<char>>`
分别跑同一套生命周期与修改 API 测试。

### C. public API 单测
对 `basic_string<CharT>` 测：

- 构造
- copy/move
- reserve/resize
- append/assign
- erase/insert/replace
- find/compare

### D. 差分测试
与：
- `std::string`
- `std::u8string`
- `std::u16string`
- `std::u32string`
- `std::wstring`

逐步对照。

### E. 随机操作测试
尤其对 mutation API 做随机差分。

## 14.2 layout policy 测试要点

### stable layout
- `small[kSmallCapacity]` metadata 正确
- size/capacity/data 正确
- heap rep 与 small rep 切换正确
- `CharT{}` null terminator 正确

### compact layout
- little-endian / big-endian codec 正确
- category bits 正确
- SSO 23-char 正确
- heap capacity 解码正确
- 不能在 sanitizer 下启用危险 fast path

---

# 15. 性能策略

## 15.1 generic baseline
先用 `stable_layout_policy<char>` 完成功能正确版本，建立基线。

## 15.2 `char` 优化
切换到 `compact_layout_policy<char>` 后，对比：

- construct small
- copy small
- move small
- append small
- compare
- find
- `vector<string>` push_back
- 日志拼接/路径拼接/JSON key 处理

## 15.3 优化约束
任何优化必须满足：

- 不破坏 public API
- 不破坏 allocator 语义
- 不破坏差分测试
- 不破坏 invariant
- 有 benchmark 证据

---

# 16. 最终推荐方案

可以把整个 amstring 方案概括为：

## 16.1 核心架构
- `basic_string`：标准风格对外接口
- `basic_string_core`：统一算法与生命周期骨架
- `LayoutPolicy`：底层存储布局策略
- `GrowthPolicy`：扩容策略

## 16.2 layout 分层
- `stable_layout_policy<CharT>`：多字符类型的稳定布局
- `compact_layout_policy<char>`：`char` 的 fbstring-like 紧凑布局

## 16.3 实现路线
- 先用 stable layout 打通所有核心功能
- 再为 `char` 引入 compact layout 特化
- 最后 benchmark 驱动优化

---

# 17. 方案总结

这版 amstring 设计，本质上是：

> **用统一的 `basic_string_core` 承载字符串生命周期、异常安全和修改算法；用 `LayoutPolicy` 在编译期注入不同的底层存储布局；其中宽字符走稳定的 generic layout，`char` 走端序感知的紧凑优化 layout。**

这是一个兼顾：

- 可维护性
- 多字符类型支持
- TDD 可控性
- `char` 性能对标 fbstring

的设计方向。
