# AMString 设计方案与开发计划（基于当前实现）

> 文档目标：将 `include/container/amstring.h` 当前雏形落地为对标 fbstring 的工业级高性能字符串容器。

---

## 1. 当前实现评估

### 1.1 已具备能力

- 三态布局雏形：`Small / Medium / Large`（通过容量字段高位编码 category）
- 小字符串（SSO）尺寸编码：`small_[maxSmallSize]`
- POD 操作基础：`PodCopy / PodMove / PodFill`
- `InitSmall` 快路径：页内 over-read 分支 + sanitize 兼容分支
- 端序兼容常量：`MagicConstants`

### 1.2 主要缺口

- 生命周期不完整：析构、拷贝、移动、赋值未形成闭环
- `InitMedium` 未实现，`Large` 路径未实现
- 缺少扩容与重分配策略：`reserve/growth/reallocate`
- 缺少核心 API：`append/resize/find/compare/...`
- 缺少异常安全策略定义（basic/strong guarantee）
- 缺少系统化验证：单测/模糊测试/sanitizer/benchmark

---

## 2. 目标设计

## 2.1 分层结构

- `AMStringCore<Char>`：只负责内存布局与低级读写（生命周期、扩容、迁移）
- `BasicAMString<Char, Traits>`：对外语义 API（行为对齐 `std::basic_string`）
- `using AMString = BasicAMString<char, std::char_traits<char>>`

## 2.2 三态语义

- **Small**：内联存储，0 堆分配
- **Medium**：独占堆内存（无共享）
- **Large**：共享块 + 引用计数；写前 `ensure_unique_for_write()`

> 说明：保留 Large 共享可降低大串拷贝成本，但所有写操作先分离，避免隐式 COW 语义风险。

## 2.3 Large 块头建议

```cpp
struct LargeHeader {
    std::atomic<uint32_t> refcnt;
    uint32_t reserved;
    size_t cap;
};
```

- 数据区紧随 header，`data = reinterpret_cast<Char*>(header + 1)`
- 释放时通过 `HeaderFromData()` 回溯

## 2.4 必备不变量

- 任意状态：`data()[size()] == '\0'`
- Small：`size <= maxSmallSize`
- Medium/Large：`capacity >= size`
- category bits 与真实存储状态一致
- `c_str()` 永不返回空指针

---

## 3. 对外接口草案（可直接落地）

```cpp
template <typename Char = char, typename Traits = std::char_traits<Char>>
class BasicAMString {
public:
    using value_type = Char;
    using size_type = size_t;
    static constexpr size_type npos = static_cast<size_type>(-1);

    // ctors / dtor / assign
    BasicAMString() noexcept;
    BasicAMString(const Char* s);
    BasicAMString(const Char* s, size_type n);
    BasicAMString(std::basic_string_view<Char, Traits> sv);
    BasicAMString(size_type n, Char ch);
    BasicAMString(const BasicAMString& other);
    BasicAMString(BasicAMString&& other) noexcept;
    BasicAMString& operator=(const BasicAMString& other);
    BasicAMString& operator=(BasicAMString&& other) noexcept;
    BasicAMString& operator=(std::basic_string_view<Char, Traits> sv);
    ~BasicAMString();

    // observers
    const Char* data() const noexcept;
    Char* data() noexcept;
    const Char* c_str() const noexcept;
    size_type size() const noexcept;
    size_type length() const noexcept;
    size_type capacity() const noexcept;
    bool empty() const noexcept;
    std::basic_string_view<Char, Traits> view() const noexcept;

    // element access
    Char& operator[](size_type i) noexcept;
    const Char& operator[](size_type i) const noexcept;
    Char& at(size_type i);
    const Char& at(size_type i) const;
    Char& front() noexcept;
    const Char& front() const noexcept;
    Char& back() noexcept;
    const Char& back() const noexcept;

    // modifiers
    void clear() noexcept;
    void reserve(size_type new_cap);
    void shrink_to_fit();
    void resize(size_type n);
    void resize(size_type n, Char ch);

    BasicAMString& append(const Char* s, size_type n);
    BasicAMString& append(std::basic_string_view<Char, Traits> sv);
    BasicAMString& append(const BasicAMString& s);
    BasicAMString& push_back(Char ch);
    void pop_back();

    BasicAMString& operator+=(std::basic_string_view<Char, Traits> sv);
    BasicAMString& operator+=(const BasicAMString& s);
    BasicAMString& operator+=(Char ch);

    // search / compare
    int compare(std::basic_string_view<Char, Traits> rhs) const noexcept;
    size_type find(std::basic_string_view<Char, Traits> needle, size_type pos = 0) const noexcept;

    friend bool operator==(const BasicAMString&, const BasicAMString&) noexcept = default;

private:
    AMStringCore<Char> core_;
};
```

---

## 4. `AMStringCore` 私有函数签名清单（实现骨架）

```cpp
template<typename Char>
class AMStringCore {
public:
    AMStringCore();
    AMStringCore(const Char* s, size_t n);
    AMStringCore(std::span<const Char> src);
    AMStringCore(const AMStringCore& other);
    AMStringCore(AMStringCore&& other) noexcept;
    AMStringCore& operator=(const AMStringCore& other);
    AMStringCore& operator=(AMStringCore&& other) noexcept;
    ~AMStringCore();

    const Char* c_str() const noexcept;
    Char* data() noexcept;
    const Char* data() const noexcept;
    size_t size() const noexcept;
    size_t capacity() const noexcept;
    bool empty() const noexcept;

    void clear() noexcept;
    void reserve(size_t new_cap);
    void shrink_to_fit();
    void resize(size_t n, Char fill = Char{});

    void append(const Char* s, size_t n);
    void append(std::span<const Char> src);
    void push_back(Char ch);
    void pop_back();

    int compare(std::span<const Char> rhs) const noexcept;
    size_t find(std::span<const Char> needle, size_t pos = 0) const noexcept;

private:
    enum class Category : uint8_t {
        isSmall = 0,
        isMedium = MagicConstants::kIsLittleEndian ? 0x80 : 0x02,
        isLarge = MagicConstants::kIsLittleEndian ? 0x40 : 0x01,
    };

    struct MediumLarge {
        Char* data_;
        size_t size_;
        size_t cap_; // encoded with category bits
        size_t capacity() const noexcept;
        void SetCapacity(size_t cap, Category cat) noexcept;
    };

    struct LargeHeader {
        std::atomic<uint32_t> refcnt;
        uint32_t reserved;
        size_t cap;
    };

    union {
        std::byte bytes_[sizeof(MediumLarge)];
        Char small_[sizeof(MediumLarge) / sizeof(Char)];
        MediumLarge ml_;
    };

    // lifecycle
    void reset_small_empty() noexcept;
    void destroy() noexcept;
    void swap(AMStringCore& other) noexcept;

    // category / layout
    Category category() const noexcept;
    bool is_small() const noexcept;
    bool is_medium() const noexcept;
    bool is_large() const noexcept;

    size_t small_size() const noexcept;
    void set_small_size(size_t sz) noexcept;

    size_t encoded_capacity() const noexcept;
    size_t decoded_capacity() const noexcept;
    void set_size(size_t sz) noexcept;

    Char* mutable_data_ptr() noexcept;
    const Char* data_ptr() const noexcept;

    // init
    void init_from(std::span<const Char> src);
    void InitSmall(std::span<const Char> src);
    void InitMedium(std::span<const Char> src);
    void InitLarge(std::span<const Char> src);

    // allocation / growth
    static size_t NextCapacity(size_t required) noexcept;
    static Char* AllocateChars(size_t cap);
    static void DeallocateChars(Char* p) noexcept;

    static LargeHeader* AllocateLarge(size_t cap);
    static void RetainLarge(Char* data) noexcept;
    static void ReleaseLarge(Char* data) noexcept;
    static LargeHeader* HeaderFromData(Char* data) noexcept;
    static const LargeHeader* HeaderFromData(const Char* data) noexcept;

    void ensure_capacity_for_append(size_t append_n);
    void reallocate_and_copy(size_t new_cap);
    void ensure_unique_for_write();
    void terminate_at(size_t sz) noexcept;

    // mutation helpers
    void append_raw(const Char* s, size_t n);
    void assign_raw(const Char* s, size_t n);
    void erase_range(size_t pos, size_t len);
    void insert_raw(size_t pos, const Char* s, size_t n);

    // constants
    static constexpr size_t lastChar = sizeof(MediumLarge) - 1;
    static constexpr size_t maxSmallSize = lastChar / sizeof(Char);
    static constexpr uint8_t categoryExtractMask = MagicConstants::kIsLittleEndian ? 0xC0 : 0x03;
    static constexpr size_t kCategoryShift = (sizeof(size_t) - 1) * 8;
    static constexpr size_t capacityExtractMask =
        MagicConstants::kIsLittleEndian
            ? ~(static_cast<size_t>(categoryExtractMask) << kCategoryShift)
            : 0x0;
};
```

---

## 5. 开发计划（执行版）

## WP1：Core 生命周期闭环（2~3 天）

- 实现析构、拷贝构造、移动构造、拷贝赋值、移动赋值
- 实现 `InitMedium` / `InitLarge` / `destroy`
- 打通 small/medium/large 基本构造与析构

**验收**：基础生命周期单测全绿

## WP2：容量增长与写路径（2~3 天）

- `reserve / reallocate / NextCapacity`
- `append / push_back / resize / clear`
- large 写前分离 `ensure_unique_for_write`

**验收**：追加、扩容、缩容行为正确；空串与边界长度无 UB

## WP3：读路径与比较（1~2 天）

- `compare`、`find`
- `view()/data()/c_str()` 行为稳定

**验收**：语义与 `std::string` 对齐（核心用例）

## WP4：健壮性验证（2 天）

- 边界与别名用例（self-append、erase/insert 边界）
- sanitizer 验证（ASAN/UBSAN/TSAN）

**验收**：无新增 sanitizer 报错

## WP5：性能优化（2 天）

- benchmark：短串构造、短串 append、中串拼接、大串拷贝
- 对比 `std::string` 与现有 `container/string.h`

**验收**：目标热点场景不劣于基线

## WP6：渐进式替换集成（持续）

- 先替换非关键路径，再替换热点路径
- 每次替换都附带回归测试

---

## 6. 测试与验收清单（DoD）

- 功能：构造/赋值/append/resize/find/compare 完整可用
- 正确性：单测通过 + sanitizer 通过
- 性能：关键场景达到或超过基线
- 维护性：注释仅保留不变量、约束与非显然设计理由

---

## 7. 风险与对策

- **风险**：位编码与端序分支出错
  - **对策**：集中封装 `category/capacity` 读写，补 `static_assert + 单测`
- **风险**：Large 引用计数与写路径竞态
  - **对策**：明确线程模型（默认非并发写）；写前唯一化
- **风险**：优化路径引入 sanitizer 误报/真报
  - **对策**：保留 sanitize 下保守路径

---

## 8. 推荐落地顺序（最小可行切片）

1. Small-only 可用版本（构造 + append + c_str）
2. 增加 Medium（无共享）
3. 增加 Large（共享 + 写前分离）
4. 增加 find/compare + benchmark

按上述顺序可持续交付，每一步都可测试、可回归、可性能对比。

---

## 9. 最小可编译实现模板（函数体空实现 + TODO + static_assert）

> 用途：一次性落地骨架，保证先可编译，再逐步填充实现。
>
> 说明：以下模板是“占位可编译”版本，函数体保留 TODO，不承诺运行语义正确性。

### 9.1 头文件模板（`include/container/amstring_skeleton.h`）

```cpp
#ifndef AETHERMIND_CONTAINER_AMSTRING_SKELETON_H
#define AETHERMIND_CONTAINER_AMSTRING_SKELETON_H

#include "any_utils.h"
#include "macros.h"

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>

namespace aethermind {

struct AMStringMagic {
    constexpr static bool kIsLittleEndian = std::endian::native == std::endian::little;
};

template<typename Char>
class AMStringCoreSkeleton {
public:
    using value_type = Char;
    using size_type = size_t;

    AMStringCoreSkeleton() noexcept {
        reset_small_empty();
    }

    AMStringCoreSkeleton(const Char* s, size_t n) {
        // TODO(amstring): implement dispatch to InitSmall/InitMedium/InitLarge
        (void)s;
        (void)n;
        reset_small_empty();
    }

    AMStringCoreSkeleton(const AMStringCoreSkeleton& other) {
        // TODO(amstring): implement deep copy / refcount retain by category
        (void)other;
        reset_small_empty();
    }

    AMStringCoreSkeleton(AMStringCoreSkeleton&& other) noexcept {
        // TODO(amstring): implement noexcept move by stealing storage
        (void)other;
        reset_small_empty();
    }

    AMStringCoreSkeleton& operator=(const AMStringCoreSkeleton& other) {
        // TODO(amstring): implement copy assignment with self-assignment guard
        (void)other;
        return *this;
    }

    AMStringCoreSkeleton& operator=(AMStringCoreSkeleton&& other) noexcept {
        // TODO(amstring): implement move assignment with destroy + steal
        (void)other;
        return *this;
    }

    ~AMStringCoreSkeleton() {
        // TODO(amstring): implement category-aware destroy
    }

    AM_NODISCARD const Char* c_str() const noexcept {
        // TODO(amstring): return small_ or heap data by category
        return small_;
    }

    AM_NODISCARD Char* data() noexcept {
        // TODO(amstring): return mutable data pointer by category
        return small_;
    }

    AM_NODISCARD const Char* data() const noexcept {
        return c_str();
    }

    AM_NODISCARD size_t size() const noexcept {
        // TODO(amstring): decode small size or return ml_.size_
        return 0;
    }

    AM_NODISCARD size_t capacity() const noexcept {
        // TODO(amstring): decode capacity for each category
        return maxSmallSize;
    }

    AM_NODISCARD bool empty() const noexcept {
        return size() == 0;
    }

    void clear() noexcept {
        // TODO(amstring): keep capacity, set logical size = 0 and null-terminate
        reset_small_empty();
    }

    void reserve(size_t new_cap) {
        // TODO(amstring): grow to new_cap when required
        (void)new_cap;
    }

    void shrink_to_fit() {
        // TODO(amstring): shrink medium/large if beneficial
    }

    void resize(size_t n, Char fill = Char{}) {
        // TODO(amstring): grow/shrink and fill newly appended chars
        (void)n;
        (void)fill;
    }

    void append(const Char* s, size_t n) {
        // TODO(amstring): ensure capacity + append raw range + null-terminate
        (void)s;
        (void)n;
    }

    void append(std::span<const Char> src) {
        append(src.data(), src.size());
    }

    void push_back(Char ch) {
        // TODO(amstring): append single char
        (void)ch;
    }

    void pop_back() {
        // TODO(amstring): remove one char if not empty
    }

    AM_NODISCARD int compare(std::span<const Char> rhs) const noexcept {
        // TODO(amstring): lexicographic compare
        (void)rhs;
        return 0;
    }

    AM_NODISCARD size_t find(std::span<const Char> needle, size_t pos = 0) const noexcept {
        // TODO(amstring): search from pos; return npos on miss
        (void)needle;
        (void)pos;
        return npos;
    }

    static constexpr size_t npos = static_cast<size_t>(-1);

private:
    enum class Category : uint8_t {
        isSmall = 0,
        isMedium = AMStringMagic::kIsLittleEndian ? 0x80 : 0x02,
        isLarge = AMStringMagic::kIsLittleEndian ? 0x40 : 0x01,
    };

    struct MediumLarge {
        Char* data_;
        size_t size_;
        size_t cap_;

        AM_NODISCARD size_t capacity() const noexcept {
            // TODO(amstring): decode cap_ with category bits
            return 0;
        }

        void SetCapacity(size_t cap, Category cat) noexcept {
            // TODO(amstring): encode cap + category into cap_
            (void)cap;
            (void)cat;
        }
    };

    struct LargeHeader {
        std::atomic<uint32_t> refcnt;
        uint32_t reserved;
        size_t cap;
    };

    union {
        std::byte bytes_[sizeof(MediumLarge)];
        Char small_[sizeof(MediumLarge) / sizeof(Char)];
        MediumLarge ml_;
    };

    void reset_small_empty() noexcept {
        small_[0] = Char{};
    }

    static constexpr size_t lastChar = sizeof(MediumLarge) - 1;
    static constexpr size_t maxSmallSize = lastChar / sizeof(Char);
    static constexpr uint8_t categoryExtractMask = AMStringMagic::kIsLittleEndian ? 0xC0 : 0x03;
    static constexpr size_t kCategoryShift = (sizeof(size_t) - 1) * 8;

    // Compile-time guards (follow repository style)
    static_assert(std::is_trivially_copyable_v<Char>, "AMStringCoreSkeleton requires trivially copyable Char");
    static_assert(sizeof(MediumLarge) % sizeof(Char) == 0, "Corrupt memory layout");
    static_assert(sizeof(Char*) == sizeof(size_t), "Pointer/size_t size assumption violation");
    static_assert((sizeof(size_t) & (sizeof(size_t) - 1)) == 0, "size_t must be power-of-two sized");
    static_assert(maxSmallSize > 0, "maxSmallSize must be positive");
    static_assert(categoryExtractMask != 0, "Category mask must be non-zero");
};

template<typename Char = char, typename Traits = std::char_traits<Char>>
class BasicAMStringSkeleton {
public:
    using value_type = Char;
    using size_type = size_t;
    static constexpr size_type npos = static_cast<size_type>(-1);

    BasicAMStringSkeleton() = default;
    explicit BasicAMStringSkeleton(const Char* s) {
        if (s == nullptr) {
            // TODO(amstring): replace with project-standard error reporting if needed
            core_ = Core();
            return;
        }
        size_type n = 0;
        while (s[n] != Char{}) {
            ++n;
        }
        core_ = Core(s, n);
    }

    BasicAMStringSkeleton(const Char* s, size_type n) : core_(s, n) {}

    AM_NODISCARD const Char* c_str() const noexcept {
        return core_.c_str();
    }

    AM_NODISCARD const Char* data() const noexcept {
        return core_.data();
    }

    AM_NODISCARD Char* data() noexcept {
        return core_.data();
    }

    AM_NODISCARD size_type size() const noexcept {
        return core_.size();
    }

    AM_NODISCARD bool empty() const noexcept {
        return core_.empty();
    }

    void clear() noexcept {
        core_.clear();
    }

    void reserve(size_type n) {
        core_.reserve(n);
    }

    void resize(size_type n, Char fill = Char{}) {
        core_.resize(n, fill);
    }

    BasicAMStringSkeleton& append(const Char* s, size_type n) {
        core_.append(s, n);
        return *this;
    }

    BasicAMStringSkeleton& append(std::basic_string_view<Char, Traits> sv) {
        core_.append(sv.data(), sv.size());
        return *this;
    }

    BasicAMStringSkeleton& push_back(Char ch) {
        core_.push_back(ch);
        return *this;
    }

    void pop_back() {
        core_.pop_back();
    }

private:
    using Core = AMStringCoreSkeleton<Char>;
    Core core_{};

    static_assert(std::is_trivially_copyable_v<Char>, "BasicAMStringSkeleton requires trivially copyable Char");
};

using AMStringSkeleton = BasicAMStringSkeleton<char>;

} // namespace aethermind

#endif // AETHERMIND_CONTAINER_AMSTRING_SKELETON_H
```

### 9.2 源文件模板（`src/container/amstring_skeleton.cpp`）

```cpp
#include "container/amstring_skeleton.h"

namespace aethermind {

// NOTE: Keep this file to host out-of-line definitions when TODOs are implemented.

} // namespace aethermind
```

### 9.3 一次性落地建议

1. 先把上述模板文件加入仓库，确保能通过编译。
2. 按 WP1 → WP6 顺序逐步替换 TODO。
3. 每替换一组 TODO，就补对应单元测试与 sanitizer 验证。

### 9.4 骨架安全护栏（Oracle 审核结论）

- TODO 只允许放在算法函数体（append/find/compare/growth），不要放在布局/标签编码/不变量建立路径。
- 不要为“临时可编译”改动对象表示（成员顺序、union 结构、tag 编码、对齐假设）。
- 保留并强化 ABI/layout 相关静态断言（`sizeof/alignof/编码掩码/容量推导`），防止静默漂移。
- 特殊成员函数（拷贝/移动/析构）必须保证对象始终处于有效状态；不能仅靠“占位返回”掩盖未实现逻辑。
- 若后续支持多字符宽度，优先使用 concept 或 `static_assert` 提前失败，不把不支持实例化拖到运行时。

---

## 10. 外部参考与实践要点（检索补充）

### 10.1 可核验参考

- Folly `fbstring` 源码入口：
  - https://github.com/facebook/folly/blob/main/folly/FBString.h
  - 原始文件（便于阅读）：https://github.com/facebook/folly/raw/refs/heads/main/folly/FBString.h
- C++ 标准字符串语义参考（连续存储、末尾空字符、API 约定）：
  - https://en.cppreference.com/w/cpp/string/basic_string
- LLVM `SmallString`（SSO 容器 API 设计参考，非 COW）：
  - https://llvm.org/doxygen/classllvm_1_1SmallString.html

### 10.2 对骨架实现有直接价值的 Do / Don’t

**Do**

- 先冻结对象表示与不变量，再填行为实现（避免 ABI 漂移）。
- 在“可编译骨架阶段”就保证 `data()/c_str()/size()` 对象状态一致（哪怕语义是最小占位）。
- 保持 `basic_string` 对齐语义：连续存储、`c_str()` 可用、`npos` 约定一致。
- 以 `SmallString` 风格优先保证小对象路径稳定与清晰，再逐步补 medium/large 增长策略。

**Don’t**

- 不要在未完成所有写路径前引入复杂隐式共享语义（尤其是隐式 COW 写入）。
- 不要把 TODO 放在布局编码与特殊成员有效性路径（构造/赋值/析构）上。
- 不要依赖“硬编码 SSO 字节数”而忽略 `sizeof(Char)` 与平台差异。
- 不要让占位实现返回会诱发 UB 的状态（例如未终止字符串、悬空指针、错误 category）。
