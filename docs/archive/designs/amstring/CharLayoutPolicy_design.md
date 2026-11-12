# CharLayoutPolicy 设计文档

## 1. 文档目的

本文档定义 `CharLayoutPolicy` 的正式设计方案，作为 `amstring` 在 `char` 路径上的专用布局策略。

本文档参照 `docs/designs/amstring/GenericLayoutPolicy_design.md` 的结构与接口约束，并结合旧实现 `include/container/amstring.h` 中可复用的 24B 小字符串布局、末字节 metadata、capacity tagging 与 endian-aware 编码思路。

本文档用于指导以下工作：

- `CharLayoutPolicy` 的 TDD 实现
- `DefaultLayoutPolicy<char>` 从 `GenericLayoutPolicy<char>` 切换到 `CharLayoutPolicy`
- `BasicStringCore` 与 `CharLayoutPolicy` 的接口对接验证
- `char` 路径的差分测试、sanitizer 验证与 benchmark 对比

`CharLayoutPolicy` 的实现必须先以失败测试定义布局行为与不变量，再补最小实现使测试通过。首版目标是建立可验证的 `char` 专用布局基线，而不是一次性引入所有极限性能优化。

---

## 2. 设计定位

`CharLayoutPolicy` 的定位如下：

1. 面向 `char` 的专用布局策略。
2. 作为 `GenericLayoutPolicy<char>` 的性能特化路径，而不是替代多 `CharT` correctness baseline。
3. 保持 24B 对象头（64-bit 平台）。
4. 保持 23-char SSO capacity。
5. 仅表达 `Small / External` 两态布局。
6. 不引入独立 discriminator。
7. 不编码 `Medium / Large` layout state。
8. 不负责 allocator、traits 算法、异常安全与容器语义。

`CharLayoutPolicy` 可以使用更贴近 `fbstring` / 旧 `AMStringCore` 的 byte-level 编码，但必须保持与 `GenericLayoutPolicy` 接口同构，使 `BasicStringCore` 不需要知道底层布局细节。

---

## 3. 职责与边界

## 3.1 负责的内容

`CharLayoutPolicy` 负责：

- 定义 `char` 专用 24B 存储布局
- 定义 `Small / External` 状态编码
- 提供 `data() / size() / capacity() / category()`
- 定义 `Small` metadata 的 byte-level 编码规则
- 定义 `External` 的 `capacity_with_tag` 编码规则
- 提供 `InitEmpty / InitSmall / InitExternal`
- 提供 `SetSmallSize / SetExternalSize / SetExternalCapacity`
- 提供布局层不变量检查

## 3.2 不负责的内容

`CharLayoutPolicy` 不负责：

- allocator 分配与释放
- `Traits::copy / move / compare / length`
- `append / insert / erase / replace`
- `reserve / resize / shrink_to_fit`
- large 容量增长策略
- 异常安全
- 所有权语义管理
- safe over-read / SIMD / branchless trick 的首版实现

上述职责全部由 `BasicStringCore` 或后续 benchmark-gated 优化阶段承担。

---

## 4. 模板参数与命名规则

## 4.1 模板参数

`CharLayoutPolicy` 不接收模板参数：

```cpp
struct CharLayoutPolicy;
```

固定约束：

```cpp
using ValueType = char;
```

不接收：

- `CharT`
- `Traits`
- `Allocator`

原因是 `CharLayoutPolicy` 只负责 `char` 的布局编码，不负责字符算法和分配策略。

## 4.2 命名规则

命名规则与 `GenericLayoutPolicy` 保持一致：

- STL 风格接口使用小写：
  - `data()`
  - `size()`
  - `capacity()`
  - `category()`
  - `is_small()`
  - `is_external()`

- 内部 helper / 初始化 / 编解码 / invariant 函数使用 CamelCase：
  - `InitEmpty`
  - `InitSmall`
  - `InitExternal`
  - `SetSmallSize`
  - `SetExternalSize`
  - `SetExternalCapacity`
  - `CheckInvariants`
  - `PackCapacityWithTag`
  - `UnpackCapacity`
  - `GetProbeByte`

- 结构体/union 成员变量使用 snake_case：
  - `ExternalRep`
  - `Storage`
  - `data`
  - `size`
  - `capacity_with_tag`
  - `small`
  - `external`
  - `raw`

## 4.3 命名空间

本文档中的以下类型默认位于 `aethermind` 命名空间：

```cpp
namespace aethermind {

struct CharLayoutPolicy;

}
```

---

## 5. 支持范围

`CharLayoutPolicy` 只正式支持：

- `char`

不支持：

- `char8_t`
- `char16_t`
- `char32_t`
- `wchar_t`

非 `char` 类型继续使用 `GenericLayoutPolicy<CharT>`。

---

## 6. 基本类型定义

## 6.1 ExternalRep

```cpp
struct ExternalRep {
    char* data;
    std::size_t size;
    std::size_t capacity_with_tag;
};
```

约束：

- `sizeof(ExternalRep) == sizeof(void*) + 2 * sizeof(std::size_t)`
- 在 64-bit 平台下应为 24B

## 6.2 Storage

```cpp
union Storage {
    char small[sizeof(ExternalRep)];
    ExternalRep external;
    std::byte raw[sizeof(ExternalRep)];
};
```

约束：

- `sizeof(Storage) == sizeof(ExternalRep)`
- 在 64-bit 平台下应为 24B
- `Storage` 必须是 trivially copyable

`Storage` trivially copyable 是硬约束，因为 `BasicStringCore` 的 storage swap 可以依赖 raw byte copy。

---

## 7. 状态定义

## 7.1 Category

`CharLayoutPolicy` 定义三种分类值：

- `Small`
- `External`
- `Invalid`

其中：

- `Small` 和 `External` 为合法状态
- `Invalid` 仅用于调试与不变量检测，不是可持久化正常状态

## 7.2 不采用的旧状态

旧 `AMStringCore` 中存在 `isMedium` 与 `isLarge` 分类。`CharLayoutPolicy` 不采用这两个 layout state。

原因：

1. 当前 `amstring` 架构规定 layout policy 只表达 `Small / External`。
2. `Large` 是 `BasicStringCore` 层基于 capacity 计算出的策略语义，不是持久化 layout tag。
3. `Medium` 会增加状态迁移和 allocator 语义复杂度，不属于首版 correctness-preserving 特化范围。

---

## 8. 存储容量定义

定义：

```cpp
StorageBytes  = sizeof(ExternalRep)
SmallSlots    = StorageBytes
MetaSlot      = SmallSlots - 1
SmallCapacity = SmallSlots - 1
```

在 64-bit 平台下：

| 项 | 值 |
|---|---:|
| `StorageBytes` | 24 |
| `SmallSlots` | 24 |
| `MetaSlot` | 23 |
| `SmallCapacity` | 23 |

含义：

- `small[0 .. 22]`：有效字符区
- `small[23]`：metadata / category probe byte

---

## 9. 状态判别规则

## 9.1 Probe byte

`CharLayoutPolicy` 采用 **last-byte probe** 方案。

probe byte 定义为：

```cpp
raw[StorageBytes - 1]
```

即 `small[MetaSlot]`。

## 9.2 Category marker 定义

`CharLayoutPolicy` 首版采用 **2-bit category marker**，这是它与 `GenericLayoutPolicy<char>` 的核心布局差异。

定义：

```cpp
CategoryBits = 2
CategoryMask = is_little_endian ? 0xC0 : 0x03
SmallMetaMask = is_little_endian ? 0x3F : 0xFC
SmallMarker = 0x00
ExternalMarker = is_little_endian ? 0x80 : 0x02
```

含义：

- little-endian：probe byte 的最高 2 bits 表示 category marker
- big-endian：probe byte 的最低 2 bits 表示 category marker

选择 2-bit marker 的原因：

1. 与旧 `AMStringCore` 的 char-oriented capacity tagging 思路一致。
2. `char` 路径可以保留 62-bit external capacity payload，而不是 `GenericLayoutPolicy<char>` 的 56-bit payload。
3. `Small` 仍可通过末字节 metadata 完成 size 解码，不需要独立 discriminator。
4. `External` 只需要 marker 判别，不要求完整 probe byte 等于固定 tag。

## 9.3 Small 编码

`Small` 状态下，metadata 编码规则为：

```cpp
SmallMeta = SmallCapacity - size
```

存入 probe byte 时：

```cpp
EncodedSmallMeta = is_little_endian ? SmallMeta : (SmallMeta << CategoryBits)
```

因此：

- little-endian 下，`probe byte` 取值范围为 `[0, 23]`，最高 2 bits 为 `SmallMarker`
- big-endian 下，`probe byte` 取值范围为 `[0, 23] << 2`，最低 2 bits 为 `SmallMarker`

解码：

```cpp
SmallMeta = is_little_endian ? (probe & SmallMetaMask) : ((probe & SmallMetaMask) >> CategoryBits)
size = SmallCapacity - SmallMeta
```

## 9.4 External 编码

`External` 状态下，probe byte 的 category marker 必须等于 `ExternalMarker`。

注意：

- `External` 状态下 probe byte 的剩余 6 bits 是 capacity payload 的一部分。
- 因此 external 判别只能检查 marker bits，不能要求整个 probe byte 等于某个固定值。

## 9.5 判别逻辑

状态判别逻辑固定为：

```cpp
marker = probe & CategoryMask

marker == SmallMarker && decoded_small_meta <= SmallCapacity => Small
marker == ExternalMarker                                    => External
otherwise                                                   => Invalid
```

---

## 10. `capacity_with_tag` 编码

## 10.1 设计目标

`capacity_with_tag` 同时承担两项职责：

1. 保存真实 external capacity（字符数）
2. 让 probe byte 的 category bits 可判定为 `External`

`CharLayoutPolicy` 使用 2-bit category marker，因此 `capacity_with_tag` 只牺牲 2 个 bits 给状态判别，其余 bits 都用于真实 external capacity payload。

## 10.2 位划分

定义：

```cpp
CategoryBits = 2
WordBits     = sizeof(std::size_t) * 8
PayloadBits  = WordBits - CategoryBits
```

在 64-bit 平台下：

```cpp
PayloadBits = 62
```

## 10.3 little-endian 编码

little-endian 下 category marker 位于 `capacity_with_tag` 的最高 2 bits：

```cpp
Packed = Capacity | (std::size_t(ExternalMarker) << ((sizeof(std::size_t) - 1) * 8))
```

解码：

```cpp
ProbeByte = Packed >> ((sizeof(std::size_t) - 1) * 8)
Marker    = ProbeByte & CategoryMask
Capacity  = Packed & CapacityMask
```

其中：

```cpp
CapacityMask = ~(std::size_t(CategoryMask) << ((sizeof(std::size_t) - 1) * 8))
```

## 10.4 big-endian 编码

big-endian 下 category marker 位于 `capacity_with_tag` 的最低 2 bits：

```cpp
Packed = (Capacity << CategoryBits) | ExternalMarker
```

解码：

```cpp
ProbeByte = Packed & 0xFF
Marker    = ProbeByte & CategoryMask
Capacity  = Packed >> CategoryBits
```

## 10.5 Capacity payload 上界

`CharLayoutPolicy` 的最大 external capacity 受 62-bit payload 约束：

```cpp
max_external_capacity = (std::size_t{1} << PayloadBits) - 1
```

在 64-bit 平台下，该上界大于 `GenericLayoutPolicy<char>` 的 56-bit payload 上界。

若请求的 external capacity 超出该上界，则调用方必须在进入 `CharLayoutPolicy` 之前拒绝该请求，或由更高层抛出长度异常；布局层不得静默截断。

---

## 11. 接口定义

## 11.1 类型与常量

`CharLayoutPolicy` 应提供：

- `ValueType`
- `SizeType`

以及：

- `kStorageBytes`
- `kSmallSlots`
- `kSmallCapacity`
- `kCategoryBits`
- `kPackedWordBits`
- `kPayloadBits`
- `kProbeByteOffset`
- `kCategoryMask`
- `kSmallMetaMask`
- `kSmallMarker`
- `kExternalMarker`

并提供：

- `max_external_capacity()`

当前设计以静态常量作为主要常量访问方式。除非后续实现证明确有必要，否则不再额外要求提供与这些常量等价的 constexpr getter。

## 11.2 STL 风格接口

`CharLayoutPolicy` 应提供以下接口：

```cpp
static bool is_small(const Storage&) noexcept;
static bool is_external(const Storage&) noexcept;
static Category category(const Storage&) noexcept;

static constexpr SizeType max_external_capacity() noexcept;

static const char* data(const Storage&) noexcept;
static char* data(Storage&) noexcept;

static SizeType size(const Storage&) noexcept;
static SizeType capacity(const Storage&) noexcept;
```

## 11.3 CamelCase 内部接口

```cpp
static void InitEmpty(Storage&) noexcept;
static void InitSmall(Storage&, const char* src, SizeType size) noexcept;
static void InitExternal(Storage&, char* ptr, SizeType size, SizeType capacity) noexcept;

static void SetSmallSize(Storage&, SizeType size) noexcept;
static void SetExternalSize(Storage&, SizeType size) noexcept;
static void SetExternalCapacity(Storage&, SizeType capacity) noexcept;

static void CheckInvariants(const Storage&) noexcept;
```

上述 `SetSmallSize` / `SetExternalSize` / `SetExternalCapacity` 都是 **state-specific mutator**：

- 不负责 `Small / External` 状态切换
- 调用方必须在调用前已知当前状态
- 这些接口只负责在既定状态下维护布局一致性与 terminator/invariant

## 11.4 Probe 与编解码 helper

```cpp
static std::uint8_t GetProbeByte(const Storage&) noexcept;
static void SetSmallProbeByte(Storage&, SizeType small_meta) noexcept;

static constexpr SizeType EncodeSmallSizeToMeta(SizeType size) noexcept;
static constexpr SizeType DecodeSmallSizeFromMeta(std::uint8_t probe) noexcept;

static constexpr std::size_t CapacityMask() noexcept;

static std::size_t PackCapacityWithTag(SizeType capacity) noexcept;
static SizeType UnpackCapacity(std::size_t packed) noexcept;
static std::uint8_t UnpackMarker(std::size_t packed) noexcept;
```

这些 helper 主要用于实现、测试和 invariant 检查。`BasicStringCore` 不应依赖这些 helper。

---

## 12. 行为约束

## 12.1 `InitEmpty`

要求：

- 初始化为合法 `Small` 空对象
- `size() == 0`
- `capacity() == SmallCapacity`
- `data()[0] == '\0'`
- `category() == Small`

## 12.2 `InitSmall`

前置条件：

- `size <= SmallCapacity`
- 若 `size > 0`，`src != nullptr`

要求：

- 初始化为合法 `Small` 对象
- 复制 `src[0 .. size-1]`
- 保证 `data()[size] == '\0'`
- 不访问 external 存储
- 首版不得使用 safe over-read

## 12.3 `InitExternal`

前置条件：

- `ptr != nullptr`
- `size <= capacity`
- `capacity <= max_external_capacity()`

要求：

- 初始化为合法 `External` 对象
- `external.data == ptr`
- `external.size == size`
- `capacity()` 正确解码
- probe byte 可判定为 `External`
- 保证 `data()[size] == '\0'`

## 12.4 `SetSmallSize`

前置条件：

- 当前对象处于 `Small`
- `size <= SmallCapacity`

要求：

- 更新 small metadata
- 维护 `data()[size] == '\0'`

## 12.5 `SetExternalSize`

前置条件：

- 当前对象处于 `External`
- `size <= capacity()`

要求：

- 更新 external size
- 维护 `data()[size] == '\0'`

## 12.6 `SetExternalCapacity`

前置条件：

- 当前对象处于 `External`
- `capacity >= external.size`
- `capacity <= max_external_capacity()`

要求：

- 保持 external tag 不变
- 更新真实 capacity
- 不修改 `external.data`
- 不修改 `external.size`

---

## 13. 不变量

## 13.1 全局不变量

任意合法 `Storage` 必须满足：

1. `category()` 可正确区分为 `Small / External / Invalid`
2. `data() != nullptr`
3. `data()[size()] == '\0'`
4. `size() <= capacity()`

其中 `Invalid` 仅用于调试、断言与不变量检查。它不是合法的持久化运行时状态，任何公开初始化接口都不应构造出 `Invalid`。

## 13.2 Small 不变量

当 `category() == Small` 时：

1. category marker 等于 `SmallMarker`
2. decoded small meta 位于 `[0, SmallCapacity]`
3. `size()` 由 metadata 正确解码
4. `capacity() == SmallCapacity`
5. `data()` 指向对象内部 `small`
6. `data()[size()] == '\0'`

## 13.3 External 不变量

当 `category() == External` 时：

1. category marker 等于 `ExternalMarker`
2. `external.data != nullptr`
3. `external.size <= UnpackCapacity(external.capacity_with_tag)`
4. `external.data[external.size] == '\0'`
5. `capacity()` 按字符数计，而不是按字节数计

---

## 14. 与 BasicStringCore 的接口约束

`BasicStringCore` 可依赖 `CharLayoutPolicy` 的以下性质：

1. `InitEmpty / InitSmall / InitExternal` 能构造合法状态
2. `SetSmallSize / SetExternalSize / SetExternalCapacity` 能维护布局一致性
3. `data() / size() / capacity()` 始终可从布局中解码
4. `CheckInvariants()` 可在 debug / test 环境中持续调用
5. `CharLayoutPolicy` 不依赖 `Traits` 和 `Allocator`
6. `Storage` 为 trivially copyable

此外：

- moved-from object 恢复为合法 empty string 的责任由 `BasicStringCore` 保证
- `CharLayoutPolicy` 仅提供 `InitEmpty` 等布局原语，不单独承担对象生命周期语义恢复责任
- `BasicStringCore` 不得读取 `capacity_with_tag` 或 probe byte 的内部编码细节

## 14.1 与 `DefaultLayoutPolicy<CharT>` 的关系

在 `CharLayoutPolicy` 引入后，默认 layout selector 应调整为：

```cpp
template<typename CharT>
struct DefaultLayoutPolicy {
    using type = GenericLayoutPolicy<CharT>;
};

template<>
struct DefaultLayoutPolicy<char> {
    using type = CharLayoutPolicy;
};
```

这表示：

- `char` 默认走 `CharLayoutPolicy`
- `char8_t / char16_t / char32_t / wchar_t` 继续走 `GenericLayoutPolicy<CharT>`
- 外部 `BasicString` public API 不暴露该选择

---

## 15. 与旧 `AMStringCore` 的关系

旧 `include/container/amstring.h` 可以作为低层布局灵感来源，但不是新实现的架构来源。

## 15.1 可复用思想

可以复用：

- 24B object target
- union overlay small/external
- last byte small metadata
- endian-aware capacity tagging
- hot path 查询尽量简单

## 15.2 不可复用内容

不可复用：

- `Small / Medium / Large` 三态 layout 分类
- 未完成的 `InitMedium`
- allocator / lifecycle 与 layout 混合的 core 形态
- template `<typename Char>` 的泛化写法
- 首版 safe over-read

## 15.3 safe over-read 处理原则

旧 `InitSmall` 中的 page-boundary safe over-read 优化首版不采用。

原因：

1. sanitizer 和 object-bound 语义风险更高
2. 当前 M7 第一目标是建立 `char` 专用 layout correctness baseline
3. M6 benchmark baseline 已建立，后续可以单独通过 benchmark-gated 方式评估该优化

若未来引入 safe over-read，必须满足：

- 默认关闭或仅在明确宏保护下启用
- ASan / UBSan 下禁用
- 有独立测试覆盖 page-boundary 行为
- 有 benchmark 证明收益

---

## 16. 测试要求

## 16.1 静态断言

必须覆盖：

- `sizeof(ExternalRep) == 24`
- `sizeof(Storage) == 24`
- `std::is_trivially_copyable_v<Storage>`
- `kSmallCapacity == 23`
- `kCategoryBits == 2`
- `kPayloadBits == sizeof(std::size_t) * 8 - 2`
- `max_external_capacity()` 与 payload 上界一致
- `AmStringLayoutPolicy<CharLayoutPolicy, char>` satisfied

## 16.2 基础状态测试

必须覆盖：

- `InitEmpty`
- `InitSmall(empty)`
- `InitSmall(one char)`
- `InitSmall(max small)`
- `InitExternal`
- `category()`
- `data()`
- `size()`
- `capacity()`
- `CheckInvariants()`

## 16.3 编码/解码测试

必须覆盖：

- `EncodeSmallSizeToMeta`
- `DecodeSmallSizeFromMeta`
- `PackCapacityWithTag`
- `UnpackCapacity`
- `UnpackMarker`
- invalid category marker
- small probe metadata 超出 `SmallCapacity`

## 16.4 Selector 测试

必须覆盖：

- `DefaultLayoutPolicy<char>::type == CharLayoutPolicy`
- `DefaultLayoutPolicy<char8_t>::type == GenericLayoutPolicy<char8_t>`
- `DefaultLayoutPolicy<char16_t>::type == GenericLayoutPolicy<char16_t>`
- `DefaultLayoutPolicy<char32_t>::type == GenericLayoutPolicy<char32_t>`
- `DefaultLayoutPolicy<wchar_t>::type == GenericLayoutPolicy<wchar_t>`
- `BasicString<char>` 不暴露 `CharLayoutPolicy` 类型细节

## 16.5 Core / public / differential 回归

切换 selector 后必须运行：

- `BasicStringCore<char>` 生命周期与修改操作测试
- `BasicString<char>` public wrapper 测试
- `BasicString<char>` 与 `std::basic_string<char>` 差分测试
- amstring 全量相关单测
- benchmark baseline 对比

---

## 17. 当前阶段明确不做的事情

`CharLayoutPolicy` 首版不做：

- `medium / large` 三段式布局
- 独立 discriminator
- direct-map backend tag
- allocator 感知逻辑
- `Traits` 感知逻辑
- safe over-read
- SIMD / branchless trick
- `AMMallocAllocator` 接入
- public API 改动

---

## 18. 实施顺序建议

推荐按以下顺序推进：

1. 新增 `include/amstring/char_layout_policy.hpp`，仅声明/实现 `CharLayoutPolicy`
2. 新增 `tests/unit/amstring/test_char_layout_policy.cpp`
3. 先完成 layout policy 独立 TDD
4. 修改 `DefaultLayoutPolicy<char>` selector
5. 调整 selector 测试
6. 确保 `BasicStringCore<char>` 通过默认 selector 自动覆盖 `CharLayoutPolicy`
7. 运行 public wrapper 与 differential regression
8. 捕获 M7 首版 benchmark 对比

实现阶段不得通过修改 `BasicStringCore` 来适配 `CharLayoutPolicy` 的特殊编码；如果需要改 core，说明 layout policy 接口同构失败，应回到设计修正。

---

## 19. 正式设计结论

`CharLayoutPolicy` 是 `amstring` 的 `char` 专用 layout policy。它保持 24B 对象头和 23-char SSO capacity，采用 `Small / External` 两态布局，不引入独立 discriminator，并通过 last-byte probe 与 2-bit endian-aware category marker 实现状态判别和 capacity tagging。

因此：

1. `CharLayoutPolicy` 只服务于 `char`
2. `CharLayoutPolicy` 只负责布局，不负责分配与容器语义
3. `CharLayoutPolicy` 必须与 `GenericLayoutPolicy` 接口同构
4. `CharLayoutPolicy` 首版以 2-bit marker 的 correctness-preserving 特化为目标
5. safe over-read / SIMD / branchless trick 等激进优化必须后续通过 benchmark 与 sanitizer gate 单独引入

---

## 20. 一句话总结

`CharLayoutPolicy` 的正式定位是：**一个 `char` 专用、保持 24B/23-char SSO、采用 `Small / External` 两态布局，并通过 2-bit endian-aware category marker 提供更紧凑 external capacity payload 的布局策略。**
