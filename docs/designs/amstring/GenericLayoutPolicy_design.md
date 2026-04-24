# GenericLayoutPolicy 设计文档

## 1. 文档目的

本文档定义 `GenericLayoutPolicy<CharT>` 的正式设计方案，作为 `amstring` 多 `CharT` 路径的统一布局基线。

本文档用于指导以下工作：

- `GenericLayoutPolicy<CharT>` 的实现
- `BasicStringCore` 与 `GenericLayoutPolicy` 的接口对接
- 单元测试、差分测试与不变量检查
- 后续 `CharLayoutPolicy` 特化设计的对照基线

---

## 2. 设计定位

`GenericLayoutPolicy<CharT>` 的定位如下：

1. 面向多 `CharT` 的通用布局策略。
2. 作为 `char / char8_t / char16_t / char32_t / wchar_t` 的 correctness baseline。
3. 保持 24B 对象头（64-bit 平台）。
4. 仅表达 `Small / External` 两态布局。
5. 不引入独立 discriminator。
6. 不负责 allocator、traits 算法、异常安全与容器语义。

`GenericLayoutPolicy<CharT>` 不是最终的 `char` 极限性能路径；`char` 的专用高性能实现由后续的 `CharLayoutPolicy` 承担。

---

## 3. 职责与边界

## 3.1 负责的内容

`GenericLayoutPolicy<CharT>` 负责：

- 定义 24B 存储布局
- 定义 `Small / External` 状态编码
- 提供 `data() / size() / capacity() / category()`
- 定义 `Small` metadata 编码规则
- 定义 `External` 的 `capacity_with_tag` 编码规则
- 提供 `InitEmpty / InitSmall / InitExternal`
- 提供 `SetSmallSize / SetExternalSize / SetExternalCapacity`
- 提供布局层不变量检查

## 3.2 不负责的内容

`GenericLayoutPolicy<CharT>` 不负责：

- allocator 分配与释放
- `Traits::copy / move / compare / length`
- `append / insert / erase / replace`
- `reserve / resize / shrink_to_fit`
- large 容量策略
- 异常安全
- 所有权语义管理

上述职责全部由 `BasicStringCore` 承担。

---

## 4. 模板参数与命名规则

## 4.1 模板参数

`GenericLayoutPolicy` 只有一个模板参数：

```cpp
template<typename CharT>
struct GenericLayoutPolicy;
```

不接收：

- `Traits`
- `Allocator`

原因是 `GenericLayoutPolicy` 只负责布局，不负责字符算法和分配策略。

## 4.2 命名规则

命名规则如下：

- STL 风格接口使用小写：
  - `data()`
  - `size()`
  - `capacity()`
  - `category()`
  - `is_small()`
  - `is_external()`

其中：

- `category()` 主要用于调试、单元测试与 invariant 检查
- 常规热路径优先使用 `is_small()` / `is_external()`，或由调用上下文直接确定状态

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
  - `ProbeMeta`

- 结构体/union 成员变量使用 snake_case：
  - `ExternalRep`
  - `Storage`
  - `data`
  - `size`
  - `capacity_with_tag`
  - `small`
  - `external`
  - `raw`

- 类中成员变量使用 snake_case_：
  - `storage_`
  - `allocator_`

这样做的目的是在保留 STL 风格查询接口与 CamelCase helper 的同时，使底层存储成员变量与普通类成员变量遵循项目统一命名约定，避免实现阶段在字段名风格上摇摆。

## 4.3 命名空间

本文档中的以下类型默认位于 `aethermind` 命名空间：

- `GenericLayoutPolicy<CharT>`
- `ExternalRep<CharT>`
- `Storage<CharT>`

即：

```cpp
namespace aethermind {

template<typename CharT>
struct GenericLayoutPolicy;

}
```

---

## 5. 支持范围

`GenericLayoutPolicy<CharT>` 正式支持以下 `CharT`：

- `char`
- `char8_t`
- `char16_t`
- `char32_t`
- `wchar_t`

约束：

- `CharT` 必须是 trivial type
- `CharT` 必须是 standard-layout type
- `sizeof(CharT)` 必须为 `1 / 2 / 4 / 8`

---

## 6. 基本类型定义

## 6.1 MetaWord

布局编码使用与 `CharT` 同宽的无符号整数作为 metadata word。

```cpp
template <size_t N>
struct UIntOfSize;

template <>
struct UIntOfSize<1> { using Type = uint8_t; };

template <>
struct UIntOfSize<2> { using Type = uint16_t; };

template <>
struct UIntOfSize<4> { using Type = uint32_t; };

template <>
struct UIntOfSize<8> { using Type = uint64_t; };

template <typename CharT>
using MetaWordT = typename UIntOfSize<sizeof(CharT)>::Type;
```

## 6.2 ExternalRep

```cpp
template <typename CharT>
struct ExternalRep {
    CharT* data;
    size_t size;
    size_t capacity_with_tag;
};
```

约束：

- `sizeof(ExternalRep<CharT>) == sizeof(void*) + 2 * sizeof(size_t)`
- 在 64-bit 平台下应为 24B

## 6.3 Storage

```cpp
template <typename CharT>
union Storage {
    CharT small[sizeof(ExternalRep<CharT>) / sizeof(CharT)];
    ExternalRep<CharT> external;
    std::byte raw[sizeof(ExternalRep<CharT>)];
};
```

约束：

- `sizeof(Storage<CharT>) == sizeof(ExternalRep<CharT>)`
- 在 64-bit 平台下应为 24B

---

## 7. 状态定义

## 7.1 Category

`GenericLayoutPolicy<CharT>` 定义三种分类值：

- `Small`
- `External`
- `Invalid`

其中：

- `Small` 和 `External` 为合法状态
- `Invalid` 仅用于调试与不变量检测，不是可持久化正常状态

---

## 8. 存储容量定义

定义：

```cpp
StorageBytes  = sizeof(ExternalRep<CharT>)
SmallSlots    = StorageBytes / sizeof(CharT)
MetaSlot      = SmallSlots - 1
SmallCapacity = SmallSlots - 1
```

含义：

- `small[0 .. MetaSlot-1]`：有效字符区
- `small[MetaSlot]`：metadata / probe slot

在 64-bit 平台下：

| CharT | sizeof(CharT) | SmallSlots | SmallCapacity |
|---|---:|---:|---:|
| char | 1 | 24 | 23 |
| char8_t | 1 | 24 | 23 |
| char16_t | 2 | 12 | 11 |
| char32_t | 4 | 6 | 5 |
| wchar_t(2B/4B) | 2/4 | 12/6 | 11/5 |

---

## 9. 状态判别规则

## 9.1 Probe slot

`GenericLayoutPolicy<CharT>` 采用 **probe slot 判别** 方案。

probe slot 定义为：

- 存储区最后一个 `CharT` 槽位
- 即 `small[MetaSlot]`
- 对应的字节区间为 `raw[ProbeByteOffset .. ProbeByteOffset + sizeof(CharT) - 1]`

## 9.2 Tag 定义

定义：

```cpp
ExternalTag = SmallCapacity + 1
```

### Small 状态下

probe slot 取值范围：

```cpp
[0, SmallCapacity]
```

### External 状态下

probe slot 取值：

```cpp
ExternalTag
```

## 9.3 Small 编码

`Small` 状态下，metadata 编码规则为：

```cpp
ProbeMeta = SmallCapacity - size
```

因此：

- `size == 0` => `ProbeMeta == SmallCapacity`
- `size == SmallCapacity` => `ProbeMeta == 0`

## 9.4 判别逻辑

状态判别逻辑固定为：

```cpp
ProbeMeta <= SmallCapacity   => Small
ProbeMeta == ExternalTag     => External
otherwise                    => Invalid
```

---

## 10. `capacity_with_tag` 编码

## 10.1 设计目标

`capacity_with_tag` 同时承担两项职责：

1. 保存真实 external capacity（字符数）
2. 让 probe slot 在 `External` 状态下解码出 `ExternalTag`

## 10.2 位划分

定义：

```cpp
ProbeBits   = sizeof(CharT) * 8
WordBits    = sizeof(size_t) * 8
PayloadBits = WordBits - ProbeBits
```

其中：

- `ProbeBits`：probe slot 对应位宽
- `PayloadBits`：capacity 可用位宽

## 10.3 little-endian 编码

```cpp
Packed = (ExternalTag << PayloadBits) | Capacity
```

### 解码

```cpp
Tag      = Packed >> PayloadBits
Capacity = Packed & CapacityMask
```

## 10.4 big-endian 编码

```cpp
Packed = (Capacity << ProbeBits) | ExternalTag
```

### 解码

```cpp
Tag      = Packed & TagMask
Capacity = Packed >> ProbeBits
```

## 10.5 Capacity payload 上界

`GenericLayoutPolicy` 的最大 external capacity 受 `PayloadBits` 约束：

- `sizeof(CharT) == 1` 时，payload 为 56 位
- `sizeof(CharT) == 2` 时，payload 为 48 位
- `sizeof(CharT) == 4` 时，payload 为 32 位

当前设计接受该上界，并将其作为 `GenericLayoutPolicy` 的正式容量边界。

若请求的 external capacity 超出 `PayloadBits` 可表示上界，则调用方必须在进入 `GenericLayoutPolicy` 之前拒绝该请求，或由更高层抛出长度异常；该越界请求不应在布局层被静默接受。

---

## 11. 接口定义

## 11.1 类型与常量

`GenericLayoutPolicy<CharT>` 应提供：

- `ValueType`
- `SizeType`
- `WordType`
- `ExternalType`
- `StorageType`

以及：

- `kStorageBytes`
- `kSmallSlots`
- `kMetaSlot`
- `kSmallCapacity`
- `kProbeBits`
- `kWordBits`
- `kPayloadBits`
- `kProbeByteOffset`
- `kExternalTag`
- `kMaxSmallMeta`

并提供：

- `max_external_capacity()`

当前设计以 **静态常量** 作为主要常量访问方式。除非后续实现证明确有必要，否则不再额外要求提供与这些常量等价的 constexpr getter，以避免测试与实现出现双入口。

## 11.2 STL 风格接口

`GenericLayoutPolicy<CharT>` 应提供以下接口：

```cpp
static bool is_small(const StorageType&) noexcept;
static bool is_external(const StorageType&) noexcept;
static Category category(const StorageType&) noexcept;

static constexpr SizeType max_external_capacity() noexcept;

static const CharT* data(const StorageType&) noexcept;
static CharT* data(StorageType&) noexcept;

static SizeType size(const StorageType&) noexcept;
static SizeType capacity(const StorageType&) noexcept;
```

## 11.3 CamelCase 内部接口

```cpp
static void InitEmpty(StorageType&) noexcept;
static void InitSmall(StorageType&, const CharT* src, SizeType size) noexcept;
static void InitExternal(StorageType&, CharT* ptr, SizeType size, SizeType capacity) noexcept;

static void SetSmallSize(StorageType&, SizeType size) noexcept;
static void SetExternalSize(StorageType&, SizeType size) noexcept;
static void SetExternalCapacity(StorageType&, SizeType capacity) noexcept;

static void CheckInvariants(const StorageType&) noexcept;
```

上述 `SetSmallSize` / `SetExternalSize` / `SetExternalCapacity` 都是 **state-specific mutator**：

- 不负责 `Small / External` 状态切换
- 调用方必须在调用前已知当前状态
- 这些接口只负责在既定状态下维护布局一致性与 terminator/invariant

## 11.4 私有 helper

```cpp
static WordType ProbeMeta(const StorageType&) noexcept;
static void SetProbeMeta(StorageType&, WordType) noexcept;
static void StoreProbeMetaAsCharT(StorageType&, WordType) noexcept;

static constexpr SizeType EncodeSmallSizeToMeta(SizeType size) noexcept;
static constexpr SizeType DecodeSmallSizeFromMeta(WordType meta) noexcept;

static constexpr size_t TagMask() noexcept;
static constexpr size_t CapacityMask() noexcept;

static size_t PackCapacityWithTag(SizeType capacity, WordType tag) noexcept;
static SizeType UnpackCapacity(size_t packed) noexcept;
static WordType UnpackTag(size_t packed) noexcept;
```

---

## 12. 行为约束

## 12.1 `InitEmpty`

要求：

- 初始化为合法 `Small` 空对象
- `size() == 0`
- `capacity() == SmallCapacity`
- `data()[0] == CharT{}`
- `category() == Small`

## 12.2 `InitSmall`

前置条件：

- `size <= SmallCapacity`

要求：

- 初始化为合法 `Small` 对象
- 保证 `data()[size] == CharT{}`
- 不访问外部存储

## 12.3 `InitExternal`

前置条件：

- `ptr != nullptr`
- `size <= capacity`

要求：

- 初始化为合法 `External` 对象
- `capacity()` 正确解码
- probe slot 可判定为 `External`

## 12.4 `SetSmallSize`

前置条件：

- 当前对象处于 `Small`
- `size <= SmallCapacity`

要求：

- 更新 small metadata
- 维护 `data()[size] == CharT{}`

## 12.5 `SetExternalSize`

前置条件：

- 当前对象处于 `External`
- `size <= capacity()`

要求：

- 更新 external size
- 维护 `data()[size] == CharT{}`

## 12.6 `SetExternalCapacity`

前置条件：

- 当前对象处于 `External`
- `capacity >= external.size`

要求：

- 保持 tag 不变
- 更新真实 capacity

---

## 13. 不变量

## 13.1 全局不变量

任意 `StorageType` 必须满足：

1. `category()` 可正确区分为 `Small / External / Invalid`
2. `data() != nullptr`
3. `data()[size()] == CharT{}`
4. `size() <= capacity()`

其中 `Invalid` 仅用于调试、断言与不变量检查。它不是合法的持久化运行时状态，任何公开初始化接口都不应构造出 `Invalid`。

## 13.2 Small 不变量

当 `category() == Small` 时：

1. probe slot 位于 `[0, SmallCapacity]`
2. `size()` 由 metadata 正确解码
3. `capacity() == SmallCapacity`
4. `data()` 指向对象内部存储
5. `data()[size()] == CharT{}`

## 13.3 External 不变量

当 `category() == External` 时：

1. probe slot 的值等于 `ExternalTag`
2. `external.data != nullptr`
3. `external.size <= UnpackCapacity(external.capacity_with_tag)`
4. `external.data[external.size] == CharT{}`
5. `capacity()` 按字符数计，而不是按字节数计

---

## 14. 与 BasicStringCore 的接口约束

`BasicStringCore` 可依赖 `GenericLayoutPolicy<CharT>` 的以下性质：

1. `InitEmpty / InitSmall / InitExternal` 能构造合法状态
2. `SetSmallSize / SetExternalSize / SetExternalCapacity` 能维护布局一致性
3. `data() / size() / capacity()` 始终可从布局中解码
4. `CheckInvariants()` 可在 debug / test 环境中持续调用
5. `GenericLayoutPolicy<CharT>` 不依赖 `Traits` 和 `Allocator`

此外：

- moved-from object 恢复为合法 empty string 的责任由 `BasicStringCore` 保证
- `GenericLayoutPolicy<CharT>` 仅提供 `InitEmpty` 等布局原语，不单独承担对象生命周期语义恢复责任

## 14.1 与 `DefaultLayoutPolicy<CharT>` 的关系

默认 layout selector 与 `GenericLayoutPolicy<CharT>` 的关系为：

```cpp
template<typename CharT>
struct DefaultLayoutPolicy {
    using type = GenericLayoutPolicy<CharT>;
};
```

这表示：

- 当前 generic 路径默认通过 `DefaultLayoutPolicy<CharT>` 选中 `GenericLayoutPolicy<CharT>`
- `char` 的后续高性能特化可以在 selector 层切换到 `CharLayoutPolicy`

---

## 15. 测试要求

## 15.1 静态断言

必须覆盖：

- `sizeof(ExternalRep<CharT>) == 24`
- `sizeof(Storage<CharT>) == 24`
- `SmallSlots / SmallCapacity` 对各 `CharT` 正确
- `ProbeBits / PayloadBits` 对各 `CharT` 正确

## 15.2 基础状态测试

每个 `CharT` 必测：

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

## 15.3 编码/解码测试

必须覆盖：

- `EncodeSmallSizeToMeta`
- `DecodeSmallSizeFromMeta`
- `PackCapacityWithTag`
- `UnpackCapacity`
- `UnpackTag`

## 15.4 多 `CharT` 测试矩阵

必须覆盖：

- `char`
- `char8_t`
- `char16_t`
- `char32_t`
- `wchar_t`

---

## 16. 当前阶段明确不做的事情

`GenericLayoutPolicy<CharT>` 当前阶段不做：

- `medium / large` 三段式布局
- 独立 discriminator
- direct-map backend tag
- allocator 感知逻辑
- `Traits` 感知逻辑
- `char` 专用极限优化
- safe over-read
- SIMD / branchless trick

---

## 17. 正式设计结论

`GenericLayoutPolicy<CharT>` 是 `amstring` 多 `CharT` 路径的正式布局基线。  
它保持 24B 对象头，采用 `Small / External` 两态布局，不引入独立 discriminator，并通过 `CharT` 宽度感知的 probe slot 与 `capacity_with_tag` 编码实现状态判别和布局解码。

因此：

1. `GenericLayoutPolicy<CharT>` 只依赖 `CharT`
2. `GenericLayoutPolicy<CharT>` 只负责布局，不负责分配与容器语义
3. `GenericLayoutPolicy<CharT>` 作为 `BasicStringCore` 的通用 correctness baseline
4. 后续 `CharLayoutPolicy` 在接口同构前提下对 `char` 进行高性能特化

---

## 18. 一句话总结

`GenericLayoutPolicy<CharT>` 的正式定位是：**一个只依赖 `CharT`、保持 24B、采用 `Small / External` 两态布局并作为多 `CharT` correctness baseline 的通用布局策略。**
