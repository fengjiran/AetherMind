# amstring 存储架构设计文档

## 1. 文档目的

本文档定义 `amstring` 的正式存储架构方案，作为后续实现、测试、性能优化与文档编写的统一基线。

---

## 2. 总体设计

### 2.1 布局状态

`amstring` 采用两段式对象布局：

- `Small`
- `External`

其中：

- `Small` 表示对象内部内联存储
- `External` 表示对象外部存储

### 2.2 External 子语义

`External` 内部再按资源管理策略区分语义层级：

- `Normal`
- `Large`

该区分属于 **External 语义层**，不属于对象布局状态。

### 2.3 双 layout policy

内部布局采用两条实现路径：

1. `GenericLayoutPolicy<CharT>`
   - 面向 `char / char8_t / char16_t / char32_t / wchar_t`
   - 作为多 `CharT` 的 correctness baseline

2. `CharLayoutPolicy`
   - 仅面向 `char`
   - 作为后续高性能优化路径

### 2.4 allocator 路线

第一阶段默认使用：

- `std::allocator`

后续通过 allocator adapter 接入：

- `ammalloc`

从第一阶段开始，`amstring` 必须是 allocator-aware 容器。

### 2.5 公共主模板命名

公共主模板正式命名为：

- `BasicString`

后续代码、文档、测试与类型别名均以 `BasicString` 为统一命名基线。

---

## 3. 分层架构

```text
BasicString
  └── BasicStringCore
       ├── layout policy
       │    ├── GenericLayoutPolicy<CharT>
       │    └── CharLayoutPolicy
       │
       ├── allocator / traits / lifecycle
       │
       └── ExternalPolicy
            ├── normal
            └── large
```

### 3.1 `BasicString`

职责：

- 提供 `std::basic_string` 风格公共接口
- 提供类型别名与容器语义
- 不负责底层布局管理

### 3.2 `BasicStringCore`

职责：

- 生命周期管理
- allocator 分配与释放
- `reserve / resize / clear / append / assign / shrink_to_fit`
- 异常安全
- 调用 `Traits`
- 调用 `LayoutPolicy`
- 管理 `External` 语义层策略

### 3.3 `LayoutPolicy`

职责：

- 定义对象布局
- 编码/解码 `Small / External`
- 提供 `data() / size() / capacity() / category()`
- 提供状态编码规则
- 提供 invariant 检查

### 3.4 `ExternalPolicy`

职责：

- 判定是否属于 `Large`
- 决定容量增长策略
- 决定页粒度对齐策略
- 决定 `shrink_to_fit()` 的大对象行为
- 后续与 `ammalloc` 的大对象分配路径协同

当前阶段，`ExternalPolicy` 作为 **`BasicStringCore` 内部的策略层概念**存在，不要求在代码层单独实现为独立模板类型。

---

## 4. 对象布局定义

## 4.1 `Small`

`Small` 状态满足：

- 数据存放在对象本体内部
- 不进行外部存储分配
- `capacity()` 固定为 `SmallCapacity`
- `data()[size()] == CharT{}` 始终成立

## 4.2 `External`

`External` 状态满足：

- `data()` 指向对象外部缓冲区
- `size()` 为当前字符数
- `capacity()` 为当前外部容量（字符数）
- `data()[size()] == CharT{}` 始终成立
- 保持独占所有权语义
- copy 深拷贝，move 偷指针

---

## 5. GenericLayoutPolicy 设计

## 5.1 设计目标

`GenericLayoutPolicy<CharT>` 作为多 `CharT` 通用布局方案，要求：

- 支持 `char / char8_t / char16_t / char32_t / wchar_t`
- 保持 24B 对象头（64-bit）
- 不引入独立 discriminator
- 以 `CharT` 宽度感知的 probe slot 实现状态判别
- 作为 generic correctness baseline

## 5.2 基本存储结构

```cpp
template <typename CharT>
struct ExternalRep {
    CharT* data;
    size_t size;
    size_t capacity_with_tag;
};

template <typename CharT>
union Storage {
    CharT small[sizeof(ExternalRep<CharT>) / sizeof(CharT)];
    ExternalRep<CharT> external;
    std::byte raw[sizeof(ExternalRep<CharT>)];
};
```

在 64-bit 平台下：

- `sizeof(ExternalRep<CharT>) == 24`
- `sizeof(Storage<CharT>) == 24`

## 5.3 Small 容量定义

```cpp
SmallSlots    = 24 / sizeof(CharT)
MetaSlot      = SmallSlots - 1
SmallCapacity = SmallSlots - 1
```

其中：

- `small[0 .. MetaSlot-1]` 存放有效字符
- `small[MetaSlot]` 存放 metadata / probe slot

## 5.4 状态判别规则

`GenericLayoutPolicy` 使用 probe slot 判别状态：

- `Small`：probe slot 的值位于 `[0, SmallCapacity]`
- `External`：probe slot 的值等于 `ExternalTag`

其中：

```cpp
ExternalTag = SmallCapacity + 1
```

### Small 编码

`Small` 状态下：

```cpp
ProbeMeta = SmallCapacity - size
```

### External 编码

`External` 状态下：

```cpp
ProbeMeta == ExternalTag
```

### 判别逻辑

```cpp
ProbeMeta <= SmallCapacity   => Small
ProbeMeta == ExternalTag     => External
otherwise                    => Invalid
```

## 5.5 `capacity_with_tag` 编码

`capacity_with_tag` 同时承担两项职责：

1. 保存真实 capacity
2. 在 `External` 状态下让 probe slot 读出 `ExternalTag`

定义：

```cpp
ProbeBits   = sizeof(CharT) * 8
WordBits    = sizeof(size_t) * 8
PayloadBits = WordBits - ProbeBits
```

### little-endian

```cpp
Packed = (ExternalTag << PayloadBits) | Capacity
```

### big-endian

```cpp
Packed = (Capacity << ProbeBits) | ExternalTag
```

### 解码

- Tag 从 `capacity_with_tag` 中解出
- Capacity 从 payload 部分解出

### Capacity payload 上界说明

`GenericLayoutPolicy` 的最大 external capacity 受 `PayloadBits` 约束：

- `sizeof(CharT) == 1` 时，capacity payload 为 56 位
- `sizeof(CharT) == 2` 时，capacity payload 为 48 位
- `sizeof(CharT) == 4` 时，capacity payload 为 32 位

当前设计接受该上界，并以此作为 `GenericLayoutPolicy` 的正式容量边界约束。

---

## 6. CharLayoutPolicy 设计方向

`CharLayoutPolicy` 的正式定位如下：

- 仅服务于 `char`
- 与 `GenericLayoutPolicy` 接口保持同构
- 保持 `Small / External` 两态布局
- 不引入独立 discriminator
- 允许使用更激进的 `byte-level` 编码与 fast path
- 目标是逼近 `fbstring` 的 `char` 路径性能

`CharLayoutPolicy` 仅在以下条件满足后引入：

- generic core 稳定
- sanitizer 通过
- differential test 稳定
- benchmark 基线建立

---

## 7. External 语义层设计

## 7.1 语义层级

`External` 内部定义两种语义层级：

- `Normal`
- `Large`

该语义层级不体现在 layout state 中。

## 7.2 Large 定义

`Large` 定义为：

> 当 `External` 对象的分配字节数达到特定阈值后，该对象进入大对象资源管理策略区间。

### 判定方式

```cpp
AllocationBytes(capacity) = (capacity + 1) * sizeof(CharT)
IsLargeCapacity(capacity) = AllocationBytes(capacity) >= kLargeThresholdBytes
```

默认阈值：

```cpp
kLargeThresholdBytes = 4096
```

该值作为第一阶段的默认基线值使用；最终阈值应与 allocator 后端（尤其是后续 `ammalloc`）的大对象策略协同确定。

## 7.3 Large 的第一阶段实现方式

第一阶段 `Large` 仅作为 **计算型谓词** 存在：

- 不持久化存储 `Large` tag
- 不将 `Large` 写入 layout state
- 只在 `BasicStringCore` 中动态判断

---

## 8. Large 语义影响范围

第一阶段，`Large` 只影响以下行为：

### 8.1 容量增长策略

- `Normal External`：约 `1.5x` 增长
- `Large External`：约 `1.25x` 增长，并按页粒度向上取整

### 8.2 `shrink_to_fit()`

- 可降级到 `Small` 时优先降级
- 否则对 `Large` 对象执行更积极的收缩策略

### 8.3 对齐策略

进入 `Large` 区间后：

- 可启用页粒度容量规划
- 可减少容量冗余比例

### 8.4 所有权语义

`Large` 仍保持：

- 独占所有权
- copy 深拷贝
- move 偷指针
- 不引入 COW
- 不引入 refcount

---

## 9. allocator 设计

## 9.1 第一阶段

默认 allocator：

- `std::allocator`

要求：

- `BasicStringCore` 必须统一通过 `std::allocator_traits<Allocator>` 进行分配与释放
- 不允许写死 `new[] / delete[]`

## 9.2 第二阶段

引入：

```cpp
template<typename T>
class AMMallocAllocator;
```

切换到 `ammalloc` 后：

- layout 不变
- core 接口不变
- allocator 路径不变
- 仅替换后端分配器实现

---

## 10. ammalloc 与 External / Large 的关系

当 `ammalloc` 在大于特定字节阈值时自动跳过中间缓存路径并直接进行页分配时，`amstring` 采用如下分工：

### `amstring`

负责：

- 布局管理
- 容量规划
- `Large` 的增长/收缩/对齐策略

### `ammalloc`

负责：

- 小对象走常规缓存路径
- 大对象走页分配路径
- 正确释放到对应后端

因此，在 `ammalloc` 接入后：

- layout 仍保持 `Small / External`
- `Large` 仍不需要成为第三种布局状态
- `Large` 的后端分配语义由 `ammalloc` 自动承接

---

## 11. BasicStringCore 设计职责

`BasicStringCore` 正式职责如下：

### 11.1 生命周期
- 构造
- 析构
- copy / move
- assign

### 11.2 allocator 交互
- allocate / deallocate
- 支持未来 `AMMallocAllocator`

### 11.3 容器操作
- `reserve`
- `resize`
- `clear`
- `append`
- `assign`
- `shrink_to_fit`
- 后续的 `insert / erase / replace`

### 11.4 traits 交互
- `Traits::copy`
- `Traits::move`
- `Traits::compare`
- `Traits::length`

### 11.5 External 语义策略
- `IsLargeCapacity`
- `NextCapacity`
- 大对象 shrink 策略
- 页粒度对齐逻辑

### 11.6 layout 调用
- `data()`
- `size()`
- `capacity()`
- `SetSmallSize`
- `SetExternalSize`
- `InitSmall`
- `InitExternal`
- `CheckInvariants`

---

## 12. 设计不变量

## 12.1 全局不变量

任意对象都必须满足：

1. `data() != nullptr`
2. `data()[size()] == CharT{}`
3. `size() <= capacity()`
4. `begin() == data()`
5. `end() == data() + size()`
6. moved-from object 为合法 empty string

## 12.2 Small 不变量

当状态为 `Small` 时：

1. probe slot 的值位于 `[0, SmallCapacity]`
2. `size()` 由 small metadata 正确解码
3. `capacity() == SmallCapacity`
4. `data()` 指向对象内部存储
5. `data()[size()] == CharT{}`

## 12.3 External 不变量

当状态为 `External` 时：

1. probe slot 的值等于 `ExternalTag`
2. `external.data != nullptr`
3. `external.size <= UnpackCapacity(external.capacity_with_tag)`
4. `external.data[external.size] == CharT{}`
5. capacity 按字符数计，而不是按字节数计

---

## 13. 模板与模块组织

### 13.1 Layout policy

```cpp
template<typename CharT>
struct GenericLayoutPolicy;

struct CharLayoutPolicy;
```

### 13.2 Layout selector

```cpp
template<typename CharT>
struct DefaultLayoutPolicy {
    using type = GenericLayoutPolicy<CharT>;
};

// Introduced only after CharLayoutPolicy is ready.
// Phase 1 keeps char on GenericLayoutPolicy<char>.
template<>
struct DefaultLayoutPolicy<char> {
    using type = CharLayoutPolicy;
};
```

### 13.3 Core

```cpp
template<typename CharT, typename Traits, typename Allocator, typename LayoutPolicy>
class BasicStringCore;
```

### 13.4 Public string

```cpp
template<typename CharT, typename Traits = std::char_traits<CharT>, typename Allocator = std::allocator<CharT>>
class BasicString;
```

---

## 14. 实现顺序

### 阶段 1
- 完成 `GenericLayoutPolicy<CharT>`
- 只实现 `Small / External` 两态
- 跑多 `CharT` 的 layout 测试

### 阶段 2
- 完成 `BasicStringCore`
- 默认使用 `std::allocator`
- 跑生命周期、reserve、resize、append、assign 测试

### 阶段 3
- 封装 `BasicString<char>`
- 建立与 `std::string` 的 differential test
- 跑 ASan / UBSan / LSan

### 阶段 4
- 放开 `char8_t / char16_t / char32_t / wchar_t`
- 跑 generic 多 `CharT` 测试矩阵

### 阶段 5
- 建立 benchmark 基线
- 评估 `char` 热路径

### 阶段 6
- 引入 `CharLayoutPolicy`
- 保持 public API 与 core 接口不变

### 阶段 7
- 引入 `AMMallocAllocator`
- 切换 allocator backend
- 验证 large 路径与页分配路径协同效果

---

## 15. 当前阶段禁止事项

当前阶段不允许：

- 在 layout 中引入 `medium / large`
- 在对象头中增加独立 discriminator
- 在 generic 路径中预留 direct-map backend tag
- 在 `Large` 中引入 COW / refcount
- 在 benchmark 基线建立前做复杂 SIMD
- 在 `char` 路径成熟前，把 `char` 与多 `CharT` 混写成一套极限布局

---

## 16. 正式设计结论

`amstring` 的正式存储架构采用两段式布局：`Small / External`。  
其中，`Small` 表示对象内部内联存储，`External` 表示对象外部存储。  
针对大对象场景，不通过引入第三种布局状态建模，而是在 `External` 语义层引入 `Large` policy，对容量增长、页粒度对齐、显式收缩以及后续可能的后端分配策略进行差异化管理。

因此：

1. `GenericLayoutPolicy<CharT>` 作为多 `CharT` correctness baseline。
2. `CharLayoutPolicy` 作为 `char` 的后续高性能特化路径。
3. `Large` 不是第三种布局状态，而是 `External` 的子语义。
4. 第一阶段默认使用 `std::allocator`，后续通过 allocator adapter 切换到 `ammalloc`。

---

## 17. 一句话总结

`amstring` 的正式长期方向是 `Small / External` 两段式布局，并在 `External` 内通过策略分层实现 `Large` 语义，同时通过双 layout policy 兼顾多 `CharT` 正确性与 `char` 路径性能优化。
