# amstring Policy-Based 架构设计文档

## 1. 文档目的

本文档定义 `amstring` 的 policy-based 总体架构方案，作为后续 `BasicString`、`BasicStringCore`、`LayoutPolicy`、allocator 后端接入以及测试体系建设的统一设计基线。

本文档用于回答以下问题：

1. `amstring` 中有哪些 policy 维度。
2. 哪些 policy 是正式模板参数。
3. 哪些 policy 只是第一阶段的内部策略概念。
4. `GenericLayoutPolicy<CharT>`、`CharLayoutPolicy`、`Traits`、`Allocator`、`ExternalPolicy` 之间如何协作。
5. 如何防止过度 policy 化。
6. 第一阶段应采用怎样的组合方案。
7. 后续如何演进到 `CharLayoutPolicy` 和 `AMMallocAllocator`。

---

## 2. 总体设计原则

`amstring` 采用 policy-based 设计，但不将所有可变点都抽象为模板 policy。

正式原则如下：

> 只有长期稳定、编译期可选择、并且对对象布局、字符语义或分配模型有根本影响的维度，才设计为模板 policy。  
> 短期策略、尚未稳定的优化策略、或只影响局部算法的策略，第一阶段保留为 `BasicStringCore` 内部实现细节。

因此，第一阶段正式模板 policy 只有：

1. `Traits`
2. `Allocator`
3. `LayoutPolicy`

其中：

- `Traits` 沿用标准库 `std::char_traits` 风格
- `Allocator` 沿用标准库 allocator 模型
- `LayoutPolicy` 是 `amstring` 自定义的布局策略维度

`ExternalPolicy`、`GrowthPolicy`、`LargePolicy` 等第一阶段不作为独立模板参数存在。

---

## 3. 开发与验证方式

`amstring` 第一阶段采用 **TDD（Test-Driven Development，测试驱动开发）** 作为默认开发方式。

基本约定如下：

1. 先写测试，再写实现。
2. 先写最小失败测试，用于精确定义当前要实现的行为、状态约束或不变量。
3. 仅编写使当前测试通过所需的最小实现，不在同一步骤中引入额外重构或无关优化。
4. 在当前测试通过后，再进行小步重构，并保持测试持续通过。
5. 新增布局规则、状态迁移规则、异常安全约束、moved-from 语义或容量策略时，必须先补对应测试，再修改实现。

对 `amstring` 而言，TDD 不只是接口级测试，还包括以下几类验证：

- layout policy 的状态判别与编码/解码测试
- `BasicStringCore` 的状态迁移测试
- moved-from 对象恢复测试
- 异常安全测试
- invariant 检查与差分测试
- `BasicString` 公共语义测试

这意味着，后续 `GenericLayoutPolicy<CharT>`、`BasicStringCore`、`BasicString` 的实现，应以“先测试定义行为，再实现满足行为”为统一工作流，而不是先写完整实现再补测试。

---

## 4. 总体结构

所有 `amstring` 核心类型默认位于 `aethermind` 命名空间，包括：

- `BasicString`
- `BasicStringCore`
- `GenericLayoutPolicy`
- `CharLayoutPolicy`
- `DefaultLayoutPolicy`

```text
BasicString<CharT, Traits, Allocator>
  └── BasicStringCore<CharT, Traits, Allocator, LayoutPolicy>
       ├── LayoutPolicy
       │    ├── GenericLayoutPolicy<CharT>
       │    └── CharLayoutPolicy
       │
       ├── Traits
       │    └── std::char_traits-like operations
       │
       ├── Allocator
       │    ├── std::allocator
       │    └── AMMallocAllocator
       │
       └── Internal strategies
            ├── ExternalPolicy concept
            ├── Normal / Large external semantics
            ├── Growth strategy
            └── shrink strategy
```

---

## 4. Public 层模板结构

## 4.1 `BasicString`

公共主模板正式命名为：

```cpp
template<
    typename CharT,
    typename Traits = std::char_traits<CharT>,
    typename Allocator = std::allocator<CharT>
>
class BasicString;
```

模板参数含义：

- `CharT`：字符类型
- `Traits`：字符算法策略
- `Allocator`：外部存储分配策略

`BasicString` 不直接暴露 `LayoutPolicy` 模板参数。

## 4.2 为什么 public 层不暴露 `LayoutPolicy`

`LayoutPolicy` 是底层实现策略，不属于第一阶段 public API。

理由：

1. 保持 public API 接近标准库字符串。
2. 避免用户直接依赖内部布局实现。
3. 允许后续透明地将 `char` 从 `GenericLayoutPolicy<char>` 切换到 `CharLayoutPolicy`。
4. 允许后续调整 internal policy 而不破坏用户代码。

因此，`LayoutPolicy` 由内部 selector 决定。

---

## 5. Core 层模板结构

`BasicStringCore` 定义为：

```cpp
template<
    typename CharT,
    typename Traits,
    typename Allocator,
    typename LayoutPolicy
>
class BasicStringCore;
```

其中：

- `CharT` 由 `BasicString` 传入
- `Traits` 由 `BasicString` 传入
- `Allocator` 由 `BasicString` 传入
- `LayoutPolicy` 由内部 `DefaultLayoutPolicy<CharT>` 选择

`BasicStringCore` 是 owning string storage manager，负责：

- 生命周期
- 分配与释放
- 核心修改操作
- 异常安全
- `Small / External` 状态切换
- `External` 语义层策略
- 调用 `LayoutPolicy`
- 调用 `Traits`
- 调用 `Allocator`

---

## 6. Policy 分类表

| Policy / 策略 | 第一阶段形态 | 是否模板参数 | 所属层级 | 职责 |
|---|---|---:|---|---|
| `Traits` | 标准库风格 policy | 是 | public/core | 字符复制、移动、比较、长度计算 |
| `Allocator` | 标准库 allocator | 是 | public/core | external buffer 分配与释放 |
| `LayoutPolicy` | 自定义布局 policy | 是，core 层 | core/layout | `Small / External` 布局编码 |
| `GenericLayoutPolicy<CharT>` | layout policy 实现 | 由 selector 选择 | layout | 多 `CharT` correctness baseline |
| `CharLayoutPolicy` | layout policy 实现 | 由 selector 选择 | layout | `char` 专用高性能布局路径 |
| `DefaultLayoutPolicy<CharT>` | selector | 否，内部类型选择 | core/layout | 为不同 `CharT` 选择布局策略 |
| `ExternalPolicy` | core 内部策略概念 | 否 | core | `Normal / Large` external 语义 |
| `GrowthPolicy` | internal helper | 否 | core | capacity 增长策略 |
| `LargePolicy` | internal helper/concept | 否 | core | 大对象容量规划与 shrink 策略 |
| `AMMallocAllocator` | allocator 实现 | 是，通过 Allocator | allocator | 后续接入 `ammalloc` |

---

## 7. LayoutPolicy 设计

命名约定总览：

- STL 风格接口使用小写，例如 `data()` / `size()` / `capacity()` / `category()`
- 内部 helper 使用 CamelCase，例如 `InitEmpty` / `SetSmallSize`
- 结构体与 union 成员变量使用 snake_case，例如 `data` / `size` / `capacity_with_tag`
- 类成员变量使用 snake_case_，例如 `storage_` / `allocator_`

详细规则见 `GenericLayoutPolicy_design.md` 与 `BasicStringCore_design.md`。

## 7.1 定位

`LayoutPolicy` 是 `amstring` 自定义的正式布局 policy。

它负责：

- 定义对象布局
- 编码/解码 `Small / External`
- 提供 `data() / size() / capacity() / category()`
- 提供 `InitEmpty / InitSmall / InitExternal`
- 提供 `SetSmallSize / SetExternalSize / SetExternalCapacity`
- 提供 `CheckInvariants`

它不负责：

- allocator 分配与释放
- `Traits` 字符操作
- 容器级修改算法
- `External` 的 `Normal / Large` 策略
- 异常安全

## 7.2 LayoutPolicy 接口约束

所有 layout policy 必须提供统一接口：

```cpp
static bool is_small(const Storage&) noexcept;
static bool is_external(const Storage&) noexcept;
static Category category(const Storage&) noexcept;

static const CharT* data(const Storage&) noexcept;
static CharT* data(Storage&) noexcept;

static SizeType size(const Storage&) noexcept;
static SizeType capacity(const Storage&) noexcept;
static constexpr SizeType max_external_capacity() noexcept;

static void InitEmpty(Storage&) noexcept;
static void InitSmall(Storage&, const CharT* src, SizeType size) noexcept;
static void InitExternal(Storage&, CharT* ptr, SizeType size, SizeType capacity) noexcept;

static void SetSmallSize(Storage&, SizeType size) noexcept;
static void SetExternalSize(Storage&, SizeType size) noexcept;
static void SetExternalCapacity(Storage&, SizeType capacity) noexcept;

static void CheckInvariants(const Storage&) noexcept;
```

## 7.3 LayoutPolicy 的正式实现

第一阶段正式实现：

```cpp
template<typename CharT>
struct GenericLayoutPolicy;
```

后续高性能特化：

```cpp
struct CharLayoutPolicy;
```

---

## 8. GenericLayoutPolicy

`GenericLayoutPolicy<CharT>` 是多 `CharT` 路径的正式布局基线。

设计结论：

1. 只依赖 `CharT`
2. 保持 24B 对象头
3. 使用 `Small / External` 两态布局
4. 不引入独立 discriminator
5. 通过 `CharT` 宽度感知的 probe slot 判别状态
6. 通过 `capacity_with_tag` 编码 external capacity 与 external tag
7. 作为 correctness baseline

支持类型：

- `char`
- `char8_t`
- `char16_t`
- `char32_t`
- `wchar_t`

---

## 9. CharLayoutPolicy

`CharLayoutPolicy` 是 `char` 的专用 layout policy。

设计定位：

1. 仅服务于 `char`
2. 与 `GenericLayoutPolicy` 接口同构
3. 保持 `Small / External` 两态布局
4. 不引入独立 discriminator
5. 允许使用更激进的 byte-level 编码与 fast path
6. 目标是逼近 `fbstring` 的 `char` 路径性能

引入前提（已满足并据此启动 M7 首轮实现）：

- generic core 稳定
- sanitizer 通过
- differential test 稳定
- benchmark 基线建立

当前状态：

- `CharLayoutPolicy` 已在 M7 首轮实现中引入
- `DefaultLayoutPolicy<char>` 已切换到 `CharLayoutPolicy`
- 非 `char` 类型仍保持 `GenericLayoutPolicy<CharT>`
- 首轮验证见 `docs/tests/amstring_charlayout_m7_validation_20260429.md`

---

## 10. DefaultLayoutPolicy 选择机制

`DefaultLayoutPolicy<CharT>` 是内部 selector。

第一阶段推荐：

```cpp
template<typename CharT>
struct DefaultLayoutPolicy {
    using type = GenericLayoutPolicy<CharT>;
};
```

当前 selector 形式：

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

外部用户不直接感知该切换。

---

## 11. Traits Policy

`Traits` 沿用标准库 `std::char_traits` 风格。

默认形式：

```cpp
typename Traits = std::char_traits<CharT>
```

`BasicStringCore` 通过 `Traits` 执行：

- `copy`
- `move`
- `compare`
- `length`
- 后续 `find`

第一阶段不重新设计新的字符算法 policy。

要求：

- core 层不得绕开 `Traits` 自定义字符语义
- `GenericLayoutPolicy` 不依赖 `Traits`
- `LayoutPolicy` 不执行字符语义算法

---

## 12. Allocator Policy

`Allocator` 沿用标准库 allocator 模型。

默认形式：

```cpp
typename Allocator = std::allocator<CharT>
```

`BasicStringCore` 必须通过：

```cpp
std::allocator_traits<Allocator>
```

执行分配与释放。

第一阶段默认：

- `std::allocator`

第二阶段引入：

```cpp
template<typename T>
class AMMallocAllocator;
```

要求：

- `BasicStringCore` 不写死 `new[] / delete[]`
- external buffer 分配数量为 `capacity + 1`
- 额外 1 个字符用于 terminator
- allocator 后端切换不影响 layout
- allocator 后端切换不影响 public API

---

## 13. ExternalPolicy 概念

`ExternalPolicy` 第一阶段不是独立模板参数，也不是独立 policy 类型。

它是 `BasicStringCore` 内部的策略层概念。

职责：

- 判定是否属于 `Large`
- 选择 `NextCapacity`
- 处理页粒度对齐
- 决定 `shrink_to_fit()` 对 large external 的行为
- 后续与 `ammalloc` 的大对象路径协同

核心函数可以表现为 `BasicStringCore` 内部 helper：

```cpp
bool IsLargeCapacity(SizeType capacity) noexcept;
SizeType NextCapacity(SizeType min_capacity) const noexcept;
SizeType RoundUpCapacityToPage(SizeType capacity) const noexcept;
```

---

## 14. GrowthPolicy 与 LargePolicy

第一阶段不单独引入：

- `GrowthPolicy`
- `LargePolicy`

它们作为 `ExternalPolicy` 概念下的内部 helper 存在。

默认策略：

- `Normal External`：约 `1.5x` 增长
- `Large External`：约 `1.25x` 增长，并按页粒度向上取整

若后续 benchmark 证明容量增长策略需要可配置，再考虑单独抽象 `GrowthPolicy`。

---

## 15. Large 语义与 ammalloc

`Large` 是 `External` 的子语义，不是 layout state。

第一阶段：

- `Large` 作为计算型谓词存在
- 不持久化存储 tag
- 不进入 `LayoutPolicy`

定义：

```cpp
AllocationBytes(capacity) = (capacity + 1) * sizeof(CharT)
IsLargeCapacity(capacity) = AllocationBytes(capacity) >= kLargeThresholdBytes
```

默认：

```cpp
kLargeThresholdBytes = 4096
```

后续接入 `ammalloc` 后：

- `amstring` 保留容量规划、增长、收缩和对齐策略
- `ammalloc` 承接大对象的后端分配路径
- layout 仍保持 `Small / External`
- 不需要把 `Large` 升级为第三种 layout state

---

## 16. 第一阶段组合方案

第一阶段推荐组合：

```cpp
using Layout = GenericLayoutPolicy<CharT>;

template<
    typename CharT,
    typename Traits = std::char_traits<CharT>,
    typename Allocator = std::allocator<CharT>
>
class BasicString;
```

内部：

```cpp
using LayoutPolicy = typename DefaultLayoutPolicy<CharT>::type;

BasicStringCore<CharT, Traits, Allocator, LayoutPolicy> core_;
```

其中第一阶段：

```cpp
DefaultLayoutPolicy<CharT>::type == GenericLayoutPolicy<CharT>
```

即：

- `char` 也先走 generic layout
- 多 `CharT` 统一建立 correctness baseline
- `CharLayoutPolicy` 后续再接入

---

## 17. 后续演进路线

### 阶段 1：Generic correctness baseline

- 实现 `GenericLayoutPolicy<CharT>`
- 实现 `BasicStringCore`
- 实现 `BasicString<char>`
- 建立 sanitizer / differential test

### 阶段 2：多 CharT 放开

- `char8_t`
- `char16_t`
- `char32_t`
- `wchar_t`

全部走 `GenericLayoutPolicy<CharT>`。

### 阶段 3：benchmark baseline

对比：

- `std::string`
- `folly::fbstring`
- `BasicString<char>`

### 阶段 4：CharLayoutPolicy

- 引入 `CharLayoutPolicy`
- 通过 `DefaultLayoutPolicy<char>` 切换
- 保持 `BasicString` public API 不变
- 保持 `BasicStringCore` 接口不变

### 阶段 5：AMMallocAllocator

- 引入 `AMMallocAllocator<T>`
- 默认 allocator 可继续为 `std::allocator`
- 通过类型别名或配置切换到 `AMMallocAllocator`
- 验证 large external 与 ammalloc 页分配路径协同效果

---

## 18. 当前阶段不做成 policy 的内容

当前阶段不单独抽象为模板 policy 的内容包括：

- `ExternalPolicy`
- `GrowthPolicy`
- `LargePolicy`
- `SmallPolicy`
- `CopyPolicy`
- `FindPolicy`
- `ComparePolicy`
- `SIMDPolicy`
- `PagePolicy`
- `BackendPolicy`

这些策略在第一阶段保留为：

- `BasicStringCore` 内部 helper
- `Traits` 行为
- `Allocator` 行为
- 后续 benchmark 驱动的优化点

---

## 19. 设计不变量

Policy-based 总体设计必须满足：

1. `BasicString` public API 不暴露内部 layout policy
2. `BasicStringCore` 是唯一 owning storage manager
3. `LayoutPolicy` 只负责布局，不负责 allocator 与 traits
4. `Traits` 只负责字符语义
5. `Allocator` 只负责分配与释放
6. `ExternalPolicy` 第一阶段不作为独立模板参数
7. `Large` 不作为 layout state
8. `CharLayoutPolicy` 必须与 `GenericLayoutPolicy` 接口同构
9. allocator 后端切换不得破坏 layout 不变量
10. public 层接口稳定性优先于内部 policy 调整

---

## 20. 测试策略

## 20.1 LayoutPolicy 测试

独立测试：

- `GenericLayoutPolicy<CharT>`
- 后续 `CharLayoutPolicy`

覆盖：

- `Small / External`
- `data / size / capacity`
- tag 编码
- invariant

## 20.2 BasicStringCore 测试

覆盖：

- 生命周期
- 分配释放
- 状态切换
- append / assign / resize / reserve
- self-overlap
- shrink_to_fit
- allocator 行为

## 20.3 BasicString 测试

覆盖：

- public API
- 与 `std::basic_string` 的 differential test
- 多 `CharT`
- 后续性能 benchmark

## 20.4 Policy 组合测试

必须覆盖：

- `GenericLayoutPolicy<char>`
- `GenericLayoutPolicy<char8_t>`
- `GenericLayoutPolicy<char16_t>`
- `GenericLayoutPolicy<char32_t>`
- `GenericLayoutPolicy<wchar_t>`
- 后续 `CharLayoutPolicy + char`
- 后续 `AMMallocAllocator + char`

---

## 21. 当前阶段禁止事项

当前阶段不允许：

- 在 public API 暴露 `LayoutPolicy`
- 将 `ExternalPolicy` 设计为独立模板参数
- 将 `GrowthPolicy` 设计为独立模板参数
- 在 layout 中加入 `Medium / Large`
- 在对象头中增加独立 discriminator
- 在 `Large` 中引入 COW / refcount
- 在 benchmark 之前引入 SIMD / safe over-read
- 为每个局部优化点提前设计独立 policy

---

## 22. 正式设计结论

`amstring` 采用受控的 policy-based 架构。

正式模板 policy 包括：

1. `Traits`
2. `Allocator`
3. `LayoutPolicy`

第一阶段内部策略包括：

1. `ExternalPolicy`
2. `GrowthPolicy`
3. `Large` external 策略

其中：

- `Traits` 与 `Allocator` 沿用标准库模型
- `LayoutPolicy` 是 `amstring` 自定义的核心布局策略
- `GenericLayoutPolicy<CharT>` 是多 `CharT` correctness baseline
- `CharLayoutPolicy` 是后续 `char` 高性能特化路径
- `ExternalPolicy` 第一阶段不作为独立模板参数
- `Large` 不作为 layout state

---

## 23. 一句话总结

`amstring` 的 policy-based 架构核心是：**Public 层只暴露 `CharT / Traits / Allocator`，Core 层内部通过 `LayoutPolicy` 选择布局实现，并将尚未稳定的 External/Growth/Large 策略保留为 `BasicStringCore` 内部概念，从而在避免过度泛化的前提下支持多 `CharT` 正确性、`char` 高性能特化和后续 `ammalloc` 后端接入。**
