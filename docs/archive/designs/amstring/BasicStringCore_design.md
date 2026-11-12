# BasicStringCore 设计文档

## 1. 文档目的

本文档定义 `BasicStringCore` 的正式设计方案，作为 `amstring` 存储架构中“布局之上、公共接口之下”的核心实现层设计基线。

本文档用于指导以下工作：

- `BasicStringCore` 的实现
- `BasicStringCore` 与 `GenericLayoutPolicy` / `CharLayoutPolicy` 的接口对接
- `BasicStringCore` 与 `Traits` / `Allocator` 的职责边界
- `reserve / resize / clear / append / assign / shrink_to_fit` 等核心操作的实现
- 异常安全、状态流转、不变量与测试策略

---

## 2. 开发方式约定

`BasicStringCore` 的实现阶段采用 TDD 方式推进。

具体要求：

- 先为目标行为编写失败测试，再编写实现
- 每次只推进一个最小行为单元，例如：
  - empty 初始化
  - `Small -> External` 状态切换
  - `reserve` 扩容
  - `resize` 的补零与截断
  - moved-from 对象恢复
  - `shrink_to_fit()` 的状态回退
- 每个行为单元完成后，先保证测试通过，再进行小步重构
- 涉及状态迁移、容量变化、异常安全或不变量的修改，必须同步补测试

因此，`BasicStringCore` 的实现顺序应由测试用例驱动，而不是由接口清单顺序驱动。

---

## 3. 设计定位

`BasicStringCore` 的定位如下：

1. 作为 `BasicString` 的内部实现核心。
2. 作为 allocator-aware 的 owning string storage manager。
3. 负责对象生命周期、内存分配、核心修改操作与异常安全。
4. 通过 `LayoutPolicy` 读写底层布局。
5. 通过 `Traits` 执行字符级复制、移动、比较与长度计算。
6. 在 `External` 状态下承担 `Normal / Large` 语义层策略管理。

`BasicStringCore` 不直接向最终用户暴露，不承担完整公共 API 的用户语义包装职责。

---

## 3. 职责与边界

## 3.1 负责的内容

`BasicStringCore` 负责：

- 对象构造、析构、copy、move、assign
- allocator 分配与释放
- `Small / External` 状态切换
- `reserve`
- `resize`
- `clear`
- `append`
- `assign`
- `push_back / pop_back`
- `shrink_to_fit`
- moved-from 对象恢复为合法 empty string
- `External` 语义层的 `Normal / Large` 策略判断
- 异常安全保证
- 调用 `LayoutPolicy` 维护底层布局一致性
- 调用 `Traits` 完成字符级操作

## 3.2 不负责的内容

`BasicStringCore` 不负责：

- 公共接口命名与用户层语义包装
- `operator+`
- 比较运算符重载
- `find / rfind / starts_with / ends_with / contains`
- `std::hash`
- stream / formatter
- 最终用户可见 API 兼容性

这些职责由 `BasicString` 或更高层模块承担。

---

## 4. 模板参数与命名规则

## 4.1 模板参数

`BasicStringCore` 定义为：

```cpp
template<typename CharT, typename Traits, typename Allocator, typename LayoutPolicy>
class BasicStringCore;
```

其中：

- `CharT`：字符类型
- `Traits`：字符算法 traits
- `Allocator`：分配器类型
- `LayoutPolicy`：布局策略类型

## 4.2 前置约束

要求：

- `LayoutPolicy::ValueType == CharT`
- `Allocator::value_type == CharT` 或可通过 allocator_traits 正常适配
- `Traits` 满足 `std::char_traits` 风格接口要求

## 4.3 命名规则

命名规则如下：

- STL 风格接口使用小写：
  - `data()`
  - `size()`
  - `capacity()`
  - `empty()`
  - `reserve()`
  - `resize()`
  - `clear()`
  - `append()`
  - `assign()`
  - `shrink_to_fit()`

- 内部 helper 使用 CamelCase：
  - `ConstructFromPointerAndSize`
  - `CopyFrom`
  - `MoveFrom`
  - `AssignFrom`
  - `Destroy`
  - `AllocateChars`
  - `DeallocateChars`
  - `EnsureCapacityForSize`
  - `ReallocateAtLeast`
  - `ReallocateExact`
  - `NextCapacity`
  - `IsLargeCapacity`
  - `RoundUpCapacityToPage`
  - `CheckInvariants`

- 类成员变量使用 snake_case_：
  - `storage_`
  - `allocator_`

---

## 5. 基本结构

## 5.1 类型别名

`BasicStringCore` 建议显式提供以下类型别名：

- `ValueType`
- `SizeType`
- `AllocatorType`
- `LayoutPolicyType`
- `Storage`
- `Pointer`
- `ConstPointer`
- `AllocatorTraits`

推荐关系为：

- `ValueType = CharT`
- `Storage` 由 `LayoutPolicy` 提供
- `AllocatorTraits = std::allocator_traits<Allocator>`

这些类型别名的目的，是让实现层、测试层与文档层对 core 的内部约定保持一致，避免在实现阶段重复拼接模板参数表达。

## 5.2 核心成员

`BasicStringCore` 的核心成员包括：

1. `typename LayoutPolicy::Storage storage_`
2. `Allocator allocator_`

其中：

- `Storage` 由 `LayoutPolicy` 提供
- `storage_` 保存对象布局本体
- `allocator_` 负责外部存储的分配与释放

推荐形式：

```cpp
typename LayoutPolicy::Storage storage_;
[[no_unique_address]] Allocator allocator_;
```

其中：

- `storage_` 保存布局本体
- `allocator_` 保存 allocator 实例
- 类成员变量统一使用 `snake_case_`

---

## 6. 与各层的关系

## 6.1 与 `BasicString`

`BasicString` 负责：

- 公共 API 封装
- 用户可见类型与语义
- 对 `BasicStringCore` 的薄包装

`BasicStringCore` 负责：

- 真正的状态管理与存储操作

## 6.2 与 `LayoutPolicy`

`LayoutPolicy` 负责：

- 底层布局定义
- `Small / External` 编码解码
- `data() / size() / capacity() / category()`
- 状态原语：`InitEmpty / InitSmall / InitExternal`
- 状态内更新原语：`SetSmallSize / SetExternalSize / SetExternalCapacity`
- 布局不变量检查

`BasicStringCore` 不重复实现这些逻辑，只通过 `LayoutPolicy` 访问与修改布局。

## 6.3 与 `Traits`

`Traits` 负责：

- `copy`
- `move`
- `compare`
- `length`

`BasicStringCore` 不自己发明字符级语义规则，而统一通过 `Traits` 调用。

## 6.4 与 `Allocator`

`Allocator` 负责：

- 分配外部缓冲区
- 释放外部缓冲区

`BasicStringCore` 必须统一通过：

```cpp
std::allocator_traits<Allocator>
```

进行分配与释放。

---

## 7. 状态模型

`BasicStringCore` 的对象布局状态只分为：

- `Small`
- `External`

由 `LayoutPolicy` 判定。

## 7.1 `Small`

- 数据位于对象内部
- `capacity()` 固定为 `LayoutPolicy` 提供的 `SmallCapacity` 上界
- 不进行外部存储分配

## 7.2 `External`

- 数据位于对象外部
- `storage_` 中保存 `data / size / capacity_with_tag`
- 对象保持独占所有权

---

## 8. External 语义层

`External` 内部定义两种策略语义：

- `Normal`
- `Large`

该语义层不体现在 layout state 中，而由 `BasicStringCore` 动态计算。

当前阶段，`ExternalPolicy` 只作为 `BasicStringCore` 内部的策略层概念存在，不要求在代码层单独实现为独立模板参数或独立 policy 类型。

## 8.1 `IsLargeCapacity`

推荐定义：

```cpp
AllocationBytes(capacity) = (capacity + 1) * sizeof(CharT)
IsLargeCapacity(capacity) = AllocationBytes(capacity) >= kLargeThresholdBytes
```

默认阈值：

```cpp
kLargeThresholdBytes = 4096
```

该阈值为第一阶段默认值；最终可与 allocator 后端协同调整。

## 8.2 语义作用范围

`Large` 只影响：

- `NextCapacity`
- 页粒度对齐策略
- `shrink_to_fit()`

`Large` 不影响：

- 所有权语义
- copy / move 基本语义
- layout state

---

## 9. 构造与析构设计

## 9.1 默认构造

默认构造要求：

- 调用 `LayoutPolicy::InitEmpty`
- 构造合法 `Small` 空对象
- `data()[0] == CharT{}`
- `size() == 0`

## 9.2 从指针与长度构造

接口：

```cpp
BasicStringCore(const CharT* src, SizeType size, const Allocator& alloc = Allocator())
```

规则：

- 若 `size <= LayoutPolicy` 提供的 `SmallCapacity` 上界，构造为 `Small`
- 否则分配 external buffer，构造为 `External`

## 9.3 从 string_view 构造

规则与指针+长度构造一致。

## 9.4 拷贝构造

规则：

- `Small` -> `Small`：直接构造 small
- `External` -> 新 external：深拷贝
- allocator 使用 `select_on_container_copy_construction`

## 9.5 移动构造

规则：

- `Small` -> 复制 small 数据，再将源对象恢复为空
- `External` -> 直接接管 external buffer，再将源对象恢复为空

## 9.6 析构

规则：

- `Small`：无外部释放
- `External`：释放 external buffer
- 析构后不要求保留有效可访问状态

---

## 10. 赋值设计

## 10.1 拷贝赋值

建议实现策略：

- 使用临时对象保证 strong exception guarantee
- 完成后 `swap`

## 10.2 移动赋值

规则：

- 若 allocator 可传播，则优先窃取资源
- 若 allocator 不可传播但相等，则窃取资源
- 若 allocator 不可传播且不等，则退化为深拷贝

第一阶段最低要求：

- copy construction 使用 `std::allocator_traits<Allocator>::select_on_container_copy_construction`
- move assignment 至少处理 `propagate_on_container_move_assignment`、allocator 相等、allocator 不等三类情况
- allocator 不等且不可传播时，不得直接窃取 external buffer，必须执行深拷贝或等价安全路径

## 10.3 assign 系列

`assign` 统一采用：

- 构造临时对象
- `swap`

作为第一阶段基线实现策略，优先保证正确性与异常安全。

对于 `assign(const CharT* src, SizeType count)`，若 `src` 指向当前对象内部存储区间，临时对象策略仍应保持正确：在临时对象构造完成前，不得修改当前对象；临时对象构造成功后再执行 `swap`。因此，第一阶段的 `assign` 策略应天然支持 self subrange assign。

---

## 11. 核心接口

## 11.1 只读接口

`BasicStringCore` 应提供：

```cpp
const CharT* data() const noexcept;
CharT* data() noexcept;
const CharT* c_str() const noexcept;

SizeType size() const noexcept;
SizeType capacity() const noexcept;
bool empty() const noexcept;
```

## 11.2 基础修改接口

```cpp
void clear() noexcept;
void reserve(SizeType new_cap);
void resize(SizeType new_size, CharT fill = CharT{});
void append(const CharT* src, SizeType count);
void append(std::basic_string_view<CharT, Traits> sv);
void append(SizeType count, CharT ch);
void assign(const CharT* src, SizeType count);
void assign(std::basic_string_view<CharT, Traits> sv);
void assign(SizeType count, CharT ch);
void push_back(CharT ch);
void pop_back() noexcept;
void shrink_to_fit();
void swap(BasicStringCore& other) noexcept(/* depends on allocator swap semantics */);
```

其中，`swap` 的 `noexcept` 条件取决于 allocator 的 swap 传播语义与 noexcept 保证；文档不保留未解释的占位写法。

第一阶段 `swap` 的 allocator 处理规则如下：

- 若 `propagate_on_container_swap` 为 true，则允许同时交换 allocator 与 storage
- 若 `propagate_on_container_swap` 为 false 且 allocator 相等，则允许交换 storage
- 若 `propagate_on_container_swap` 为 false 且 allocator 不等，则不得执行会导致错误释放的裸 storage 交换；该场景应采用保守安全路径，或在第一阶段明确禁止

---

## 12. 容量规划

## 12.1 `EnsureCapacityForSize`

职责：

- 确保对象容量至少满足 `required_size`
- 若当前容量足够，则不动作
- 若不足，则调用重分配逻辑

## 12.2 `NextCapacity`

默认策略：

- `Normal External`：约 `1.5x`
- `Large External`：约 `1.25x`

并遵循：

- 至少满足 `min_capacity`
- 不超过 `LayoutPolicy::max_external_capacity()`

因此，`LayoutPolicy` 必须显式提供 `max_external_capacity()`，供 `BasicStringCore` 在容量规划阶段执行上界约束。

## 12.3 页粒度对齐

在 `Large` 区间内可启用页粒度对齐：

```cpp
RoundUpCapacityToPage(capacity)
```

该对齐策略由 `BasicStringCore` 决定，不由 `LayoutPolicy` 决定。

---

## 13. 重分配策略

## 13.1 `ReallocateExact`

职责：

- 精确分配指定 capacity 的 external buffer
- 拷贝现有内容
- 写入 terminator
- 更新布局为合法 `External`

## 13.2 `ReallocateAtLeast`

职责：

- 根据 `NextCapacity(min_capacity)` 选择目标容量
- 再调用 `ReallocateExact`

## 13.3 state transition

允许的状态切换：

- `Small -> Small`
- `Small -> External`
- `External -> External`
- `External -> Small`（仅在显式 shrink/降级时）

禁止在布局层之外构造未定义状态。

---

## 14. 核心行为约束

## 14.1 `clear`

规则：

- 不释放 external capacity
- 仅将 size 设为 0
- 保持 `data()[0] == CharT{}`

`clear()` 采用与 `std::basic_string` 一致的基线语义：清空内容但不主动释放已有 external capacity；释放行为由析构、显式 shrink 或后续重分配路径决定。

## 14.2 `reserve`

规则：

- 仅增长，不收缩
- 不改变 size
- 保持内容不变

## 14.3 `resize`

规则：

- 缩小时：直接缩小 size
- 扩大时：确保容量足够，填充 `fill`
- 维护 terminator

## 14.4 `append`

规则：

- 支持普通追加
- 支持 embedded null
- 支持 self-overlap 场景
- 必要时在重分配后重新定位源指针

## 14.5 `push_back / pop_back`

规则：

- `push_back` 视为 `append(&ch, 1)`
- `pop_back` 缩小 size 1，并维护 terminator
- `BasicStringCore::pop_back()` 在空字符串上采用 defensive no-op

说明：

- core 层采用 defensive no-op 是内部安全策略
- 若 public 层 `BasicString::pop_back()` 需要严格对齐 `std::basic_string`，可在 public 层另行定义前置条件
- differential test 需要明确区分 core 层防御式语义与 public 层标准兼容语义

## 14.6 `shrink_to_fit`

规则：

- `Small`：无操作
- `External` 且 `size <= SmallCapacity`：降级为 `Small`
- `External` 且 `size > SmallCapacity`：重新分配为更紧凑的 external 容量

external shrink 目标容量规则：

- 基础目标容量为当前 `size`
- 若当前对象或目标容量处于 `Large` 区间，可由 External 语义策略选择 page-rounded shrink capacity
- shrink 后必须满足 `capacity >= size`
- shrink 后必须保持 `data()[size()] == CharT{}`

---

## 15. 自引用与 overlap 规则

`append / assign / replace` 等操作必须考虑源区间与当前存储重叠。

第一阶段要求：

- 识别 `src` 是否落在当前对象存储范围内
- 记录相对 offset
- 若发生重分配，则重分配后重新定位源区间
- 复制时对重叠区间使用 `Traits::move`

当前阶段至少必须正确支持：

```cpp
s.append(s.data(), s.size());
s.append(s.data() + offset, n);
```

---

## 16. 异常安全

## 16.1 目标

第一阶段实现目标：

- 构造失败不泄漏
- `reserve / resize / append / assign / shrink_to_fit` 尽量提供 strong exception guarantee
- `clear / pop_back` 为 no-throw

### `append` 异常安全边界

第一阶段对 `append` 的异常安全约束如下：

- 对分配失败路径提供 strong exception guarantee
- 对需要重分配的追加路径采用 allocate-copy-commit 策略，提交前不破坏原对象
- 对不需要重分配的原地追加路径，默认以标准 `std::char_traits` 的 `copy / move` 不抛异常为前提
- 若未来支持可能抛异常的 custom traits，则原地追加路径的 strong exception guarantee 需要单独设计

## 16.2 推荐策略

采用：

- allocate
- copy/move
- commit

即：

1. 先分配新缓冲区
2. 拷贝原数据与新增数据
3. 全部成功后再切换状态
4. 失败则回滚并释放临时资源

---

## 17. 不变量

## 17.1 全局不变量

任意时刻对象必须满足：

1. `data() != nullptr`
2. `data()[size()] == CharT{}`
3. `size() <= capacity()`
4. `LayoutPolicy::category(storage_)` 为合法状态
5. moved-from 对象恢复为合法 empty string

其中，moved-from 对象恢复协议由 `BasicStringCore` 负责：move 后源对象应通过 `LayoutPolicy::InitEmpty(storage_)` 恢复为合法 empty `Small` 状态。

## 17.2 `Small` 不变量

依赖 `LayoutPolicy` 保证。

## 17.3 `External` 不变量

依赖 `LayoutPolicy` 保证，并增加：

- external buffer 为本对象独占所有
- allocator 后续能正确释放该缓冲区

---

## 18. 与 LayoutPolicy 的接口约束

`BasicStringCore` 依赖 `LayoutPolicy` 提供：

- `InitEmpty`
- `InitSmall`
- `InitExternal`
- `SetSmallSize`
- `SetExternalSize`
- `SetExternalCapacity`
- `data`
- `size`
- `capacity`
- `is_small`
- `is_external`
- `category`
- `max_external_capacity`
- `CheckInvariants`

`BasicStringCore` 可以依赖 `LayoutPolicy` 暴露出的状态与读写原语，但不直接解释 probe slot、tag 或 `capacity_with_tag` 的底层编码细节。

---

## 19. 与 Allocator 的接口约束

`BasicStringCore` 必须统一通过：

```cpp
std::allocator_traits<Allocator>
```

进行分配与释放。

推荐最小路径：

```cpp
Pointer AllocateChars(SizeType capacity);
void DeallocateChars(Pointer ptr, SizeType capacity) noexcept;
```

其中：

- 分配数量为 `capacity + 1`
- 额外 1 个字符位置始终用于 terminator

---

## 20. 与 Traits 的接口约束

`BasicStringCore` 通过 `Traits` 完成：

- `copy`
- `move`
- `compare`
- `length`

第一阶段不允许在 core 中绕开 `Traits` 自定义字符语义。

---

## 21. 测试要求

## 21.1 生命周期测试

必须覆盖：

- 默认构造
- 指针+长度构造
- string_view 构造
- 拷贝构造
- 移动构造
- 拷贝赋值
- 移动赋值
- 析构

## 21.2 容量与修改测试

必须覆盖：

- `clear`
- `reserve`
- `resize`
- `append`
- `assign`
- `push_back`
- `pop_back`
- `shrink_to_fit`

## 21.3 状态切换测试

必须覆盖：

- `Small -> Small`
- `Small -> External`
- `External -> External`
- `External -> Small`

## 21.4 自引用测试

必须覆盖：

- `append(self)`
- `append(self subrange)`

## 21.5 多 `CharT` 测试

generic 路径必须覆盖：

- `char`
- `char8_t`
- `char16_t`
- `char32_t`
- `wchar_t`

## 21.6 sanitizer / differential

必须支持：

- ASan
- UBSan
- LSan
- 与 `std::basic_string` 的 differential test

---

## 22. 当前阶段明确不做的事情

`BasicStringCore` 当前阶段不做：

- `medium / large` 三段式布局
- COW / refcount
- direct-map backend tag
- `insert / erase / replace` 的完整实现
- SIMD 优化
- safe over-read
- `char` 专用极限 fast path
- allocator propagation 的复杂扩展场景之外的行为创新

---

## 23. 正式设计结论

`BasicStringCore` 是 `amstring` 的正式核心实现层。  
它在 `LayoutPolicy` 之上管理对象生命周期、allocator 分配与释放、核心修改操作、异常安全以及 `External` 语义层策略；同时通过 `Traits` 处理字符级操作，通过 `LayoutPolicy` 维护底层布局一致性。

因此：

1. `BasicStringCore` 是 owning string storage manager
2. `BasicStringCore` 不重复实现布局编码
3. `BasicStringCore` 承担 `Normal / Large` 的 external 策略管理
4. `BasicStringCore` 是 `BasicString` 的唯一底层状态管理核心

---

## 24. 一句话总结

`BasicStringCore` 的正式定位是：**位于 `BasicString` 与 `LayoutPolicy` 之间、统一管理生命周期、allocator、核心修改操作、异常安全与 `External` 语义策略的 owning string core。**
