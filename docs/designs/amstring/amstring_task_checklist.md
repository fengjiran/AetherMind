# amstring 任务清单

## 1. 文档目的

本文档将 `amstring` 开发计划与里程碑计划进一步拆解为可执行的 checklist，用于 Phase 1 开发跟踪。

参考文档：

- `amstring_development_plan.md`
- `amstring_milestones.md`
- `amstring_policy_based_architecture_design.md`
- `amstring_storage_architecture_design.md`
- `GenericLayoutPolicy_design.md`
- `BasicStringCore_design.md`

若 checklist 与设计文档冲突，以设计文档为准。

---

## 2. 使用规则

- 必须遵循 TDD：先测试，后实现
- 每个行为单元尽量对应一个原子提交
- 每个里程碑结束前必须完成验证项
- 在 generic baseline 稳定前，不进入 `CharLayoutPolicy` 与 `AMMallocAllocator`

---

## M0：开发基线建立

### 目标

建立新的实现与测试入口，切断旧路径干扰。

### TDD / 基础设施

- [x] 确认 `GenericLayoutPolicy` / `DefaultLayoutPolicy` / `BasicStringCore` / `BasicString` 为当前唯一开发主线
- [x] 确认测试文件命名方案
- [x] 新建 `tests/unit/amstring/test_generic_layout_policy.cpp`
- [x] 新建 `tests/unit/amstring/test_basic_string_core.cpp`
- [x] 新建 `tests/unit/amstring/test_basic_string.cpp`
- [x] 为 `char / char8_t / char16_t / char32_t / wchar_t` 定义参数化测试矩阵

### 实现准备

- [x] 确认相关头文件的最终命名边界
- [x] 确认 `DefaultLayoutPolicy<CharT>` 将在第一阶段统一选择 `GenericLayoutPolicy<CharT>`
- [x] 确认 `Large` 仅作为 core 内部计算型语义

### 验证

- [x] 新测试文件可参与构建
- [x] 测试骨架可编译

---

## M1：Generic correctness baseline

### 目标

实现 `GenericLayoutPolicy<CharT>` 作为多 `CharT` correctness baseline。

### TDD：先写失败测试

- [x] 为空对象编写 `InitEmpty` 行为测试
- [x] 为 `sizeof(Storage<CharT>) == 24` 编写静态断言测试
- [x] 为 `SmallCapacity` 在不同 `CharT` 下的值编写测试
- [x] 为 `Small / External / Invalid` 分类编写测试
- [x] 为 `ProbeMeta` 编码规则编写测试
- [x] 为 `PackCapacityWithTag` / `UnpackCapacity` / `UnpackTag` 编写边界测试
- [x] 为 `max_external_capacity()` 上界编写测试
- [x] 为 terminator 维护编写测试
- [x] 为 `CheckInvariants` 识别非法状态编写测试

### 实现：类型与常量

- [x] 实现 `Category`
- [x] 实现 `ExternalRep<CharT>`
- [x] 实现 `Storage<CharT>`
- [x] 实现 `kStorageBytes`
- [x] 实现 `kSmallSlots`
- [x] 实现 `kSmallCapacity`
- [x] 实现 `kProbeBits`
- [x] 实现 `kPackedWordBits`
- [x] 实现 `kPayloadBits`
- [x] 实现 `kProbeByteOffset`
- [x] 实现 `kExternalTag`

### 实现：只读接口与 helper

- [x] 实现 `is_small()`
- [x] 实现 `is_external()`
- [x] 实现 `category()`
- [x] 实现 `data()`
- [x] 实现 `size()`
- [x] 实现 `capacity()`
- [x] 实现 `max_external_capacity()`
- [x] 实现 probe/meta 读写 helper
- [x] 实现 `PackCapacityWithTag`
- [x] 实现 `UnpackCapacity`
- [x] 实现 `UnpackTag`

### 实现：状态原语

- [x] 实现 `InitEmpty`
- [x] 实现 `InitSmall`
- [x] 实现 `InitExternal`
- [x] 实现 `SetSmallSize`
- [x] 实现 `SetExternalSize`
- [x] 实现 `SetExternalCapacity`
- [x] 实现 `CheckInvariants`

### 验证

- [x] `char` 参数化测试通过
- [x] `char8_t` 参数化测试通过
- [x] `char16_t` 参数化测试通过
- [x] `char32_t` 参数化测试通过
- [x] `wchar_t` 参数化测试通过
- [x] narrow target 构建通过
- [x] 相关测试命令通过（建议使用单测过滤）

---

## M2：Layout selector 接入

### 目标

实现 `DefaultLayoutPolicy<CharT>` 并接入 generic 路线。

### TDD：先写失败测试

- [x] 编写 `DefaultLayoutPolicy<char>` 选择 generic 的测试
- [x] 编写其他 `CharT` 选择 generic 的测试
- [x] 编写 core 默认布局选择行为测试

### 实现

- [x] 实现 `DefaultLayoutPolicy<CharT>`
- [x] 将 `BasicStringCore` 的默认布局参数接入 selector
- [x] 确认 public 层不暴露 `LayoutPolicy`

### 验证

- [x] selector 测试通过
- [x] core 编译通过并使用正确布局类型

---

## M3：BasicStringCore 生命周期闭环

### 目标

实现构造、析构、copy、move 与只读接口。

### TDD：先写失败测试

- [x] 编写默认构造得到 empty small string 的测试
- [x] 编写小字符串指针+长度构造测试
- [x] 编写大字符串指针+长度构造测试
- [x] 编写 copy small 测试
- [x] 编写 copy external 深拷贝测试
- [x] 编写 move small 后源对象恢复测试
- [x] 编写 move external 后源对象恢复测试
- [x] 编写析构 external 正确释放测试
- [x] 编写 `data()` / `c_str()` / `size()` / `capacity()` / `empty()` 测试
- [x] 编写 `select_on_container_copy_construction` 路径测试

### 实现：成员与只读能力

- [x] 实现类型别名
- [x] 实现 `storage_`
- [x] 实现 `allocator_`
- [x] 实现默认构造
- [x] 实现指针+长度构造
- [x] 实现析构
- [x] 实现拷贝构造
- [x] 实现移动构造
- [x] 实现 `data()`
- [x] 实现 `c_str()`
- [x] 实现 `size()`
- [x] 实现 `capacity()`
- [x] 实现 `empty()`

### 验证

- [x] 生命周期相关测试通过
- [x] moved-from 恢复测试通过
- [x] external buffer 分配大小满足 `capacity + 1`

---

## M4：BasicStringCore 修改操作闭环

### 目标

完成第一阶段核心修改语义与状态迁移。

### TDD：先写失败测试

- [ ] 为 `clear()` 编写保持 capacity 的测试
- [ ] 为 `reserve()` 编写扩容测试
- [ ] 为 `resize()` 截断测试
- [ ] 为 `resize()` 补字符测试
- [ ] 为 `append(const CharT*, SizeType)` 编写测试
- [ ] 为 `append(SizeType, CharT)` 编写测试
- [ ] 为 `assign(const CharT*, SizeType)` 编写测试
- [ ] 为 self subrange assign 编写测试
- [ ] 为 `push_back()` 编写测试
- [ ] 为 `pop_back()` 编写测试
- [ ] 为 `swap()` 在不同 allocator 传播场景下编写测试
- [ ] 为 `shrink_to_fit()` 编写回退到 `Small` 的测试
- [ ] 为 `Small -> Small` 状态迁移编写测试
- [ ] 为 `Small -> External` 状态迁移编写测试
- [ ] 为 `External -> External` 状态迁移编写测试
- [ ] 为 `External -> Small` 状态迁移编写测试
- [ ] 为异常安全路径编写测试

### 实现：容量与重分配

- [ ] 实现 `EnsureCapacityForSize`
- [ ] 实现 `NextCapacity`
- [ ] 实现 `IsLargeCapacity`
- [ ] 实现 `RoundUpCapacityToPage`
- [ ] 实现 `ReallocateExact`
- [ ] 实现 `ReallocateAtLeast`

### 实现：修改接口

- [ ] 实现 `clear()`
- [ ] 实现 `reserve()`
- [ ] 实现 `resize()`
- [ ] 实现 `append(const CharT*, SizeType)`
- [ ] 实现 `append(std::basic_string_view<CharT, Traits>)`
- [ ] 实现 `append(SizeType, CharT)`
- [ ] 实现 `assign(const CharT*, SizeType)`
- [ ] 实现 `assign(std::basic_string_view<CharT, Traits>)`
- [ ] 实现 `assign(SizeType, CharT)`
- [ ] 实现 `push_back()`
- [ ] 实现 `pop_back()`
- [ ] 实现 `swap()`
- [ ] 实现 `shrink_to_fit()`

### 验证

- [ ] 状态迁移矩阵测试通过
- [ ] exception safety 基线测试通过
- [ ] terminator 保持测试通过
- [ ] `Large` 仅影响 core 策略、不影响 layout state 的测试通过

---

## M5：BasicString public wrapper

### 目标

在 core 上建立 public API 薄包装。

### TDD：先写失败测试

- [ ] 编写默认构造 public 语义测试
- [ ] 编写构造与赋值 public 语义测试
- [ ] 编写 `data()` / `c_str()` / `size()` / `capacity()` / `empty()` public 测试
- [ ] 编写 `clear()` / `reserve()` / `resize()` public 测试
- [ ] 编写 `append()` / `assign()` public 测试
- [ ] 编写 `push_back()` / `pop_back()` public 测试
- [ ] 编写 `shrink_to_fit()` public 测试
- [ ] 编写 small/external 对用户透明的测试

### 实现

- [ ] 定义 `BasicString<CharT, Traits, Allocator>`
- [ ] 接入 `BasicStringCore`
- [ ] 实现第一阶段 public API
- [ ] 补齐必要类型别名
- [ ] 保持 public 层不暴露 `LayoutPolicy`

### 验证

- [ ] public 语义测试通过
- [ ] 与 core 行为保持一致

---

## M6：差分验证与质量收口

### 目标

用差分测试、sanitizer 和 benchmark baseline 收口质量。

### TDD / 差分测试

- [ ] 建立 `BasicString` 与 `std::basic_string` 的差分测试矩阵
- [ ] 覆盖构造、赋值、append、assign、resize、reserve、shrink_to_fit
- [ ] 覆盖多 `CharT` 场景
- [ ] 覆盖随机操作序列测试

### 质量验证

- [ ] 运行 narrowest relevant unit tests
- [ ] 运行完整 amstring 相关单测
- [ ] 运行 ASan 验证
- [ ] 运行 UBSan 验证
- [ ] 必要时运行 TSan 变体
- [ ] 建立 benchmark baseline

### 交付记录

- [ ] 记录差分测试结论
- [ ] 记录 sanitizer 结果
- [ ] 记录 benchmark baseline

---

## M7：后续优化（当前不启动）

### 目标

保留后续优化入口，但不在当前闭环中提前进入。

### 启动前提

- [ ] generic core 稳定
- [ ] sanitizer 通过
- [ ] differential test 稳定
- [ ] benchmark baseline 已建立

### 后续任务

- [ ] 设计并实现 `CharLayoutPolicy`
- [ ] 为 `char` 接入专用高性能路径
- [ ] 接入 `AMMallocAllocator`
- [ ] 优化 `Large` 区间的页粒度容量规划

---

## 3. 推荐提交节奏

- [ ] `test(amstring): add GenericLayoutPolicy TDD skeleton`
- [ ] `feat(amstring): implement GenericLayoutPolicy core encoding`
- [ ] `test(amstring): add GenericLayoutPolicy invariant coverage`
- [ ] `feat(amstring): add DefaultLayoutPolicy selector`
- [ ] `feat(amstring): implement BasicStringCore lifecycle`
- [ ] `test(amstring): cover BasicStringCore state transitions`
- [ ] `feat(amstring): implement BasicStringCore mutating operations`
- [ ] `feat(amstring): add BasicString public wrapper`
- [ ] `test(amstring): add differential and regression coverage`

---

## 4. 完成标准

以下条件满足时，可认为 `amstring` Phase 1 主线开发完成：

- [ ] `GenericLayoutPolicy<CharT>` 通过多 `CharT` correctness 测试
- [ ] `DefaultLayoutPolicy<CharT>` 接入完成
- [ ] `BasicStringCore` 生命周期与修改语义闭环完成
- [ ] `BasicString` public wrapper 完成
- [ ] differential test 稳定通过
- [ ] sanitizer 通过
- [ ] benchmark baseline 建立完成
