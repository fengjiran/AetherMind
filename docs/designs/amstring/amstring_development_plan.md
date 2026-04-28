# amstring 开发计划

## 1. 文档目的

本文档基于以下设计文档，给出 `amstring` Phase 1 的正式开发计划：

- `amstring_policy_based_architecture_design.md`
- `amstring_storage_architecture_design.md`
- `GenericLayoutPolicy_design.md`
- `BasicStringCore_design.md`

本文档服务于实现推进，不重新定义架构；若与设计文档冲突，以设计文档为准。

---

## 2. 开发总原则

### 2.1 Canonical baseline

- 当前 `amstring` 设计文档是 canonical baseline
- 不对齐旧实现、旧 checklist 或旧命名
- 第一阶段先建立 generic correctness baseline，再进入优化路径

### 2.2 TDD 工作流

每个开发单元遵循以下闭环：

1. 先写失败测试
2. 用测试精确定义当前行为、不变量或状态迁移
3. 编写最小实现使测试通过
4. 在测试持续通过前提下进行小步重构
5. 再进入下一个行为单元

### 2.3 实现顺序原则

实现顺序固定为：

1. `GenericLayoutPolicy<CharT>`
2. `DefaultLayoutPolicy<CharT>`
3. `BasicStringCore`
4. `BasicString`
5. 差分测试、sanitizer、benchmark
6. 后续优化路径（`CharLayoutPolicy`、`AMMallocAllocator`）

### 2.4 Phase 1 范围约束

第一阶段约束如下：

- 在 generic baseline 稳定前，`char` 先走 `GenericLayoutPolicy<char>`
- `CharLayoutPolicy` 作为 M6 之后再进入的优化路径（当前已在 M7 首轮实现中启动）
- `Large` 只作为 `BasicStringCore` 内部计算型语义
- `LayoutPolicy` 只表达 `Small / External` 两态
- `BasicString` 不暴露 `LayoutPolicy` 模板参数

---

## 3. 代码与测试组织建议

### 3.1 代码侧

建议围绕以下实现单元推进：

- `GenericLayoutPolicy<CharT>`
- `DefaultLayoutPolicy<CharT>`
- `BasicStringCore<CharT, Traits, Allocator, LayoutPolicy>`
- `BasicString<CharT, Traits, Allocator>`

### 3.2 测试侧

建议按职责拆分测试文件：

- `amstring/test_generic_layout_policy.cpp`
- `amstring/test_basic_string_core.cpp`
- `amstring/test_basic_string.cpp`

其中：

- `GenericLayoutPolicy` 重点覆盖布局编码、状态判别、不变量、多 `CharT`
- `BasicStringCore` 重点覆盖生命周期、状态迁移、容量策略、异常安全
- `BasicString` 重点覆盖 public 语义与对 `std::basic_string` 的差分验证

---

## 4. Phase 0：开发基线整理

### 4.1 目标

建立新的实现与测试入口，避免旧路径继续影响后续开发。

### 4.2 工作项

1. 确认新命名与头文件边界
2. 明确旧实现为废弃路径
3. 建立新测试文件骨架
4. 确认 `char / char8_t / char16_t / char32_t / wchar_t` 的测试覆盖矩阵

### 4.3 验收标准

- 新开发路径清晰
- 测试文件骨架已建立
- 不再依赖旧设计文档推进实现

---

## 5. Phase 1：测试骨架先行

### 5.1 目标

先搭建 TDD 骨架，再逐步填充实现。

### 5.2 工作项

1. 建立 `GenericLayoutPolicy` 参数化测试骨架
2. 建立 `BasicStringCore` 行为测试骨架
3. 建立 `BasicString` public 语义测试骨架
4. 明确差分测试入口与 sanitizer 验证入口

### 5.3 第一批失败测试建议

#### `GenericLayoutPolicy`

- `InitEmpty` 构造合法 `Small` 空对象
- `SmallCapacity` 与 `Storage` 尺寸符合设计
- `Small` probe 编码正确
- `External` tag 解码正确
- `capacity_with_tag` 编解码正确
- `CheckInvariants` 能识别非法状态

#### `BasicStringCore`

- 默认构造为空 small string
- 指针+长度构造在 `Small / External` 两侧行为正确
- copy 构造 external 走深拷贝
- move 构造后源对象恢复为空

#### `BasicString`

- 默认构造、`size()`、`capacity()`、`empty()` 语义正确
- small/external 对用户透明

### 5.4 验收标准

- 测试骨架可编译
- 初始失败测试符合预期

---

## 6. Phase 2：实现 `GenericLayoutPolicy<CharT>`

### 6.1 目标

先建立多 `CharT` 通用布局 correctness baseline。

### 6.2 实现顺序

1. 类型与常量
   - `Category`
   - `ExternalRep`
   - `Storage`
   - `kStorageBytes`
   - `kSmallSlots`
   - `kSmallCapacity`
   - `kProbeBits`
   - `kPayloadBits`
   - `kProbeByteOffset`
   - `kExternalTag`
   - `max_external_capacity()`
2. 只读接口
   - `is_small()`
   - `is_external()`
   - `category()`
   - `data()`
   - `size()`
   - `capacity()`
3. 编码 helper
   - probe 读写
   - `PackCapacityWithTag`
   - `UnpackCapacity`
   - `UnpackTag`
4. 初始化接口
   - `InitEmpty`
   - `InitSmall`
   - `InitExternal`
5. state-specific mutator
   - `SetSmallSize`
   - `SetExternalSize`
   - `SetExternalCapacity`
6. `CheckInvariants`

### 6.3 测试重点

- 64-bit 下 `sizeof(Storage<CharT>) == 24`
- `SmallCapacity` 对不同 `CharT` 的推导正确
- `Small / External / Invalid` 分类正确
- terminator 维护正确
- little-endian / big-endian 编码逻辑正确
- 容量上界通过 `max_external_capacity()` 显式约束

### 6.4 验收标准

- 多 `CharT` 参数化测试全部通过
- invariant 测试通过
- 编码与解码行为符合设计文档

---

## 7. Phase 3：实现 `DefaultLayoutPolicy<CharT>`

### 7.1 目标

将第一阶段 selector 固定到 generic 路线。

### 7.2 工作项

1. 实现：

```cpp
template<typename CharT>
struct DefaultLayoutPolicy {
    using type = GenericLayoutPolicy<CharT>;
};
```

2. 将 `BasicStringCore` 的默认布局选择绑定到 selector

### 7.3 验收标准

- 所有 `CharT` 默认选中 `GenericLayoutPolicy<CharT>`
- 外部用户不感知 selector 细节

---

## 8. Phase 4：实现 `BasicStringCore` 生命周期与只读能力

### 8.1 目标

先让 `BasicStringCore` 成为稳定的 owning string storage manager。

### 8.2 实现顺序

1. 类型别名与成员变量
2. 默认构造
3. 指针+长度构造
4. 析构
5. 拷贝构造
6. 移动构造
7. 只读接口：`data` / `c_str` / `size` / `capacity` / `empty`

### 8.3 核心约束

- 构造与析构统一通过 `LayoutPolicy` 维护布局
- 分配与释放统一通过 `std::allocator_traits<Allocator>`
- external buffer 分配数量为 `capacity + 1`
- move 后源对象恢复为合法 empty string

### 8.4 测试重点

- `Small` 构造正确
- `External` 构造正确
- copy 深拷贝
- move 窃取资源并恢复源对象
- allocator copy construction 规则正确

### 8.5 验收标准

- 生命周期相关测试通过
- moved-from 恢复测试通过
- external 释放路径正确

---

## 9. Phase 5：实现 `BasicStringCore` 修改操作

### 9.1 目标

完成第一阶段 core 基线语义。

### 9.2 实现顺序

1. `clear`
2. `reserve`
3. `resize`
4. `append`
5. `assign`
6. `push_back`
7. `pop_back`
8. `swap`
9. `shrink_to_fit`

### 9.3 核心约束

- `clear()` 不主动释放 external capacity
- `assign` 第一阶段优先使用“临时对象 + swap”路径
- allocator 不等且不可传播时，move assignment 不得偷 external buffer
- `Large` 只影响增长策略、页粒度对齐和 `shrink_to_fit()`
- 不允许构造 layout 文档之外的未定义状态

### 9.4 测试重点

- `Small -> Small`
- `Small -> External`
- `External -> External`
- `External -> Small`
- self subrange assign
- exception safety
- terminator 保持
- 容量上界约束

### 9.5 验收标准

- core 行为测试全部通过
- 关键状态迁移覆盖完整
- 异常安全基线成立

---

## 10. Phase 6：实现 `BasicString`

### 10.1 目标

在稳定 core 之上提供 public API 薄包装。

### 10.2 工作项

1. 定义 `BasicString<CharT, Traits, Allocator>`
2. 接入 `BasicStringCore`
3. 提供第一阶段最小 public API
4. 补齐标准风格类型别名与容器语义

### 10.3 第一阶段 public API 建议

- 构造与赋值
- `data()` / `c_str()` / `size()` / `capacity()` / `empty()`
- `clear()` / `reserve()` / `resize()`
- `append()` / `assign()`
- `push_back()` / `pop_back()`
- `shrink_to_fit()`

### 10.4 测试重点

- 与 `std::basic_string` 基线语义一致
- small/external 对用户透明
- allocator-aware 语义正确

### 10.5 验收标准

- public 语义测试通过
- `BasicString` 不暴露 `LayoutPolicy`

---

## 11. Phase 7：差分验证与质量收口

### 11.1 目标

证明 generic 路线稳定可用，再考虑优化路径。

### 11.2 工作项

1. 与 `std::basic_string` 做差分测试
2. 建立多 `CharT` 回归矩阵
3. 加强 invariant 测试
4. 执行 sanitizer 验证
5. 建立 benchmark baseline

### 11.3 验收标准

- differential test 稳定通过
- sanitizer 通过
- benchmark baseline 可复现

---

## 12. Phase 8：后续优化阶段

当前已进入该阶段的第一部分：`CharLayoutPolicy`。

当前状态：

1. [x] `CharLayoutPolicy` 设计、首版实现、selector 切换与首轮验证
2. [ ] `AMMallocAllocator`
3. [ ] 更细粒度的 `Large` 页对齐与容量规划
4. [ ] `char` 专用 benchmark 优化

进入条件：

- generic core 稳定
- sanitizer 通过
- differential test 稳定
- benchmark 基线建立

说明：上述进入条件已满足到足以启动 `CharLayoutPolicy` 路线；剩余优化项继续以后续验证与 benchmark 为门槛推进。

---

## 13. 推荐 commit 粒度

建议按以下粒度推进：

1. `test(amstring): add GenericLayoutPolicy TDD skeleton`
2. `feat(amstring): implement GenericLayoutPolicy core encoding`
3. `test(amstring): add GenericLayoutPolicy invariant coverage`
4. `feat(amstring): add DefaultLayoutPolicy selector`
5. `feat(amstring): implement BasicStringCore lifecycle`
6. `test(amstring): cover BasicStringCore state transitions`
7. `feat(amstring): implement BasicStringCore mutating operations`
8. `feat(amstring): add BasicString public wrapper`
9. `test(amstring): add differential and regression coverage`

---

## 14. 主要风险与缓解

### 14.1 布局与语义混淆

风险：把 `Large` 错做成第三种 layout state。

缓解：始终保持 `LayoutPolicy` 只表达 `Small / External` 两态，`Large` 仅由 core 动态计算。

### 14.2 过早优化

风险：在 generic baseline 稳定前过早引入 `CharLayoutPolicy`。

缓解：以差分测试、sanitizer、benchmark 基线作为优化前置条件。

### 14.3 allocator 语义错误

风险：move assignment / swap 在 allocator 不等场景下错误窃取资源。

缓解：优先按设计文档中的保守路径实现，并用独立测试覆盖传播语义。

### 14.4 TDD 漂移

风险：先写完整实现，再回头补测试。

缓解：每个 commit 聚焦单一行为单元，先提交测试，再提交实现或同 commit 保持 test-first 证据。

---

## 15. 结论

`amstring` 的第一阶段开发应以 `GenericLayoutPolicy<CharT>` 为正确性起点，以 `BasicStringCore` 为语义与生命周期核心，以 `BasicString` 为 public 包装边界，并通过 TDD、差分测试和 sanitizer 收口质量。

在 generic baseline 未稳定前，不进入 `CharLayoutPolicy` 与 `AMMallocAllocator` 优化路径。
