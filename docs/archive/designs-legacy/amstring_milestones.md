# amstring 里程碑计划

## 1. 文档目的

本文档将 `amstring` 开发计划整理为里程碑版本，用于阶段推进、验收和进度跟踪。

若与详细开发计划冲突，以 `amstring_development_plan.md` 与设计文档为准。

---

## 2. 里程碑规划原则

- 里程碑必须遵循 TDD 推进
- 先 generic correctness baseline，再 core，再 public wrapper，再优化
- 每个里程碑都要有清晰的 scope、关键任务、退出条件与主要风险
- 不在 generic 路线稳定前引入 `CharLayoutPolicy`

---

## M0：开发基线建立

### Scope

- 确认 canonical 设计文档
- 建立新的测试与实现入口
- 切断旧文档/旧路径对开发的影响

### 关键任务

1. 明确实现对象：`GenericLayoutPolicy`、`DefaultLayoutPolicy`、`BasicStringCore`、`BasicString`
2. 建立测试文件骨架
3. 建立多 `CharT` 测试矩阵

### 退出条件

- 新开发入口清晰
- 测试骨架可编译
- 团队可按新文档直接推进实现

### 主要风险

- 仍受旧实现命名与结构影响

---

## M1：Generic correctness baseline

### Scope

- `GenericLayoutPolicy<CharT>`
- 多 `CharT` 支持
- probe/tag 编码与 invariant 基线

### 关键任务

1. 实现 `ExternalRep`、`Storage`、`Category`
2. 实现 `Small / External / Invalid` 判别
3. 实现 `PackCapacityWithTag` / `UnpackCapacity` / `UnpackTag`
4. 实现 `InitEmpty` / `InitSmall` / `InitExternal`
5. 实现 `SetSmallSize` / `SetExternalSize` / `SetExternalCapacity`
6. 实现 `CheckInvariants`

### 退出条件

- `sizeof(Storage<CharT>) == 24`（64-bit）
- `char / char8_t / char16_t / char32_t / wchar_t` 参数化测试通过
- 编码/解码/invariant 测试通过

### 主要风险

- 端序相关编码错误
- `max_external_capacity()` 上界处理不一致

---

## M2：Layout selector 接入

### Scope

- `DefaultLayoutPolicy<CharT>`
- generic 路线作为统一默认路径

### 关键任务

1. 实现 `DefaultLayoutPolicy<CharT>`
2. 将 core 默认布局绑定到 selector

### 退出条件

- 所有 `CharT` 默认选择 `GenericLayoutPolicy<CharT>`
- 外部 API 不暴露 selector 细节

### 主要风险

- selector 与 core 默认模板参数不一致

---

## M3：BasicStringCore 生命周期闭环

### Scope

- 构造、析构、copy、move
- allocator-aware 生命周期
- 只读接口

### 关键任务

1. 实现成员与类型别名
2. 实现默认构造与指针+长度构造
3. 实现析构
4. 实现拷贝构造与移动构造
5. 实现 `data` / `c_str` / `size` / `capacity` / `empty`

### 退出条件

- `Small` 与 `External` 两类构造测试通过
- copy 深拷贝测试通过
- move 后源对象恢复为空
- allocator 生命周期路径正确

### 主要风险

- external 释放路径错误
- moved-from 状态恢复不完整

---

## M4：BasicStringCore 修改操作闭环

### Scope

- `clear`
- `reserve`
- `resize`
- `append`
- `assign`
- `push_back` / `pop_back`
- `swap`
- `shrink_to_fit`

### 关键任务

1. 实现容量规划与重分配逻辑
2. 实现 `Small <-> External` 状态迁移
3. 实现 `Large` 计算型语义
4. 实现异常安全基线路径

### 退出条件

- `Small -> Small`
- `Small -> External`
- `External -> External`
- `External -> Small`
- self subrange assign
- exception safety 基线测试全部通过

### 主要风险

- allocator 不等场景下错误窃取 buffer
- `clear()` / `shrink_to_fit()` 语义偏离设计

---

## M5：BasicString public wrapper

### Scope

- `BasicString<CharT, Traits, Allocator>`
- public API 薄包装

### 关键任务

1. 接入 `BasicStringCore`
2. 提供第一阶段 public API
3. 完成标准风格类型别名与容器语义

### 退出条件

- public 语义测试通过
- `BasicString` 不暴露 `LayoutPolicy`
- small/external 对用户透明

### 主要风险

- public 层越过 core 直接操作布局
- API 语义与 `std::basic_string` 基线不一致

---

## M6：差分验证与质量收口

### Scope

- differential test
- sanitizer
- 回归矩阵
- benchmark baseline

### 关键任务

1. 与 `std::basic_string` 建立差分测试
2. 建立多 `CharT` 回归矩阵
3. 执行 ASan / UBSan / LSan / 必要时 TSan
4. 建立 benchmark baseline

### 退出条件

- differential test 稳定
- sanitizer 通过
- benchmark baseline 可复现

### 主要风险

- generic 路线 correctness 与性能表现不平衡

---

## M7：后续优化里程碑

### Scope

- `CharLayoutPolicy`
- `AMMallocAllocator`
- large/page 对齐优化

### 当前状态

M7 已启动。

已完成：

- `CharLayoutPolicy` 正式设计文档
- `CharLayoutPolicy` 首版 2-bit marker 实现
- `DefaultLayoutPolicy<char>` selector 切换
- `BasicString<char>` 差分测试验证
- CharLayoutPolicy 首轮 benchmark 记录

未完成：

- `AMMallocAllocator` 接入
- `Large` / page 对齐进一步优化

### 启动前提

- generic core 稳定
- sanitizer 通过
- differential test 稳定
- benchmark baseline 已建立

### 目标

1. 为 `char` 提供专用高性能路径
2. 接入 `ammalloc` 后端
3. 优化 large object 容量规划与页粒度行为

---

## 3. 里程碑依赖关系

```text
M0 -> M1 -> M2 -> M3 -> M4 -> M5 -> M6 -> M7
```

其中：

- `M1` 是 correctness foundation
- `M3 + M4` 构成 core implementation 主体
- `M6` 是进入优化阶段的硬门槛

---

## 4. 推荐阶段性交付物

### M1 交付物

- `GenericLayoutPolicy` 实现
- 参数化布局测试
- invariant 测试

### M3 交付物

- `BasicStringCore` 生命周期实现
- allocator-aware 构造析构测试

### M4 交付物

- core 修改操作实现
- 状态迁移测试
- 异常安全测试

### M5 交付物

- `BasicString` public wrapper
- public 语义测试

### M6 交付物

- differential test suite
- sanitizer 验证结果
- benchmark baseline 文档或结果记录

---

## 5. 结论

`amstring` 的推进顺序应严格围绕：

1. generic correctness baseline
2. core 生命周期
3. core 修改操作
4. public wrapper
5. 差分验证与质量收口
6. 后续优化

`M6` 达成后，已启动 `CharLayoutPolicy` 路线；`AMMallocAllocator` 与进一步容量规划优化仍处于后续阶段。
