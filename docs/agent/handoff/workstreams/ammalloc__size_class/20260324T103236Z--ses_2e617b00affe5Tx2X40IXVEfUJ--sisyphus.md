---
kind: handoff
schema_version: "1.1"
created_at: 2026-03-24T10:32:36Z
session_id: ses_2e617b00affe5Tx2X40IXVEfUJ
task_id: T-b4d09f62-8b6b-49f7-a75e-99ae3ce9842a
module: ammalloc
submodule: size_class
slug: null
agent: sisyphus
status: active
bootstrap_ready: true
memory_status: not_needed
supersedes: 20260313T100000Z--ses_current--sisyphus.md
closed_at: null
closed_reason: null
---

# Handoff: ammalloc SizeClass class-based policy sync

## 目标
同步 `SizeClass` 的实现、契约注释、测试和设计文档，使 batch/page 策略、size=0 语义和设计说明与当前代码一致。

## 当前状态

### 已完成
- ✅ `SizeClass::CalculateBatchSize()` 改为按 size class 查表：`BatchByIndex(Index(original_size))`，不再按原始请求值直接计算（`ammalloc/include/ammalloc/size_class.h:245-264`）。
- ✅ `SizeClass::GetMovePageNum()` 改为按 size class 查表：`MovePagesByIndex(Index(original_size))`，非法输入 `0` 或 `>MAX_TC_SIZE` 返回 `0`（`ammalloc/include/ammalloc/size_class.h:266-279`）。
- ✅ 新增/保留编译期表：`small_index_table_`、`size_table_`、`batch_table_`、`move_page_table_`（`ammalloc/include/ammalloc/size_class.h:303-358`）。
- ✅ `Index()`/`Size()`/策略接口的公共注释已同步到当前契约，包括 `[0, kSmallSizeThreshold]` 查表快路径、size=0 分层语义、class-based policy（`ammalloc/include/ammalloc/size_class.h:145-148,245-279`）。
- ✅ 采样式 consteval 校验仍在：边界映射、`Size(Index(s)) >= s`、idempotent、单调性（`ammalloc/include/ammalloc/size_class.h:375-460`）。
- ✅ 设计文档 `docs/designs/ammalloc/size_class_design.md` 已同步：
  - size=0 分层语义（`53-63`）
  - 1024B 查表快路径（`77, 216-228, 540-541`）
  - class-based batch/page policy（`312-368`）
  - 48 个桶和表大小（`512-553`）
  - 顶层 `am_malloc` 与 `CentralCache` 示例调用链（`469-498`）
  - 文档版本更新到 2.1（`606-609`）
- ✅ 本轮创建了两个提交：
  - `54f7c25` `fix(ammalloc): make SizeClass batch and page policies class-based`
  - `a16ff0e` `docs(ammalloc): sync SizeClass design doc with implementation`

### 未完成
- 未涉及 stable memory 回写。
- 未涉及新的 benchmark 运行。

## 涉及文件
- `ammalloc/include/ammalloc/size_class.h`
- `tests/unit/test_size_class.cpp`
- `docs/designs/ammalloc/size_class_design.md`
- `ammalloc/include/ammalloc/thread_cache.h`
- `ammalloc/src/thread_cache.cpp`
- `ammalloc/src/central_cache.cpp`
- `ammalloc/src/ammalloc.cpp`
- `ammalloc/include/ammalloc/config.h`

## 已确认接口与不变量
- `SizeConfig::kSmallSizeThreshold == 1024`（`ammalloc/include/ammalloc/config.h:44-50`）。
- `Index(0) == 0`，`RoundUp(0) == 8`（`ammalloc/include/ammalloc/size_class.h:371-372`）。
- `CalculateBatchSize(0) == 0`，`GetMovePageNum(0) == 0`（`ammalloc/include/ammalloc/size_class.h:256-279`；测试 `tests/unit/test_size_class.cpp:311-317`）。
- `RoundUp(size > MAX_TC_SIZE)` 原样透传；`Index(size > MAX_TC_SIZE)` 返回 `max()` sentinel（`ammalloc/include/ammalloc/size_class.h:230-238`；测试 `tests/unit/test_size_class.cpp:302-326`）。
- `ThreadCache::Allocate()` 传 raw size 给 `Index()`，传 `RoundUp(size)` 给 `CentralCache` 慢路径（`ammalloc/include/ammalloc/thread_cache.h:32-45`）。
- `ThreadCache::FetchFromCentralCache()` 以 rounded/class size 调用 `CalculateBatchSize(size)`（`ammalloc/src/thread_cache.cpp:27-45`）。
- `CentralCache::InitTransferCache()` / `GetOneSpan()` 都已按 class size 使用 `CalculateBatchSize(Size(i))` / `GetMovePageNum(size)`（`ammalloc/src/central_cache.cpp:265-297`）。
- 顶层 `am_malloc()` 对 `size > MAX_TC_SIZE` 直接走 `am_malloc_slow_path()`；`am_free()` 对大对象 `span->obj_size == 0` 走慢路径释放（`ammalloc/src/ammalloc.cpp:96-123,150-181`）。
- `MAX_TC_SIZE` 必须是 2 的次幂，且最后一个 class 精确落在 `MAX_TC_SIZE`（`ammalloc/include/ammalloc/size_class.h:378-380`）。

## 阻塞点
- 无阻塞点。

## 推荐下一步
1. 如果要继续演进 `SizeClass`，优先决定是否把 `tests/unit/test_size_class.cpp` 中非功能性的格式漂移单独清理，避免后续 diff 噪音。
2. 如果要继续推进 allocator 前端策略，下一步应回到 `thread_cache` / `central_cache` 的动态水位线或 TransferCache 策略，并以当前 class-based `batch/page` 作为稳定前提。
3. 如需发布远端，当前分支 `main` 已领先 `origin/main` 两个提交，可直接 `git push`。

## 验证方式
- 构建：`cmake --build build --target aethermind_unit_tests -j` — 已执行并通过。
- 单测：`./build/tests/unit/aethermind_unit_tests --gtest_filter=SizeClass*` — 已执行并通过（14/14）。
- git 状态：`git status` — 已执行，工作区干净。
