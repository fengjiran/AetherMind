# AetherMind 问题跟踪

> 已知缺陷与优化待办清单，短生命周期条目。长生命周期/重大方向见 [decisions/](decisions/) 与 [improvement-plan/](improvement-plan/)。
>
> 条目格式：`- [ ] <背景与方案简述>（关联：文件/PR/issue 链接）`。修复完成后勾选并在 CHANGELOG 登记（行为可见时）。

## 未解决

- [ ] `ElementwiseMul` operator inference 与 CPU kernel 支持 broadcast，但 `GraphOpBuilder::AddElementwiseMul` 拒绝不同完整 `TensorSpec`，三层语义不一致（关联：[系统能力演进路线图 §6.2](improvement-plan/06-system-capability-evolution-roadmap.md#62-p0修复-graphopbuilder-与-operator-semantic-漂移)）
- [ ] `WorkspaceRequirement::lifetime/reusable` 尚未影响 offset 规划，`PlanWorkspaceRequirements` 仍顺序累加全部 requirement（关联：[系统能力演进路线图 §13.3](improvement-plan/06-system-capability-evolution-roadmap.md#133-p1workspace-lifetime-真正生效)）
- [ ] activation arena 为所有 activation 顺序分配，尚无 producer/last-consumer liveness reuse，峰值接近全部 activation bytes 之和（关联：[系统能力演进路线图 §9.3](improvement-plan/06-system-capability-evolution-roadmap.md#93-p1activation-liveness-planning)）
- [ ] PRD 的单线程/OpenMP、compile-time dispatch/planning-time resolve、多 Session 验收/单 active KV slot 等表述需要按当前事实与产品目标重新裁决（关联：[系统能力演进路线图 §17.1](improvement-plan/06-system-capability-evolution-roadmap.md#171-当前需要裁决的漂移)）
- [ ] GEMM 尚缺三项 production 门禁证据：裸机 perf/governor/microcode、稳定 5% 级性能门禁（WSL2 噪声 floor 在 binding/小形状超阈值）、raw artifact 的 CI/object-storage retention URL（关联：[GEMM 证据记录与状态](operators/gemm/README.md)）

## 已解决

- [x] HF reader 将 sliding_window 当作未知字段跳过，存在滑窗配置被静默构图为 full attention 的风险（修复：2026-10-26 显式解析，缺失/null 接受，所有显式窗口值拒绝；覆盖加载、直接构图和 runner 分配前失败边界，关联：[第一批验收记录](tests/sliding_window_admission_validation_2026-10-26.md)）
- [x] `HfModelValidator` 接受 `gelu/relu`，而 `BuildLlamaDense` 固定生成 `SiluMul`，存在 accepted config 被静默编译为错误 MLP 语义的风险（修复：2026-10-21 validator 收窄为仅 `silu`，模型加载与直接构图入口均拒绝 `gelu/relu`；回归 4/4 通过，关联：[系统能力演进路线图 §5.2](improvement-plan/06-system-capability-evolution-roadmap.md#52-p0统一接受的模型和实际语义已闭环)）
- [x] `KVCacheManager::Init` 重置 generation 导致旧 view 复活（修复：2026-10-11 跨 re-init 保留 generation、拒绝回绕；正式回归覆盖成功/失败 re-init 与已释放副本，关联：[08 号 KV/Attention 演进提案](improvement-plan/08-kv-cache-and-attention-capability-evolution.md)）
- [x] `KVCacheManager::ResetSession/ReleaseSession` 接受跨 Manager view（修复：2026-10-11 校验 layout/storage/slot owner，拒绝操作且保持双方 reservation；正式回归已通过，关联：[08 号 KV/Attention 演进提案](improvement-plan/08-kv-cache-and-attention-capability-evolution.md)）
- [x] `InferenceSession::Generate` 请求分配 OOM 逃逸异常或因 CPU allocator abort 终止（修复：2026-10-11 将 CPU `ENOMEM` 上报为可恢复失败、在编排和 KV 冷路径边界转换 `bad_alloc`，验证 KV 清理和同 Session 重试，关联：[InferenceSession 设计](modules/inference.md)）
- [x] `docs/designs/` 24 篇设计文档按模块合并为 `docs/modules/` 12 篇；源文档归档到 `docs/archive/designs/`，活跃引用切换到模块文档
- [x] 核心 9 篇文档中 `Phase 1 / Phase 2` 术语 83 处待分类迁移（修复：2026-09-17 全部迁移为具体 capability 表述；AGENTS.md/README.md/docs/README.md/prd/architecture_overview/public-api 均为 0 处，仅本文件保留元描述）
- [x] verify_docs.py 覆盖不足（历史：2026-09-14 曾实现 D5a 全部 7 项；Phase 0 已删除该工具，改由 `docs/README.md` 的写作规则约束）
- [x] designs/amstring/ 中 development_plan、milestones、task_checklist 分类违规（修复：2026-09-14 归档到 docs/archive/designs-legacy/）
- [x] 01 号方案 §2.2 kernel 覆盖表漂移：Linear、RoPE、Argmax 标为"无/阻塞"，实际存在 reference kernel（修复：2026-09-14 更新为 FP32 reference / 可用）
- [x] code_review_guide.md 失效命令与 ammalloc 残留（修复：2026-09-14 标 Deprecated，待重写；docs/README.md 索引同步）
- [x] docs/README.md 未登记 03 号方案：演进提案表只列到 01（修复：2026-09-14 新增 02/03 条目）
- [x] 03 号方案自相矛盾：Draft 提案自称治理权威、核心集含 Draft 提案、agent/ 纳入门禁、authority 作为类型、--no-* 语义矛盾、命名违规计数不准（修复：2026-09-14 版本 1.2 修正 6 条）
- [x] docs/issues.md 未解决条目为空，与已知缺口不一致（修复：2026-09-14 登记 6 条未解决 + 7 条已解决）
