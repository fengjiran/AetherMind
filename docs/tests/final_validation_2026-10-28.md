# 第一至第五批集成与最终验收

- **状态**: Current
- **日期**: 2026-10-28
- **验收对象**: [13 号方案](../improvement-plan/13-deepinfer-quality-and-validation.md) 已实现的五批，使用 [统一入口](../guides/final-validation.md)。
- **基线**: `55f520f60a68`（完整 commit 及 dirty patch 见 raw summary）；前五批已由用户提交，本轮集成改动未暂存、未提交。
- **边界**: 本记录不将整个 13 号方案、PRD、其他平台/精度/家族或安装交付门禁标记为完成。

## 1. 结果

收尾版本完整矩阵的 summary.json 为：

- status=passed
- milestone_complete=true
- source_unchanged=true
- assets_unchanged=true
- 七个请求 profile 全部通过

| profile / 集合 | 实际执行 | 结果 |
|---|---:|---|
| baseline production/lifecycle | 411 | 通过 |
| baseline external numerical，含 near-limit | 20 | 通过 |
| AVX2 production/lifecycle | 411 | 通过 |
| AVX2 ISA candidate/learning + fixtures | 62 | 通过 |
| AVX2 external numerical，含 near-limit | 20 | 通过 |
| ASAN/UBSAN/LSAN production/lifecycle | 317 | 通过，无 sanitizer 报告 |
| ASAN/UBSAN/LSAN ISA | 62 | 通过，无 sanitizer 报告 |
| ASAN/UBSAN/LSAN external numerical | 20 | 通过，无 sanitizer 报告 |
| TSAN 适用 validation/thread-smoke | 239 | 通过，无 TSAN 报告 |
| 独立 CLI 黑盒 | 39 个方法 | 通过 |
| AVX2 / baseline 模型 benchmark 行为合同 | 各 14 个方法 | 通过 |

每个 CTest 切片的 expected/discovered/executed/passed 均一致，failed/skipped/disabled/not_run 均为零；全部确切名称 SHA256 与冻结策略一致。根代理已独立重新解析所有 inventory/JUnit 及 unittest 方法清单，并核对实际 library/binary/cache/compile database 身份。

不同切片有交集，不能将表中数量相加为独立功能测试数。CTest 数量包括 fixture、Python 和 CMake 合同，不全是 instrumented C++。normal profile 保留 interposer/preload 故障注入；sanitizer 明确排除不兼容项，不将这些排除计为通过。

## 2. 新增集成能力

- [资产准备工具](../../tools/prepare_validation_assets.py)：默认离线校验；可显式获取固定 source、生成 admission 或导入/再生 numerical candidate。stage 校验后发布，禁止覆盖既有资产或自动改变期望值。
- [统一驱动](../../tools/run_final_validation.py)：复用七个现有 presets，构建可并发、结果逐项归档；支持局部矩阵和独立 fresh 构建。局部通过不将 milestone_complete 置 true。
- [执行证据核对器](../../tools/verify_ctest_report.py)：核对冻结 inventory、JUnit 完整性和确切测试名称，拒绝缺项、重复、disabled、skip、not-run、错误状态和非有限时间。
- [冻结策略](../../tests/fixtures/final_validation/policy.json)：九个 CTest 切片及 CLI/benchmark verbose 方法集合，数量与摘要均需人工审核后更新。
- 三个新 stdlib 合同接入既有 CTest；最终分别通过 30 个报告核对合同、20 个资产准备合同、22 个编排合同。
- 修复两条历史文档断链：被移除的 src/function.cpp 改为明确历史路径，保留 M2/M4 当时的事实，不指向无关现存代码。

本轮没有增加核心推理 API、kernel 算法或 runtime capability registry。tools/verify_llama_numerical_reference.py 在本轮出现的外部格式调整已保留，纳入执行源码快照，不冒充本轮功能修改。

## 3. 资产与执行身份

执行前后都重新检查：

- 固定 revision source 的 config/weight 身份；
- admission 的 20 个 HF variants 与 generated manifest；
- numerical 的 5 组、120 个 tensor、manifest 和冻结 checksum。

所有检查通过，source/admission manifest/numerical manifest 的前后原始身份也完全一致。参考数据没有重写或重新晋升。

另在全新 .models/integration-bootstrap-52d85916 实际离线引导：复制固定源，生成并验证 admission，复制受校验 numerical bundle，再发布最终目录，全部成功。最初尝试放到仓库 build/ 下的生成目录被已有 admission generator 的“.models 内生成”合同明确拒绝；失败日志保留，改用合法路径后通过。该失败未留下可误当成功的最终 admission 目录。

显式网络下载和指定 HF Python 的再生路径有合同覆盖，本轮未执行网络下载或重新生成 numerical baseline。干净 CI 仍须提供源/参考 artifact 或显式准备它们，不能假定本地 .models 随 Git 交付。

源码快照覆盖实际代码、构建、测试、tools 与 policy；文档可在运行后归档。HEAD、dirty 状态、patch、命令、时间、sanitizer 环境、实际 cache/compile database/binary/library SHA 均保存。默认矩阵重配置并构建既有 preset 目录，未将其冒充全 fresh 或云 CI 运行。

## 4. TSAN 与失败路径

原生 TSAN --gtest_list_tests 在 main 前出现已知 unexpected memory mapping；auto launcher 保存失败证据，然后用进程级 setarch x86_64 -R 重试。最终 239 项适用集合通过。未修改系统级 ASLR，也未把其他启动错误当作可忽略的同类故障。

ASAN 使用 detect_leaks=1，UBSAN/TSAN 遇报告立即失败；没有 suppression。基础线程和同步产品路径的通过不构成并发 Session 支持。既有 TSAN atomic_thread_fence 建模警告与 warnings profile 边界仍按 Q2 跟踪。

一次额外 configure 中断暴露了初版状态文件仍为 running 的收尾缺口。现已补齐 SIGINT/异常处理：保留日志和源码快照、写 failed summary，SIGINT 返回 130；长构建日志实时写入，build stage 在启动前记录。中断运行 74bf62a9 保留原始状态副本并恢复为 failed，不属于已验收矩阵。

首轮完整矩阵 59d720b7 已通过；完成上述收尾后，又重新执行全部七 profile，最终使用 dea2cf94 的结果，未用旧结果替代新源码的验收。

## 5. 复现与原始证据

```bash
python3 tools/prepare_validation_assets.py
python3 tools/run_final_validation.py \
  --profiles baseline avx2 asan tsan cli benchmark benchmark-baseline \
  --models-dir "$PWD/.models" \
  --policy "$PWD/tests/fixtures/final_validation/policy.json" \
  --spdlog-source "$PWD/build/_deps/spdlog-src" \
  --google-benchmark-source "$PWD/build/_deps/google_benchmark-src" \
  --jobs 4 --build-concurrency 3 --tsan-launcher auto
```

最终 raw：build/validation/final/20261028T064533Z_55f520f60a68_desktop-54h5mmi_dea2cf94/

其中 summary.json 和 status.json 为汇总；ctest/ 保存全部九份 inventory/JUnit/日志；probes/ 保存三个黑盒 probe 日志及方法身份；各 profile/ 保存 configure/build 日志。source-before/after 与 policy/build artifact 摘要绑定实际执行版本。此前成功和中断的运行均保留。

统一入口、fresh/局部运行、资产引导和 CI 接入方式见 [最终验收指南](../guides/final-validation.md)。这里只提供本地已执行证据和 CI-ready 命令，没有宣称某个托管 CI workflow 或 artifact retention 已交付。

## 6. 独立 fresh 构建与最终检查

另行真实运行 `--profiles cli --fresh-build`，在独立的本次 builds/ 目录完成配置、核心库/CLI 构建和 39 个黑盒方法，全部通过。GTest/Python 的 CMake 查找被禁用；这是构建依赖边界，不影响外部 Python 验收工具运行。该局部 run 的 status=passed、milestone_complete=false，确认局部成功不会误标完整验收。

raw：build/validation/final/20261028T065121Z_55f520f60a68_desktop-54h5mmi_17e535e8/

最终文档检查为 0 problem、29 个既有 warning，git diff --check 通过；三个新增 stdlib 合同在收尾版本为 22/30/20 项，全部通过。本轮改动未暂存、未提交。

## 7. 退出边界

本次五批集成在所列 glibc Linux x86_64/AVX2-FMA 环境通过，覆盖已实现的模型准入、公开生成、数值/生命周期、ISA、sanitizer 与测量合同。数值与公开生成证据仍限定为已列明 FP32 Llama/standard RoPE 组合。

尚未验收的事项保持开放：

- 物理无 AVX、其他 compiler/OS/architecture、完整 warnings profile；
- 所有 rewrite/packing/prepared kernel 的逐层故障定位和其他 RoPE/精度/模型家族；
- 代表性大模型、冷启动、PMU 归因、可靠性能回退 gate 与模型质量；
- 机械模块依赖检查、正式 install/export 安装树及源码树外安装 consumer。

既有 C/C++ build-contract、C ABI/CLI 测试已经复用，但不等同于安装交付完成。13 号方案保持 In Progress。
