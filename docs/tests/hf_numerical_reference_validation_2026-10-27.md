# HF 外部数值参考第四批验收

- **状态**: Current
- **日期**: 2026-10-27
- **关联方案**: [13 号方案 Q3](../improvement-plan/13-deepinfer-quality-and-validation.md)、[算子工作流](../guides/operator-development-workflow.md)
- **执行基线**: `ff2f94797033938e8e5d7c2e39810904142f67fd` 加工作树补丁；执行时包含第三批及本批未提交改动，未暂存、未提交。
- **环境**: WSL2 Ubuntu 24.04、Linux x86_64、Intel Core Ultra 9 285H（AVX2/FMA）、GCC 14.2.0、CMake/CTest 3.28.3、GoogleTest 1.16.0。

## 1. 实现与职责边界

新增 [HF 参考导出器](../../tools/generate_llama_numerical_reference.py)、[只读校验/差异工具](../../tools/verify_llama_numerical_reference.py)、[配方及冻结 checksum](../../tests/fixtures/inference_numerical/README.md) 和 [C++ 固定轨迹测试](../../tests/unit/inference/test_hf_fixed_trajectory.cpp)。生成模型和 reference.safetensors 放在 gitignored `.models/`；仓库仅保存生成逻辑、配方、环境要求和已审阅的身份/摘要。

独立参考采用 CPython 3.12.3、torch 2.9.0+cpu、transformers 4.57.1、numpy 2.3.4、safetensors 0.6.2。使用 CPU FP32/eager attention、eval/inference_mode、确定性算法、单 intra/inter-op 线程；导入 torch 前固定 ATEN_CPU_CAPABILITY=default、MKL_CBWR=COMPATIBLE 和线程环境。真实 checkpoint 固定 HF revision，合成权重也先序列化再由 HF 重新加载。参考生成不调用 AetherMind 的 graph、RoPE、attention 或 kernel 计算。

teacher forcing 固定 Prefill 输入及两步 Decode 输入，独立 greedy 另从 fresh cache 开始，避免自由生成分叉掩盖固定轨迹的定位。manifest 记录参考源码/二进制、模型、generator、recipe、requirements、文件及 tensor payload 的身份；普通校验、比较和 CI 均不更新期望值。本次使用最终导出器重新生成另一目录，120 个 tensor、资产及元数据均与冻结基线完全一致。

C++ 测试沿真实 ModelLoader → ModelCompiler → Runtime → PrepareExecutableModel → Execute/Generate 路径运行。内部观察 helper 留在 tests，提取旧 direct 测试已有的 PrepareContext/FindOutput；既有 scalar oracle 数学实现未改变。snapshot 在有效期内复制到 owning vector，Safetensors 字节通过 memcpy 读取。临时 context 在释放 KV reservation 前销毁，显式释放使 RAII view 失效。未增加公开 Generate 调试接口或核心库 Python 依赖，也未修改生产数学实现。

## 2. 数值合同与覆盖范围

| 参考组 | 来源/形状 | 固定轨迹 |
|---|---|---|
| mha-untied | 固定 revision 的 f32-explicit MHA；2 层、QH/KVH=4、D=4 | Prefill 3 token + 两步 Decode |
| mha-tied | 同源 f32-tied；embedding/head 绑定 | 同上 |
| gqa-tail | seed=20261027；2 层、H=24、I=37、V=31、QH/KVH=4/2、D=6 | 同上 |
| mqa-tail | seed=20261028；1 层、H=18、I=22、V=29、QH/KVH=3/1、D=6 | 同上 |
| mha-near-limit | 同源 untied MHA；context=2048 | Prefill 2046 + 两步 Decode，提交到 2048 |

真实源为 [peft-internal-testing/tiny-random-LlamaForCausalLM](https://huggingface.co/peft-internal-testing/tiny-random-LlamaForCausalLM/tree/d0b81c5e23f2d61bb7a5f430a8cd6f3e711c0cb3)，revision 为 d0b81c5e23f2d61bb7a5f430a8cd6f3e711c0cb3；沿用 [admission 配方](../../tests/fixtures/inference_admission/recipe.json) 的来源校验。

每组 24 个 tensor，共 120 个。短 Prefill 比较全部 logits 行和完整 vocab；near-limit Prefill 仅保存/比较最后一行 logits，但比较所有位置的 Argmax token、所有层的 committed K/V。两步 Decode 均比较完整单行 logits、token、全部 committed K/V 和 watermark。K 为 post-RoPE，V 未旋转，布局为 [layer, committed_position, kv_head, head_dim]。

逐元素要求 `abs(actual - expected) <= atol + rtol * abs(expected)`：logits 为 atol=2e-5、rtol=2e-4，KV 为 atol=2e-6、rtol=2e-4。shape、选行、token、watermark 必须精确相等，非有限值失败。差异报告包含首次失败的阶段和 logits/KV 坐标。

四组短模型在 O1/plain、O2/packed 下分别进行固定轨迹及公开 Generate 检查；Generate 每次生成 3 token，同一 Session 连续执行两个 fresh request。near-limit 的 C++ 验证采用固定 teacher forcing。preset 通过不表示每个可选 kernel 在所有形状上均被选中。

## 3. 实际执行结果

| 正式构建与集合 | 项数 | 结果 |
|---|---:|---|
| baseline 新增聚焦 GTest | 27 | 全部通过 |
| AVX2 新增聚焦 GTest | 27 | 全部通过 |
| ASAN/UBSAN/LSAN 新增聚焦 GTest | 27 | 全部通过，无 sanitizer 报告 |
| baseline 生产/lifecycle CTest | 406 | 全部通过，0 failed/skipped |
| AVX2 生产/lifecycle CTest | 406 | 全部通过，0 failed/skipped |
| ASAN/UBSAN/LSAN 生产/lifecycle CTest | 312 | 全部通过，0 failed/skipped，无 sanitizer 报告 |
| baseline numerical-model CTest | 20 | 全部通过，含 near-limit |
| AVX2 validation-numerical preset | 20 | 全部通过，含 near-limit |
| ASAN/UBSAN/LSAN numerical-model CTest | 20 | 全部通过，含 near-limit，无 sanitizer 报告 |
| stdlib checker 契约测试 | 46 | 全部通过 |
| 冻结资产校验 | 5 组 / 120 tensors | 全部通过 |
| 最终导出器重新生成与比较 | 120 tensors | identical，metadata/asset/tensor differences 均为空 |

27 个 GTest 由 8 个固定轨迹、8 个公开 Generate、2 个 near-limit、8 个比较器合同和 1 个真实权重扰动组成。20 个 CTest 是其中 19 个需资产的 GTest 加独立数值 fixture。上述集合有交集，不能相加为独立测试数；生产集合还含 CLI、Python、CMake 和其他 fixtures，并非全部为 instrumented C++。

ASAN 使用 halt_on_error=1:detect_leaks=1，UBSAN 使用 halt_on_error=1:print_stacktrace=1；未关闭 LSAN、未增加 suppression。该 profile 依照既有合同排除 interposer/preload 专用集合，其通过不作为 interposer 零分配证明。

三个 compile_commands 均确认 AETHERMIND_TEST_NUMERICAL_DIR 仅定义在新增数值测试 TU；该 TU 无 -mavx2/-mfma。baseline 正式构建排除可选实现，同时 O1/O2 的模型输出符合相同外部数值合同。

## 4. 失败检测与门禁证据

- C++ 比较器检查扰动 logits、KV、NaN、缺元素、选行顺序、watermark 和缺 stage；错误包含首个差异坐标。
- 真实权重扰动仅修改临时克隆中 model.norm.weight 的一个 F32。测试要求 Load、Compile、Execute 成功后，独立数值比较在 prefill logits 失败，避免将格式拒绝冒充数值检测。
- 46 个 stdlib 测试覆盖缺/重复记录、错误 dtype/shape/offset、NaN/Inf、位置与 KV 前缀、模型身份、摘要、路径/符号链接及 candidate 差异流程。candidate 有变化即需审阅，误差容限内的变化也不自动晋升。
- 7 个实际隔离 CMake/CTest 探针符合预期：numerical assets 独立于 admission；optional 缺失明确 Disabled/missing_numerical_assets；required 缺失时 fixture 失败、依赖 Not Run；未构建、空 discovery、零 numerical case 均失败；相对数值路径在配置时拒绝。探针 body 为 GTest 协议 dummy，证明门禁行为；数值正确性依据上表正式二进制。
- 3 个实际 exporter 目标路径探针确认拒绝既有输出、源目录重叠和悬空符号链接；冻结 manifest 摘要未变。
- required 模式将原生 required-execution guard 施加到数值用例。配置/编译允许先于资产准备，运行门禁在资产缺失或变化时明确失败；near-limit 有独立 long-context label 和超时。

## 5. 复现

先按 [数值 fixture 使用说明](../../tests/fixtures/inference_numerical/README.md) 准备并核验独立参考数据。默认路径为 .models/inference-numerical，可用绝对 AETHERMIND_TEST_NUMERICAL_DIR 覆盖。

```bash
python3 tools/verify_llama_numerical_reference.py \
  --root .models/inference-numerical \
  --expected tests/fixtures/inference_numerical/checksums.json
python3 tests/unit/inference/test_verify_llama_numerical_reference.py -q

cmake --preset validation-avx2
cmake --build --preset validation-avx2
ctest --preset validation-numerical
ctest --test-dir build/validation/validation-avx2 \
  -L '^validation$' -LE '^isa-avx2$' -j2 --no-tests=error --output-on-failure

cmake --preset validation-baseline
cmake --build --preset validation-baseline
ctest --preset validation-baseline
ctest --test-dir build/validation/validation-baseline \
  -L '^numerical-model$' -j2 --no-tests=error --output-on-failure

cmake --preset validation-asan-ubsan
cmake --build --preset validation-asan-ubsan
ctest --preset validation-asan-ubsan
ASAN_OPTIONS=halt_on_error=1:detect_leaks=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
ctest --test-dir build/validation/validation-asan-ubsan \
  -L '^numerical-model$' -j2 --no-tests=error --output-on-failure
```

本次配置复用既有 build/_deps/spdlog-src，通过 FETCHCONTENT_SOURCE_DIR_SPDLOG 指定固定的 spdlog 1.17.0 源码；参考依赖安装在隔离开发环境，没有安装为系统或核心库依赖。validation-numerical 的 JUnit 文件独立于 ISA 验证报告。

## 6. 证据与未验证范围

原始聚焦日志/XML、CTest 日志、重复导出 diff、参考环境依赖表、编译合同与负面探针保存在 gitignored build/validation/numerical-artifacts。run-manifest.json 记录本次 dirty HEAD、相关源码与实际二进制/库/compile_commands 的摘要；各正式 preset 目录保存 CMakeCache、build-numerical.log、compile_commands 与 JUnit。ctest-contract/verified/results.json 和 exporter-contract/results.json 保存失败探针的命令、状态及对应日志。

C++ 格式检查和 git diff --check 通过。文档检查前后均保留既有 2 个 broken-link 问题及 29 个 warning，均未由本批引入；两个旧链接指向已不存在的 src/function.cpp。

本批验证 standard RoPE、所列 FP32 模型/形状及同一 AVX2/FMA 主机上的 baseline/AVX 构建。没有取得物理无 AVX 主机证据，未重跑 TSAN，也未新增并发 Session 支持。未验收扩展 RoPE、其他精度/家族、质量指标、模型性能、所有优化 pass/kernel 的独立故障定位或全平台矩阵。Q3 的该子项完成不关闭 Q4/Q5 或 Q2 的剩余平台/warnings 工作。
