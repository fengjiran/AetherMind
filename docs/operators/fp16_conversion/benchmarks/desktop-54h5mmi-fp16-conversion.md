# FP16 批量转换基准记录 — DESKTOP-54H5MMI

| 项 | 值 |
|---|---|
| CPU | Intel Core Ultra 9 285H；WSL 暴露 family 6 / model 197 / stepping 2 |
| OS / kernel | WSL2，Linux 6.6.87.2-microsoft-standard-WSL2，x86-64 |
| Compiler / build | GCC 14.2.0，Release `-O3 -DNDEBUG`，C++20；ISA 文件独立 `-fno-lto` |
| ISA | 本机 OS-usable AVX/F16C/AVX2；逐轮 effective features 见 model raw JSON |
| Affinity / threads | 单线程，guest CPU 0；不据此区分物理 P/E core 或 SMT |
| 环境限制 | microcode 暴露为 `0xffffffff`；物理内存通道、host governor/turbo 不可确认；不做跨机器结论 |

## F16C widening 与 AVX2 整数 narrowing

- commit：`4055338583` + dirty；baseline/candidate 为同一实现构建，差别仅是 prepared 函数表。baseline 是本次 portable scalar batch，不是对旧提交二进制的测量。
- 命令：`cmake --build cmake-build-release --target aethermind_unit_tests aethermind_benchmark -j 8`；`aethermind_benchmark --benchmark_filter=BM_Fp16 --benchmark_min_time=0.02s --benchmark_repetitions=9 --benchmark_enable_random_interleaving=true --benchmark_out_format=json`，以 CPU affinity 0 启动。每轮完整参数保存在 raw command JSON。
- 比较：先做三组独立进程 scalar/selected 交错 A/B，每 case 五次 repetitions；观察到进程间明显偏移，保留全部数据，再补充两轮同进程随机交错，各九次 repetitions。下表报告补充实验中每轮 CPU-time median 的范围，不采用最好单次结果。

| Case，N = 1,048,576，offset = 0 | Scalar median | Selected median | 每轮比值 |
|---|---:|---:|---:|
| checked FP16→FP32 | 1.153–1.180 ms | 0.150–0.174 ms | 6.80–7.66× |
| checked FP32→FP16 | 1.010–1.043 ms | 0.606–0.753 ms | 1.39–1.67× |
| `ConvertRawModelWeightsToFp32`，含分配/去重/释放 | 1.214–2.205 ms | 0.261–0.262 ms | 4.64–8.45× |

- 吞吐 bytes 口径为每元素读写合计 6 bytes，不等于实际 DRAM traffic。小输入、offset=1 和全部 repetitions 也在 raw 中；权重准备的较宽波动范围不能压缩成通用加速倍数。
- 正确性：Release 255 个聚焦测试通过，包括现有四组独立 FP32 trajectory/Generate 比较；新增 callback 测试证明 Load 使用 backend 表、每个唯一 F16 tensor 调用一次、自定义 backend 默认 scalar。widening 覆盖 65,536 个位模式在各 lane 的排列；narrowing 覆盖全部有限 midpoint 和 200,003 个随机 FP32 位模式。
- 构建/安全：scalar、仅 F16C、仅 AVX2 各 12 个测试通过；ASan/UBSan 的 12 个测试通过。当前沙箱 ptrace 阻止 LeakSanitizer 退出检查，ASan profile 设置 `ASAN_OPTIONS=detect_leaks=0`，未验证泄漏。四个 profile 和主构建的 ISA manifest 检查通过；13 个 verifier 正反例通过。
- raw：`benchmark-results/operators/fp16_conversion/20261110T030819Z_4055338583_desktop-54h5mmi_vector/`。`metadata.json`、`batch-commands.json`、`interleaved-commands.json`、各 JSON/repetitions、build logs 和 XML 均保留在本地工作区；未发布外部 artifact。
- 结论：Accepted，限定本机 batch primitive 和以下 P1 准备路径。没有 native 16-bit consumer 的 narrowing 不推导整模型或生成吞吐收益。AVX-512/SAE 和其他 ISA 后续独立验收。

## 真实 HF 加载、编译与准备调用链

- commit：同上；显式 C++ `ExecutableModel::Load`，CPU O1/plain，`WeightPolicy::kConvertToFp32`。
- checkpoint：工作区现有 `tiny-random-LlamaForCausalLM`（H=16、I=64、L=2、vocab=32000）的 21 个 F32 tensor，经 Python `struct` RNE 转成 F16，模型拓扑不变。F16 payload 2,064,544 bytes；checkpoint 来源及转换后 SHA-256 见 `model-fixture.json`。这是小型随机 HF 测试模型，不代表生产规模或模型质量。
- baseline：仅在 Runtime policy 禁用 F16C，复用新 scalar batch；candidate：允许 F16C。其余 FP32 kernel policy 相同，未改变 preset 或 precision。
- 边界：Runtime 创建与 metadata 映射在计时外；三次 warmup 后每进程 25 次 `Load→Convert→Compile→Prepare→model 析构` wall-time。文件系统 cache 已预热，不是冷启动 I/O。三组 A/B 独立进程，CPU affinity 0。
- 命令与制品：raw 内 `model_preparation.cpp`、`model-probe-compile.json` 和 `model-commands.json` 给出完整源码、编译参数、模型目录与执行参数。

| A/B 组 | Scalar median | F16C median |
|---|---:|---:|
| 0 | 2.954 ms | 1.777 ms |
| 1 | 2.583 ms | 1.615 ms |
| 2 | 2.704 ms | 1.652 ms |

- 三组 median 的中位数为 2.704 ms → 1.652 ms，约 1.64×，wall-time 约下降 38.9%。这里只说明该 checkpoint 的 warm-cache 准备路径；没有测量 token 吞吐或跨平台收益。
- raw：同一 run directory 下的 `model-*.stdout`、`model-*.stderr`、`model-summary.json` 和 fixture hashes。
- 结论：Accepted，F16C 接入 P1 准备路径有该限定范围内的实际调用链证据；保持 scalar 与 feature-policy fallback。
