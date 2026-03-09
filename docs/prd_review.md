这份 docs/aethermind_prd.md 方向上是对的，但现在更像“愿景文档 + 路线图 + 技术蓝图 + 部分实施要求”的混合体，不是一份足够稳健的 PRD。  
如果按“合理性”看，Phase 1 的 CPU-first 嵌入式 runtime 是合理且可行的；如果按“可执行性”看，Phase 2+ 到 Phase 4 混入了太多远期能力，导致文档内部出现了明显冲突和过度承诺。
总体判断
- Phase 1 方向合理：CPU-first、单请求、确定性、量化推理、C/C++ API，这些和当前仓库基线以及现有计划是一致的，见 docs/aethermind_prd.md:132、docs/aethermind_prd.md:156、docs/aethermind_prd.md:179，也和 .sisyphus/plans/cpu-first-llama-runtime-v1.md:21、.sisyphus/plans/cpu-first-llama-runtime-v1.md:62 一致
- Phase 2+ 作为长期愿景也合理，但更像战略路线，不适合在同一份 PRD 里写成接近承诺的交付项
- 当前最大问题不是“技术路线错”，而是“边界不清、相互打架、验收条件不够严谨”
合理的地方
- CPU-first 嵌入式起步是现实路线，行业上也有对应成功范式；比起一开始就上多机多卡，这条路线更符合当前仓库成熟度
- 对 ammalloc、零分配 decode、确定性 golden tests、Safetensors、量化线性层这些关注点是工程上对的，和现有 repo 的强项相匹配
- 把产品分成 嵌入式 -> 服务化 -> 分布式 -> Agentic 四阶段，本身是一个好的叙事框架
不合理或不够稳的地方
- docs/aethermind_prd.md:33 直接承诺 “C ABI 从 Phase 1 到 Phase 4 保持稳定” 太激进；当前 include/c_api.h:16 还是很薄的底层对象接口，不足以支撑这种长期稳定性承诺
- docs/aethermind_prd.md:304 写“使用 #pragma once”，但仓库当前规范和绝大多数头文件实际都用 include guards，见 AGENTS.md:93、include/c_api.h:5、include/device.h:5
- docs/aethermind_prd.md:302 把 std::expected 放进工程规范，但当前仓库和构建依赖里并没有一个明确的 backport 策略，这会误导实现
- docs/aethermind_prd.md:284 到 docs/aethermind_prd.md:287 的 Phase 1 指标写得过满，尤其 7B INT4, 4-core CPU, <=4GB, >=10 tok/s 组合太紧；单看权重接近可行，但加上 KV cache、workspace、runtime 开销后，这个总内存目标很容易失真
- docs/aethermind_prd.md:292 的 “8x A100 上 7B 模型 TTFT < 50ms” 不仅偏乐观，还缺少 batch、prompt length、cache hit 定义、并发条件和编译/内核配置，难以验证
内部冲突
- Prefix Caching 的 phase 定义互相冲突：图示像 Phase 4，正文又写 Phase 2+，路线图也放在 Phase 2，见 docs/aethermind_prd.md:27、docs/aethermind_prd.md:51、docs/aethermind_prd.md:122、docs/aethermind_prd.md:169、docs/aethermind_prd.md:361
- Speculative Decoding / Multi-LoRA / FP8 也有类似 phase 泄漏，分别在图示、功能需求、里程碑里落在不同阶段，见 docs/aethermind_prd.md:27、docs/aethermind_prd.md:148、docs/aethermind_prd.md:211、docs/aethermind_prd.md:214、docs/aethermind_prd.md:379
- docs/aethermind_prd.md:89 到 docs/aethermind_prd.md:92 把 Token Manager 放到 Phase 2+，但 docs/aethermind_prd.md:180 到 docs/aethermind_prd.md:183 的 Phase 1 又要求 Prompt -> Prefill -> Decode -> Output；这意味着 V1 到底是 token-id API，还是文本 API，文档没有定死
- Phase 1 说是“生产级嵌入式推理库”，但成功标准只列了 3 条，缺少零稳态分配、golden tests、TSAN/ASAN、加载失败语义、KV parity 这类真正决定“可生产”的条件，和 .sisyphus/plans/cpu-first-llama-runtime-v1.md:53 不一致
和仓库现实的距离
- 当前 repo 的强项在 tensor/type、storage、Function/Any、object、ammalloc，不是推理 runtime 本身
- 当前 repo 里基本还没有真正的 loader / KV cache / decode loop / scheduler / serving / CUDA runtime 这些 inference substrate
- 所以这份 PRD 里很多表述更像“目标状态”，而不是“当前项目正在进入的实施状态”；这点文档里应该明确标注，不然读者会误以为 Phase 1 只是收尾，而不是从 runtime 主体开始建设
可行性判断
- Phase 1：可行，但前提是严格收缩成 token-based、单请求、CPU-only、单模型、greedy baseline 的窄 runtime
- Phase 2：可行，但不能简单理解成“在 Phase 1 上自然叠加”；PagedAttention、Chunked Prefill、Continuous Batching、Prefix Cache 会逼出新的 block manager / scheduler / serving boundary
- Phase 3+：作为路线图合理，但不适合写成近期产品承诺；MoE/EP/FP8/多机调度是独立复杂度台阶
- “一个稳定 C ABI 贯穿所有阶段” 这件事风险最高，更现实的表达应该是“版本化、向前兼容优先、必要时通过服务 API 作为稳定边界”
最值得改的 6 点
1. 把文档拆成两层：V1 PRD 和 Long-term Roadmap  
2. 用一张唯一真值表重写 phase 映射，消除 Prefix Cache / Speculative / Multi-LoRA / FP8 的重复归属  
3. 明确 V1 API 边界：token ids in/out 还是 text in/out；如果是后者，tokenizer 必须进 Phase 1  
4. 把 C ABI 永久稳定 改成 versioned compatibility policy  
5. 把性能指标拆成 must-pass gates 和 stretch targets，并写清测试条件  
6. 删除 PRD 里的局部编码规范重复项，直接引用 AGENTS.md；至少修掉 #pragma once 和 std::expected 这两处误导