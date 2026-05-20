---
ModelLoader / Weight Prepack 实施批次方案
总体原则
按你这份设计文档和当前代码现状，最合理的推进方式是：
1. 先打通最窄闭环
   - 先支持 HF 单文件目录
   - 先完成 Reader + Config + Resolver
   - 暂时不碰分片和复杂错误增强
2. 先让加载链路能构造最基础的 ModelInstance
   - 即使一开始 raw-only，也要尽量尽早形成从 artifact 到 ModelInstance 的闭环
   - 然后再叠加 prepack
3. 严格收敛到现有主线
   - 不引入新的正式终态模型对象
   - 不让 execution 感知 HF 细节
   - 所有 packed weights 最终都进 ModelInstance::StorePackedWeights(...)
---
Batch 0：接口骨架与目录落位
目标
先把 loader 子系统的文件骨架和命名落下来，但不急着一次性填满逻辑。
建议新增文件
include/aethermind/model/
  model_load_options.h
  model_loader.h
  model_validator.h
  resolved_tensor_index.h
  model_instance_builder.h
  weight_prepack_planner.h
src/model/
  model_loader.cpp
  model_validator.cpp
  resolved_tensor_index.cpp
  model_instance_builder.cpp
  weight_prepack_planner.cpp
include/aethermind/model/formats/hf/
  hf_directory_reader.h
  hf_config_parser.h
  hf_safetensors_index.h
  hf_tensor_name_resolver.h
src/model/formats/hf/
  hf_directory_reader.cpp
  hf_config_parser.cpp
  hf_safetensors_index.cpp
  hf_tensor_name_resolver.cpp
本 batch 做什么
- 只定义类型和最小 public API
- 明确依赖边界
- 不做复杂实现
关键约束
- 不要在这个阶段引入新 runtime object
- ModelLoader::Load(...) 的返回类型直接定死为：
  - StatusOr<std::unique_ptr<ModelInstance>>
  验证
- 头文件能编译
- lsp_diagnostics 干净
风险
- 低
- 主要是接口边界定义是否干净
---
Batch 1：HF 单文件 Reader 最小闭环
目标
支持：
<model_dir>/
  config.json
  model.safetensors
并能完成：
- 目录识别
- 文件存在性检查
- 基础 safetensors 文件打开
- tensor 表建立
建议实现内容
1. ModelLoadOptions
建议最小字段先只放：
- std::filesystem::path model_dir
必要时后续再补：
- strict validation 开关
- 是否要求 prepack
- backend hint
2. HfDirectoryReader
职责：
- 校验目录存在
- 识别 config.json
- 识别 model.safetensors
- 返回“单文件布局”
3. HfSafetensorsIndex
先只支持单文件 safetensors header 解析
- 解析 header length
- 解析 header JSON
- 建立：
  - tensor name
  - dtype
  - shape
  - data offsets
4. tensor table / raw view 聚合接口
输出一个统一 tensor 表，例如：
- std::unordered_map<std::string, RawTensorView>
这里的 RawTensorView 建议定义成：
- 数据指针
- dtype
- shape
- byte size
- backing owner 引用
暂时不做
- sharded safetensors
- 真正的 mmap 优化（如果先用一次性只读加载也可以）
- 复杂错误上下文增强
验证
- 单元测试：能识别合法目录
- 单元测试：缺 config.json 报错
- 单元测试：缺 model.safetensors 报错
- 单元测试：非法 safetensors header 报错
成功标准
- 能从单文件 safetensors 建出 tensor name -> raw tensor view 表
---
Batch 2：config.json 解析 + 基础 Validator
目标
先完成最小模型支持判定，保证 loader 不会把明显不支持的模型往后传。
建议实现内容
1. HfConfigParser
解析最小必要字段，建议先只支持 Llama-family dense decoder-only 所需信息，例如：
- architecture / model_type
- hidden size
- intermediate size
- num hidden layers
- num attention heads
- vocab size
- rms norm epsilon
- tie word embeddings（如需要）
2. ModelValidator::ValidateConfig()
先做 config 级校验：
- 是否属于支持的 architecture
- 是否满足 dense decoder-only 范围
- 关键字段是否存在且合法
3. ModelValidator::ValidateTensorSet()
先做最基本的 tensor 完整性检查：
- embed_tokens
- final_norm
- 每层：
  - q/k/v/o
  - gate/up/down
  - input/post_attn norm
  暂时不做
- 所有 shape 的完全严校验
- 所有 dtype 组合
- tied embedding 的复杂语义处理
验证
- config 缺字段时报错
- architecture 不支持时报错
- 缺关键 tensor 报错
成功标准
- 能阻止不支持模型进入 resolver / builder
---
Batch 3：Resolver / ResolvedTensorIndex
目标
把字符串 tensor 名称解析成强类型结构，彻底建立后续不再字符串查表的基础。
建议实现内容
1. ResolvedTensorIndex
按文档定义的数据结构落地：
- ResolvedAttentionRawWeights
- ResolvedFfnRawWeights
- ResolvedNormRawWeights
- ResolvedDecoderLayerRaw
- ResolvedTensorIndex
2. HfTensorNameResolver
实现 Llama dense 命名解析：
- model.layers.{i}.self_attn.q_proj.weight
- ...k_proj.weight
- ...v_proj.weight
- ...o_proj.weight
- ...mlp.gate_proj.weight
- ...mlp.up_proj.weight
- ...mlp.down_proj.weight
- norm / embed / final_norm / lm_head
3. ResolvedTensorIndexBuilder
输入：
- config
- tensor table
输出：
- ResolvedTensorIndex
关键原则
resolver 完成后：
- 后续 prepack / builder 禁止继续依赖字符串查表
验证
- 单元测试：正确解析 1 层、2 层、多层模型命名
- 单元测试：某一层缺 q_proj 时明确报错
- 单元测试：层数不完整时报错
成功标准
- 任何后续逻辑都只依赖 ResolvedTensorIndex
---
Batch 4：ModelInstanceBuilder 的 raw-only 闭环
目标
先不做 prepack planner，先验证：
- reader + validator + resolver 的结果，能否组装成一个可长期持有 backing 的 ModelInstance
为什么要单独做这一批
因为最大的生命周期风险在这里：
> raw tensor view 的 backing 必须被 ModelInstance 持有，否则 reader 析构后全悬挂。
> 所以应该尽早单独验证这个问题。
> 建议实现内容
1. ModelInstanceBuilder::Create(...)
输入：
- config
- resolved index
- raw tensor backing owner / tensor table backing
输出：
- std::unique_ptr<ModelInstance>
2. ModelInstance 生命周期扩展
当前 ModelInstance 似乎主要持有 BackendSidecar。  
这一批可能需要给 ModelInstance 增加：
- backing owner 持有能力
- raw 常量权重视图持有能力
- 模型静态元数据持有能力
但要注意：
- 仍然不能演变成一个新的并列 runtime object
- 只是把 ModelInstance 补充成设计文档要求的正式终态
验证
- reader 销毁后，ModelInstance 内部 raw view 仍然有效
- 生命周期单元测试
- 不发生悬挂指针
成功标准
- 可以构造一个“raw-only 但合法”的 ModelInstance
---
Batch 5：WeightPrepackPlanner + CPU prepack 接入
目标
把需要 prepack 的线性层权重真正纳入加载链路。
建议实现内容
1. WeightPrepackPlanner::BuildRequests()
根据：
- config
- resolved tensor index
- backend
- registry
生成 prepack 请求列表，至少覆盖：
- q_proj
- k_proj
- v_proj
- o_proj
- gate_proj
- up_proj
- down_proj
- optional lm_head
2. WeightPrepackPlanner::Plan()
先做最小 planner：
- 收集 request
- 计算总大小/对齐需求
即使一开始不做复杂 arena，也要先把 planning 接口立住。
3. WeightPrepacker::Run(...)
把 planner 输出逐项接到：
- CpuWeightPrepacker::Pack(...)
- ModelInstance::StorePackedWeights(...)
4. 保持 raw 的项
以下不要 prepack：
- embed_tokens.weight
- RMSNorm
- final_norm
验证
- ModelInstance 中能查到每个预期 packed 权重
- duplicate selector/op_type 会被 sidecar 拒绝
- execution 能从 FindPackedWeights(...) 取到 packed data
成功标准
- 完成真正意义上的“加载期预打包接入主线”
---
Batch 6：顶层 ModelLoader::Load(...) 闭环
目标
把前面所有层串起来，形成正式入口。
顶层流程
Load(options, backend, registry)
  -> 识别 HF 目录
  -> 解析 config
  -> 建立 tensor table
  -> Validate
  -> Resolve
  -> Build ModelInstance
  -> Prepack + StorePackedWeights
  -> return ModelInstance
这一步要保证
- 对外唯一正式入口是 ModelLoader::Load(...)
- 返回唯一正式终态：std::unique_ptr<ModelInstance>
验证
- 单文件 HF 目录加载成功
- 最窄 integration test 跑通
- ExecutionPlanBuilder 可以直接消费结果
成功标准
- 单文件模型完整加载闭环成立
---
Batch 7：Sharded safetensors 支持
目标
支持：
config.json
model.safetensors.index.json
model-00001-of-0000N.safetensors
...
建议实现内容
1. HfSafetensorsIndex::Load()
解析：
- metadata.total_size
- weight_map
2. shard 聚合
建立：
- tensor name -> shard
- shard -> tensors reverse map
3. 文件一致性检查
- shard 文件缺失
- shard map 冲突
- tensor 重复归属
- 路径非法
验证
- 单元测试：shard 缺失报错
- 单元测试：合法 sharded 目录成功建表
- 单元测试：跨 shard tensor 查找正确
成功标准
- 分片目录与单文件目录拥有统一的 reader 输出接口
---
Batch 8：错误处理增强 + 观测性 + 回归测试收口
目标
把 loader 从“能工作”提升到“可维护、可定位问题”。
建议实现内容
- artifact error 上下文增强
- validation error 上下文增强
- resolve error 指出 layer id / tensor name
- prepack error 指出 op type / selector / backend
- profiling / timing 埋点（如果当前项目已有约定）
- 最终 loader 测试矩阵补全
验证
- 错误消息包含：
  - model path
  - tensor name
  - layer id
  - shape
  - dtype
  - backend
  成功标准
- 加载失败时可快速定位问题
- 测试覆盖达到可持续迭代水平
---
我建议的实际执行顺序
如果按“最小闭环优先”，我建议不是完全照文档 checklist 顺序机械推进，而是这样：
推荐顺序
1. Batch 0：接口骨架
2. Batch 1：HF 单文件 Reader
3. Batch 2：Config Parser + 基础 Validator
4. Batch 3：Resolver / ResolvedTensorIndex
5. Batch 4：ModelInstanceBuilder raw-only 生命周期闭环
6. Batch 5：Prepack Planner + CPU prepack 接入
7. Batch 6：ModelLoader::Load(...) 串联
8. Batch 7：Sharded safetensors
9. Batch 8：错误增强与收口
---
为什么我建议先做 Batch 4 再做完整 prepack
因为当前最大的真实风险不是“planner 不够优雅”，而是：
> raw backing 生命周期如果没设计对，后面所有 loader 工作都不稳。
> 先把 ModelInstanceBuilder 和 backing ownership 收敛好，再接 prepack，会更稳。
---
每个 batch 的提交建议
为了符合你之前一直强调的“不要把所有内容堆到一个 commit”，我建议：
- 每个 batch 至少 1 个 commit
- 复杂 batch 拆成 2 个 commit
例如：
- Batch 1  
  - commit A: HF directory discovery
  - commit B: safetensors single-file header parsing
- Batch 2  
  - commit A: config parser
  - commit B: basic validator
- Batch 5  
  - commit A: prepack planner request building
  - commit B: prepack run + store into ModelInstance
---
当前最适合先做的第一批
如果你现在要我开始推进，我建议从：
Batch 0 + Batch 1
原因：
- 风险最低
- 不会过早碰生命周期复杂点
- 能最快验证 HF 本地目录读取方向是否顺手
- 也是后续所有工作最自然的基础

---







1. 核心目标
- 支持分片布局：识别并加载包含 model.safetensors.index.json 和多个 model-0000X-of-0000Y.safetensors 分片文件的模型目录。
- 零拷贝与生命周期安全：继续利用 mmap（Memory Mapping）实现多分片文件的零拷贝加载，并通过 std::shared_ptr<const RawStorage> 确保每个张量视图（RawWeightView）的底层内存映射生命周期安全。
- 严格的正确性校验：
- 校验分片索引文件格式。
- 校验所有分片文件是否存在。
- 校验分片文件中的张量是否与索引中的 weight_map 严格一致（防止张量跨分片冲突、重复或缺失）。
2. 架构设计与职责划分
2.1 新增 HfSafetensorsIndex 类
负责解析 model.safetensors.index.json 文件。
- 输入：索引文件路径。
- 输出：weight_map（张量名称 -> 分片文件名的映射表）和唯一分片文件名集合。
- 实现方式：继承/利用现有的 hf::HfJsonReader 编写轻量、高效的 JSON 解析器，避免引入外部重量级 JSON 库。
2.2 更新 HfDirectoryReader
- InspectDirectory：
- 当检测到 model.safetensors.index.json 时，不再返回 kUnimplemented 错误，而是正确识别为 HfDirectoryLayout::kShardedSafetensors 布局。
- LoadRawWeightTable：
- 如果是单文件布局，保持原有逻辑。
- 如果是分片布局：
1. 调用 HfSafetensorsIndex::Load 解析索引文件。
2. 遍历所有唯一的分片文件，使用 HfSafetensorsFile::Open 进行内存映射。
3. 遍历每个分片文件中的张量，校验其是否在索引中、是否归属于当前分片、是否存在重复，并将其 RawWeightView 聚合到统一的 RawWeightTable 中。
4. 最终校验索引中列出的所有张量是否都已成功加载。
3. 详细实施步骤与代码设计
Step 1: 新建 include/aethermind/model/formats/hf/hf_safetensors_index.h
定义分片索引类：
#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_SAFETENSORS_INDEX_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_SAFETENSORS_INDEX_H
#include "aethermind/base/status.h"
#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>
namespace aethermind {
class HfSafetensorsIndex {
public:
    static StatusOr<HfSafetensorsIndex> Load(const std::filesystem::path& index_path);
    AM_NODISCARD const std::unordered_map<std::string, std::string>& GetWeightMap() const noexcept {
        return weight_map_;
    }
    AM_NODISCARD std::unordered_set<std::string> GetUniqueShards() const;
private:
    explicit HfSafetensorsIndex(std::unordered_map<std::string, std::string> weight_map) noexcept
        : weight_map_(std::move(weight_map)) {}
    std::unordered_map<std::string, std::string> weight_map_{};
};
} // namespace aethermind
#endif // AETHERMIND_MODEL_FORMATS_HF_HF_SAFETENSORS_INDEX_H
Step 2: 新建 src/model/formats/hf/hf_safetensors_index.cpp
利用 HfJsonReader 实现索引解析：
#include "aethermind/model/formats/hf/hf_safetensors_index.h"
#include "aethermind/model/formats/hf/hf_json_reader.h"
#include "aethermind/model/formats/hf/hf_utils.h"
namespace aethermind {
namespace {
class SafetensorsIndexParser final : public hf::HfJsonReader {
public:
    explicit SafetensorsIndexParser(std::string_view input) noexcept
        : HfJsonReader(input) {}
    StatusOr<std::unordered_map<std::string, std::string>> Parse() {
        if (!TryConsume('{')) {
            return Status::InvalidArgument("Safetensors index must be a JSON object");
        }
        std::unordered_map<std::string, std::string> weight_map;
        SkipWhitespace();
        if (!TryConsume('}')) {
            while (true) {
                const auto key = ParseString();
                if (!key.ok()) {
                    return key.status();
                }
                AM_RETURN_IF_ERROR(Expect(':', "after safetensors index key"));
                
                if (*key == "weight_map") {
                    auto parsed_map = ParseWeightMap();
                    if (!parsed_map.ok()) {
                        return parsed_map.status();
                    }
                    weight_map = std::move(*parsed_map);
                } else {
                    AM_RETURN_IF_ERROR(SkipValue());
                }
                SkipWhitespace();
                if (TryConsume('}')) {
                    break;
                }
                AM_RETURN_IF_ERROR(Expect(',', "between safetensors index fields"));
            }
        }
        SkipWhitespace();
        if (!AtEnd()) {
            return Status::InvalidArgument("Safetensors index contains trailing JSON content");
        }
        return weight_map;
    }
private:
    StatusOr<std::unordered_map<std::string, std::string>> ParseWeightMap() {
        if (!TryConsume('{')) {
            return Status::InvalidArgument("weight_map must be a JSON object");
        }
        std::unordered_map<std::string, std::string> weight_map;
        SkipWhitespace();
        if (!TryConsume('}')) {
            while (true) {
                const auto key = ParseString();
                if (!key.ok()) {
                    return key.status();
                }
                AM_RETURN_IF_ERROR(Expect(':', "after weight_map key"));
                
                const auto value = ParseString();
                if (!value.ok()) {
                    return value.status();
                }
                weight_map.emplace(*key, *value);
                SkipWhitespace();
                if (TryConsume('}')) {
                    break;
                }
                AM_RETURN_IF_ERROR(Expect(',', "between weight_map fields"));
            }
        }
        return weight_map;
    }
};
} // namespace
StatusOr<HfSafetensorsIndex> HfSafetensorsIndex::Load(const std::filesystem::path& index_path) {
    auto text = hf::ReadFileText(index_path);
    if (!text.ok()) {
        return text.status();
    }
    auto weight_map = SafetensorsIndexParser(*text).Parse();
    if (!weight_map.ok()) {
        return Status(weight_map.status().code(),
                      hf::FormatPathMessage(weight_map.status().message(), index_path));
    }
    return HfSafetensorsIndex(std::move(*weight_map));
}
std::unordered_set<std::string> HfSafetensorsIndex::GetUniqueShards() const {
    std::unordered_set<std::string> shards;
    for (const auto& [_, shard] : weight_map_) {
        shards.insert(shard);
    }
    return shards;
}
} // namespace aethermind
Step 3: 修改 src/model/formats/hf/hf_directory_reader.cpp
1. InspectDirectory：
// 替换原来的 has_sharded_index 错误返回：
if (has_sharded_index) {
    return HfDirectoryDescriptor{
            .layout = HfDirectoryLayout::kShardedSafetensors,
            .model_dir = model_dir,
            .config_path = config_path,
            .safetensors_path = {},
            .safetensors_index_path = safetensors_index_path,
    };
}
2. LoadRawWeightTable：
实现分片文件的加载、校验与聚合逻辑（如前文设计所示）。
4. 测试策略
在 tests/unit/model/test_hf_directory_reader.cpp 中新增以下测试用例：
1. InspectShardedDirectory：验证能正确识别分片布局目录。
2. LoadsShardedRawWeightTable：使用临时目录创建模拟的 model.safetensors.index.json 和两个分片 safetensors 文件，验证能否成功加载并合并所有张量。
3. RejectsMissingShardFile：验证当索引中列出的某个分片文件缺失时，能正确报错（NotFound）。
4. RejectsMismatchedShardTensor：验证当分片文件中包含的张量与索引中指定的归属分片不一致时，能正确报错。
5. RejectsDuplicateTensorAcrossShards：验证当不同分片中出现同名张量时，能正确报错。
6. RejectsMissingTensorInShard：验证当索引中列出的张量在实际分片文件中不存在时，能正确报错。
5. 风险与复杂度评估
- 内存开销：由于 HfSafetensorsFile::Open 内部使用 mmap，即使同时打开数十个分片文件，也只会占用虚拟内存地址空间，物理内存仅在实际读取张量数据时按需加载（Lazy Loading），因此内存开销极低且非常安全。
- 生命周期安全：RawWeightView 内部的 storage 智能指针会延长底层 mmap 的生命周期，即使 HfSafetensorsFile 临时对象被析构，内存映射依然有效，完全避免了悬挂指针风险。