//
// Created by richard on 11/24/25.
//

#ifndef AETHERMIND_INTERNED_STRINGS_H
#define AETHERMIND_INTERNED_STRINGS_H

#include "symbol.h"

namespace aethermind {

#define FORALL_NS_SYMBOLS(_)         \
    _(namespaces, prim)              \
    _(namespaces, cuda)              \
    _(namespaces, attr)              \
    _(namespaces, namespaces)        \
    _(prim, Assign)                  \
    _(prim, BroadcastingChunk)       \
    _(prim, BroadcastSizes)          \
    _(prim, ReductionSizes)          \
    _(prim, Constant)                \
    _(prim, ChunkSizes)              \
    _(prim, ConstantMKLDNNTensor)    \
    _(prim, BroadcastMKLDNNTensors)  \
    _(prim, MKLDNNGroup)             \
    _(prim, MKLDNNHardSwish)         \
    _(prim, MKLDNNHardSigmoid)       \
    _(prim, MKLDNNHardTanh)          \
    _(prim, MKLDNNClamp)             \
    _(prim, StaticRuntimeCopyOuts)   \
    _(prim, Drop)                    \
    _(prim, Eval)                    \
    _(prim, Expand) /* onnx */       \
    _(prim, FusionGroup)             \
    _(prim, CudaFusionGroup)         \
    _(prim, CudaFusionGuard)         \
    _(prim, oneDNNFusionGroup)       \
    _(prim, oneDNNFusionGuard)       \
    _(prim, FunctionalGraph)         \
    _(prim, add_optional)            \
    _(prim, view_copy)               \
    _(prim, permute_copy)            \
    _(prim, reshape_copy)            \
    _(prim, squeeze_copy)            \
    _(prim, t_copy)                  \
    _(prim, transpose_copy)          \
    _(prim, unsqueeze_copy)          \
    _(prim, flatten_copy)            \
    _(prim, expand_copy)             \
    _(prim, expand_as_copy)          \
    _(prim, DifferentiableGraph)     \
    _(prim, TensorExprGroup)         \
    _(prim, TensorExprDynamicGroup)  \
    _(prim, StaticSubgraph)          \
    _(prim, If)                      \
    _(prim, Jump)   /* debug */      \
    _(prim, JumpNZ) /* debug */      \
    _(prim, JumpZ)  /* debug */      \
    _(prim, Load)                    \
    _(prim, Loop)                    \
    _(prim, Param)                   \
    _(prim, PackPadded)  /* onnx */  \
    _(prim, PadPacked)   /* onnx */  \
    _(prim, Placeholder) /* debug */ \
    _(prim, Print)                   \
    _(prim, EmptyListLiteral)        \
    _(prim, LegacyTypedConstructor)  \
    _(prim, PythonOp)                \
    _(prim, IgnoredPythonOp)         \
    _(prim, Reverse)                 \
    _(prim, Return)                  \
    _(prim, ReturnStmt)              \
    _(prim, BreakStmt)               \
    _(prim, ContinueStmt)            \
    _(prim, ComprehensionScope)      \
    _(prim, Store)                   \
    _(prim, AutogradZero)            \
    _(prim, AutogradAnyNonZero)      \
    _(prim, AutogradAllNonZero)      \
    _(prim, AutogradAllZero)         \
    _(prim, Starred)                 \
    _(prim, TupleConstruct)          \
    _(prim, TupleUnpack)             \
    _(prim, TupleIndex)              \
    _(prim, TupleSlice)              \
    _(prim, ListConstruct)           \
    _(prim, ListUnpack)              \
    _(prim, DictConstruct)           \
    _(prim, ModuleContainerIndex)    \
    _(prim, EnumName)                \
    _(prim, EnumValue)               \
    _(prim, StringIndex)             \
    _(prim, NumToTensor)             \
    _(prim, Uninitialized)           \
    _(prim, VarConcat)               \
    _(prim, VarStack)                \
    _(prim, With)                    \
    _(prim, Enter)                   \
    _(prim, Exit)                    \
    _(prim, IfThenElse)              \
    _(prim, Guard)                   \
    _(prim, BailOut)                 \
    _(prim, TypeCheck)               \
    _(prim, RequiresGradCheck)       \
    _(prim, FallbackGraph)           \
    _(prim, FusedConcat)             \
    _(prim, ConstantChunk)           \
    _(prim, MMTreeReduce)            \
    _(prim, MMBatchSide)             \
    _(prim, list)                    \
    _(prim, dict)                    \
    _(prim, min)                     \
    _(prim, max)                     \
    _(prim, abs)

enum class keys : uint32_t {
#define Key(ns, s) ns##_##s,
    FORALL_NS_SYMBOLS(Key)
#undef Key
            num_symbols
};

#define DEFINE_SYMBOL(ns, s)                                   \
    namespace ns {                                             \
    constexpr Symbol s(static_cast<uint32_t>(keys::ns##_##s)); \
    }
FORALL_NS_SYMBOLS(DEFINE_SYMBOL)

#undef DEFINE_SYMBOL

class InternedStrings {
public:
    InternedStrings();

    Symbol symbol(const String& s);

    std::pair<const char*, const char*> string(Symbol sym);

    Symbol ns(Symbol sym);

private:
    struct SymbolInfo {
        Symbol ns;
        String qual_name;
        String unqual_name;
    };

    Symbol _symbol(const String& s);

    std::unordered_map<String, Symbol> string_to_symbol_;
    std::vector<SymbolInfo> symbol_infos_;
    std::mutex mutex_;
};


}// namespace aethermind

#endif//AETHERMIND_INTERNED_STRINGS_H
