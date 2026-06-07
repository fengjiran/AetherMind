#ifndef AETHERMIND_OPERATORS_OPERATOR_REGISTRY_H
#define AETHERMIND_OPERATORS_OPERATOR_REGISTRY_H

#include "aethermind/base/status.h"
#include "aethermind/operators/op_type.h"
#include "aethermind/operators/operator.h"

#include <any>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace aethermind {

/// Factory for creating Operator instances from OpType.
///
/// Each concrete operator registers its factory function at static init time.
/// The factory accepts an opaque std::any parameter (typically the operator's
/// Params struct) and returns a unique_ptr<Operator>.
class OperatorRegistry {
public:
    using FactoryFunc = std::function<StatusOr<std::unique_ptr<Operator>>(const std::any& params)>;
    using ParamFactoryFunc = std::function<StatusOr<std::any>()>;

    struct Descriptor {
        FactoryFunc factory_{};
        ParamFactoryFunc make_default_params_{};
    };

    /// Register a factory for an OpType.
    AM_NODISCARD static Status Register(OpType op_type, FactoryFunc factory);
    AM_NODISCARD static Status Register(OpType op_type, Descriptor descriptor);

    static bool RegisterOrAbort(OpType op_type, FactoryFunc factory, const char* op_name);
    static bool RegisterOrAbort(OpType op_type, Descriptor descriptor, const char* op_name);

    template<typename OpClass>
    static StatusOr<std::unique_ptr<Operator>> CreateTypedOperator(const std::any& params) {
        using Params = typename OpClass::Params;
        const auto* typed_params = std::any_cast<Params>(&params);
        if (typed_params == nullptr) {
            return Status::InvalidArgument("Wrong params type for operator");
        }
        return std::make_unique<OpClass>(*typed_params);
    }

    /// Create an Operator instance for the given OpType with the given params.
    /// The params must contain the correct Params struct for this OpType.
    AM_NODISCARD static StatusOr<std::unique_ptr<Operator>> Create(
            OpType op_type,
            const std::any& params);

    AM_NODISCARD static StatusOr<std::any> CreateDefaultParams(OpType op_type);

private:
    static std::unordered_map<OpType, Descriptor>& Registry();
    static std::mutex& Mutex();
};

/// Helper macro for registering an operator.
/// Usage: AM_REGISTER_OPERATOR(OpType::kRmsNorm, RmsNormOp)
#define AM_REGISTER_OPERATOR(op_type, OpClass)                               \
    namespace {                                                              \
    static const bool _am_reg_##OpClass = OperatorRegistry::RegisterOrAbort( \
            op_type,                                                         \
            OperatorRegistry::Descriptor{                                    \
                    .factory_ = &OperatorRegistry::CreateTypedOperator<OpClass>, \
                    .make_default_params_ = []() -> StatusOr<std::any> {     \
                        return std::any{typename OpClass::Params{}};         \
                    },                                                       \
            },                                                               \
            #OpClass);                                                       \
    }

}// namespace aethermind

#endif
