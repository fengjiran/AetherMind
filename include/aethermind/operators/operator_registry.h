#ifndef AETHERMIND_OPERATORS_OPERATOR_REGISTRY_H
#define AETHERMIND_OPERATORS_OPERATOR_REGISTRY_H

#include "aethermind/base/status.h"
#include "aethermind/operators/op_type.h"
#include "aethermind/operators/operator.h"

#include <any>
#include <functional>
#include <memory>
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

    /// Register a factory for an OpType. Called at static init.
    /// Returns false if a factory already exists for this OpType.
    static bool Register(OpType op_type, FactoryFunc factory);

    /// Create an Operator instance for the given OpType with the given params.
    /// The params must contain the correct Params struct for this OpType.
    AM_NODISCARD static StatusOr<std::unique_ptr<Operator>> Create(
            OpType op_type,
            const std::any& params);

private:
    static std::unordered_map<OpType, FactoryFunc>& Registry();
};

/// Helper macro for registering an operator.
/// Usage: AM_REGISTER_OPERATOR(OpType::kRmsNorm, RmsNormOp)
#define AM_REGISTER_OPERATOR(op_type, OpClass)                                         \
    namespace {                                                                        \
    static const bool _am_reg_##OpClass = OperatorRegistry::Register(                  \
            op_type,                                                                   \
            [](const std::any& params) -> StatusOr<std::unique_ptr<Operator>> {        \
                try {                                                                  \
                    auto p = std::any_cast<typename OpClass::Params>(params);          \
                    return std::make_unique<OpClass>(p);                               \
                } catch (const std::bad_any_cast&) {                                   \
                    return Status::InvalidArgument("Wrong params type for " #OpClass); \
                }                                                                      \
            });                                                                        \
    }

}// namespace aethermind

#endif
