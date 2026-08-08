#ifndef AETHERMIND_OPERATORS_OPERATOR_REGISTRY_H
#define AETHERMIND_OPERATORS_OPERATOR_REGISTRY_H

/// @file operator_registry.h
/// @brief Thread-safe factory registry for executable operator instances.

#include "aethermind/base/status.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/op_type.h"
#include "aethermind/operators/operator.h"

#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace aethermind {

/// @brief Factory registry that creates operator instances from `OpType`.
///
/// Each concrete operator registers its factory function at static init time.
/// The factory accepts the typed graph/operator parameter variant and returns
/// a unique_ptr<Operator>.
class OperatorRegistry {
public:
    using FactoryFunc = std::function<StatusOr<std::unique_ptr<Operator>>(const OpParams& params)>;
    using ParamFactoryFunc = std::function<StatusOr<OpParams>()>;

    struct Descriptor {
        FactoryFunc factory_{};
        ParamFactoryFunc make_default_params_{};
    };

    /// @brief Registers the factories associated with an operator type.
    ///
    /// @param op_type Operator type used as the registry key.
    /// @param descriptor Operator and default-parameter factories to register.
    /// @return Ok on success, or an error when the key or descriptor is invalid.
    AM_NODISCARD static Status Register(OpType op_type, Descriptor descriptor);

    /// @brief Registers an operator type and aborts if registration fails.
    ///
    /// @param op_type Operator type used as the registry key.
    /// @param descriptor Factories to register.
    /// @param op_name Static name included in failure diagnostics.
    /// @return True after successful registration.
    static bool RegisterOrAbort(OpType op_type, Descriptor descriptor, const char* op_name);

    /// @brief Creates a concrete operator after validating its parameter variant.
    ///
    /// @tparam OpClass Concrete operator type exposing a nested `Params` type.
    /// @param params Parameter variant expected by `OpClass`.
    /// @return New operator instance, or `kInvalidArgument` for a wrong variant.
    template<typename OpClass>
    static StatusOr<std::unique_ptr<Operator>> CreateTypedOperator(const OpParams& params) {
        using Params = OpClass::Params;
        const auto* typed_params = std::get_if<Params>(&params);
        if (typed_params == nullptr) {
            return Status::InvalidArgument("Wrong params type for operator");
        }
        return std::make_unique<OpClass>(*typed_params);
    }

    /// @brief Creates an operator for a registered type and parameter variant.
    ///
    /// The params must contain the correct Params struct for this OpType.
    ///
    /// @param op_type Registered operator type to construct.
    /// @param params Typed parameters for the requested operator.
    /// @return New operator instance, or an error for an unknown type or invalid parameters.
    AM_NODISCARD static StatusOr<std::unique_ptr<Operator>> Create(
            OpType op_type,
            const OpParams& params);

    /// @brief Creates the default parameter variant for a registered operator.
    ///
    /// @param op_type Registered operator type whose defaults are requested.
    /// @return Default typed parameters, or an error when no factory is registered.
    AM_NODISCARD static StatusOr<OpParams> CreateDefaultParams(OpType op_type);

private:
    static std::unordered_map<OpType, Descriptor>& Registry();
    static std::mutex& Mutex();
};

/// @brief Registers a concrete operator during static initialization.
///
/// `OpClass` must derive from `Operator` and expose a nested `Params` type.
#define AM_REGISTER_OPERATOR(op_type, OpClass)                                   \
    namespace {                                                                  \
    static const bool _am_reg_##OpClass = OperatorRegistry::RegisterOrAbort(     \
            op_type,                                                             \
            OperatorRegistry::Descriptor{                                        \
                    .factory_ = &OperatorRegistry::CreateTypedOperator<OpClass>, \
                    .make_default_params_ = []() -> StatusOr<OpParams> {         \
                        return OpParams{typename OpClass::Params{}};             \
                    },                                                           \
            },                                                                   \
            #OpClass);                                                           \
    }

}// namespace aethermind

#endif
