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

class OperatorRegistry {
public:
    using FactoryFunc = std::function<StatusOr<std::unique_ptr<Operator>>(const std::any& params)>;

    static bool Register(OpType op_type, FactoryFunc factory);

    AM_NODISCARD static StatusOr<std::unique_ptr<Operator>> Create(
            OpType op_type,
            const std::any& params);

private:
    static std::unordered_map<OpType, FactoryFunc>& Registry();
};

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
