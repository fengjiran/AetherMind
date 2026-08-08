#ifndef AETHERMIND_OPERATOR_NAME_H
#define AETHERMIND_OPERATOR_NAME_H

/// @file operator_name.h
/// @brief Qualified operator names retained for compatibility dispatch paths.

#include "aethermind/base/macros.h"
#include "utils/hash.h"

#include <optional>
#include <string>
#include <string_view>

namespace aethermind {

/// @brief Qualified operator name paired with an optional overload name.
///
/// For example, `aethermind::add` may use overload name `Tensor`.
/// Kept for schema/legacy-dispatch compatibility while the new backend-owned
/// dispatch mainline gradually migrates to OpType-based resolution.
class OperatorName final {
public:
    OperatorName() = default;
    OperatorName(std::string_view name, std::string_view overload_name)
        : name_(name), overload_name_(overload_name) {}

    AM_NODISCARD std::string_view name() const noexcept {
        return name_;
    }

    AM_NODISCARD std::string_view overload_name() const noexcept {
        return overload_name_;
    }

    /// @brief Extracts the first namespace component from the qualified name.
    ///
    /// Examples: "aethermind::add" -> "aethermind",
    ///           "aethermind::nn::linear" -> "aethermind" (first segment only),
    ///           "::add" -> nullopt (global namespace, no qualifier).
    /// @return First namespace component, or `nullopt` for an unqualified or
    ///         explicitly global name. The view is invalidated on destruction.
    AM_NODISCARD std::optional<std::string_view> GetNamespace() const noexcept;

    friend bool operator==(const OperatorName& lhs, const OperatorName& rhs) {
        return lhs.name_ == rhs.name_ && lhs.overload_name_ == rhs.overload_name_;
    }

    // C++20: operator!= is synthesized from operator==, and </>/<=/>=
    // are synthesized from operator<=>. Defines a total order over
    // (name_, overload_name_) consistent with the former operator<.
    friend auto operator<=>(const OperatorName& lhs, const OperatorName& rhs) {
        if (auto cmp = lhs.name_ <=> rhs.name_; cmp != 0) {
            return cmp;
        }
        return lhs.overload_name_ <=> rhs.overload_name_;
    }

private:
    std::string name_;
    std::string overload_name_;
};

std::ostream& operator<<(std::ostream& os, const OperatorName& opName);
std::string ToString(const OperatorName& opName);

}// namespace aethermind

template<>
struct std::hash<aethermind::OperatorName> {
    size_t operator()(const aethermind::OperatorName& x) const noexcept {
        return aethermind::get_hash(x.name(), x.overload_name());
    }
};// std::hash<OperatorName>

#endif// AETHERMIND_OPERATOR_NAME_H
