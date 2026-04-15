//
// Created by 赵丹 on 2025/8/12.
//

#ifndef AETHERMIND_OPERATOR_NAME_H
#define AETHERMIND_OPERATOR_NAME_H

#include "macros.h"
#include "utils/hash.h"

#include <optional>
#include <string>
#include <string_view>

namespace aethermind {

/// Unique identifier for an operator: name (e.g., "aethermind::add") + overload_name (e.g., "Tensor").
class OperatorName final {
public:
    OperatorName() = default;
    OperatorName(std::string_view name, std::string_view overload_name)
        : name_(name), overload_name_(overload_name) {}

    AM_NODISCARD std::string_view name() const noexcept { return name_; }
    AM_NODISCARD std::string_view overload_name() const noexcept { return overload_name_; }

    /// Extracts namespace from name if present ("aethermind::add" -> "aethermind").
    /// Returned view is valid only while this OperatorName exists.
    AM_NODISCARD std::optional<std::string_view> GetNamespace() const noexcept;

    friend bool operator==(const OperatorName& lhs, const OperatorName& rhs) {
        return lhs.name_ == rhs.name_ && lhs.overload_name_ == rhs.overload_name_;
    }

    friend bool operator!=(const OperatorName& lhs, const OperatorName& rhs) {
        return !(lhs == rhs);
    }

    friend bool operator<(const OperatorName& lhs, const OperatorName& rhs) {
        return lhs.name_ < rhs.name_ ||
               (lhs.name_ == rhs.name_ && lhs.overload_name_ < rhs.overload_name_);
    }

private:
    std::string name_;
    std::string overload_name_;
};

std::ostream& operator<<(std::ostream& os, const OperatorName& opName);
std::string ToString(const OperatorName& opName);

}// namespace aethermind

namespace std {
template<>
struct hash<aethermind::OperatorName> {
    size_t operator()(const aethermind::OperatorName& x) const noexcept {
        return aethermind::get_hash(x.name(), x.overload_name());
    }
};
}// namespace std

#endif// AETHERMIND_OPERATOR_NAME_H
