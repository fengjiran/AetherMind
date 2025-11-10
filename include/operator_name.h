//
// Created by 赵丹 on 2025/8/12.
//

#ifndef AETHERMIND_OPERATOR_NAME_H
#define AETHERMIND_OPERATOR_NAME_H

#include <cstring>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

namespace aethermind {

class OperatorName final{
public:
    OperatorName(std::string name, std::string overload_name)
        : name_(std::move(name)), overload_name_(std::move(overload_name)) {}

    std::string name() const {
        return name_;
    }

    std::string overload_name() const {
        return overload_name_;
    }

    // Return the namespace of this OperatorName, if it exists.  The
    // returned string_view is only live as long as the OperatorName
    // exists and name is not mutated
    std::optional<std::string_view> GetNamespace() const {
        auto pos = name_.find("::");
        if (pos == std::string::npos) {
            return std::nullopt;
        }
        return std::string_view(name_.data(), pos);
    }

    // Returns true if successfully set the namespace
    bool SetNamespaceIfNotSet(const char* ns) {
        if (GetNamespace().has_value()) {
            return false;
        }

        auto ns_len = strlen(ns);
        auto old_name_len = name_.size();
        name_.resize(ns_len + 2 + old_name_len);

        // Shift current value of name to the end of the new space.
        name_.replace(name_.size() - old_name_len, old_name_len, name_, 0, old_name_len);
        name_.replace(0, ns_len, ns, ns_len);
        name_[ns_len] = ':';
        name_[ns_len + 1] = ':';
        return true;
    }

    friend bool operator==(const OperatorName& lhs, const OperatorName& rhs) {
        return lhs.name_ == rhs.name_ && lhs.overload_name_ == rhs.overload_name_;
    }

    friend bool operator!=(const OperatorName& lhs, const OperatorName& rhs) {
        return !operator==(lhs, rhs);
    }

private:
    std::string name_;
    std::string overload_name_;
};

std::ostream& operator<<(std::ostream& os, const OperatorName& opName);

std::string toString(const OperatorName& opName);

}// namespace aethermind

namespace std {
template<>
struct hash<aethermind::OperatorName> {
    size_t operator()(const aethermind::OperatorName& x) const {
        return std::hash<std::string>()(x.name()) ^ ~std::hash<std::string>()(x.overload_name());
    };
};
}// namespace std
#endif//AETHERMIND_OPERATOR_NAME_H
