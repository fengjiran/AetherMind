//
// Created by richard on 11/25/25.
//

#ifndef AETHERMIND_ALIAS_INFO_H
#define AETHERMIND_ALIAS_INFO_H

#include "container/string.h"
#include "symbol.h"
#include "utils/hash.h"

#include <set>
#include <unordered_set>
#include <vector>

namespace aethermind {

/**
 * class AliasInfo
 *
 * Data structure to hold aliasing information for an `Argument`. They can be
 * nested to represent aliasing information on contained types.
 *
 * There is a `beforeSet` which describes the aliasing information before the
 * operator executes, and an `afterSet` that describes aliasing info
 * after execution.
 */
class AliasInfo {
public:
    AliasInfo() = default;

    AliasInfo(bool is_write, const std::set<String>& before_qual_strings,
              const std::set<String>& after_qual_strings) : is_write_(is_write) {
        for (const auto& s: before_qual_strings) {
            before_set_.insert(Symbol::FromQualString(s));
        }

        for (const auto& s: after_qual_strings) {
            after_set_.insert(Symbol::FromQualString(s));
        }
    }

    // Symbol for the set that can alias anything
    static Symbol WildcardSet() {
        static const Symbol wc = Symbol::FromQualString("alias::*");
        return wc;
    }

    void SetIsWrite(bool is_write) {
        is_write_ = is_write;
    }

    bool IsWrite() const {
        return is_write_;
    }

    void AddBeforeSet(const Symbol& sym) {
        before_set_.insert(sym);
    }

    void AddAfterSet(const Symbol& sym) {
        after_set_.insert(sym);
    }

    void AddContainedType(AliasInfo alias_info) {
        contained_types_.push_back(std::move(alias_info));
    }

    const std::unordered_set<Symbol>& GetBeforeSets() const {
        return before_set_;
    }

    const std::unordered_set<Symbol>& GetAfterSets() const {
        return after_set_;
    }

    const std::vector<AliasInfo>& GetContainedTypes() const {
        return contained_types_;
    }

    Symbol GetBeforeSet() const {
        AM_CHECK(before_set_.size() == 1);
        return *before_set_.begin();
    }

    Symbol GetAfterSet() const {
        AM_CHECK(after_set_.size() == 1);
        return *after_set_.begin();
    }

    bool IsWildcardBefore() const {
        return before_set_.contains(WildcardSet());
    }

    bool IsWildcardAfter() const {
        return after_set_.contains(WildcardSet());
    }

private:
    std::unordered_set<Symbol> before_set_;
    std::unordered_set<Symbol> after_set_;
    std::vector<AliasInfo> contained_types_;
    bool is_write_{};
};

inline bool operator==(const AliasInfo& lhs, const AliasInfo& rhs) {
    return lhs.IsWrite() == rhs.IsWrite() &&
           lhs.GetBeforeSets() == rhs.GetBeforeSets() &&
           lhs.GetAfterSets() == rhs.GetAfterSets() &&
           lhs.GetContainedTypes() == rhs.GetContainedTypes();
}

inline std::ostream& operator<<(std::ostream& os, const AliasInfo& alias_info) {
    os << "(";
    bool first = true;
    for (const auto& sym: alias_info.GetBeforeSets()) {
        if (first) {
            first = false;
        } else {
            os << "|";
        }
        os << sym.ToUnQualString();
    }

    if (alias_info.IsWrite()) {
        os << "!";
    }

    if (alias_info.GetBeforeSets() != alias_info.GetAfterSets()) {
        first = true;
        for (const auto& sym: alias_info.GetAfterSets()) {
            if (first) {
                first = false;
            } else {
                os << "|";
            }
            os << sym.ToUnQualString();
        }
    }
    os << ")";
    return os;
}

}// namespace aethermind

namespace std {
template<>
struct hash<aethermind::AliasInfo> {
    size_t operator()(const aethermind::AliasInfo& alias_info) const noexcept {
        auto hash = std::hash<bool>()(alias_info.IsWrite());

        // NOTE: for unordered_set hashes, we couldn't use hash_combine
        // because hash_combine is order dependent. Instead, we choose to
        // use XOR as the combining function as XOR is commutative.
        size_t before_set_hash_seed = 0;
        for (const auto& sym: alias_info.GetBeforeSets()) {
            auto aym_hash = std::hash<aethermind::Symbol>()(sym);
            before_set_hash_seed = before_set_hash_seed ^ aym_hash;
        }

        size_t after_set_hash_seed = 0;
        for (const auto& sym: alias_info.GetAfterSets()) {
            auto aym_hash = std::hash<aethermind::Symbol>()(sym);
            after_set_hash_seed = after_set_hash_seed ^ aym_hash;
        }

        hash = aethermind::hash_combine(hash, before_set_hash_seed);
        hash = aethermind::hash_combine(hash, after_set_hash_seed);
        for (const auto& inner_alias_info: alias_info.GetContainedTypes()) {
            auto inner_hash = std::hash<aethermind::AliasInfo>()(inner_alias_info);
            hash = aethermind::hash_combine(hash, inner_hash);
        }
        return hash;
    }
};
}// namespace std

#endif//AETHERMIND_ALIAS_INFO_H
