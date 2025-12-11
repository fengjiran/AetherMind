//
// Created by richard on 10/31/25.
//

#ifndef AETHERMIND_UTILS_QUALIFIED_NAME_H
#define AETHERMIND_UTILS_QUALIFIED_NAME_H

#include "container/array_view.h"
#include "container/string.h"

namespace aethermind {

// Represents a name of the form "foo.bar.baz"
class QualifiedName {
public:
    QualifiedName() = default;

    // `name` can be a dotted string, like "foo.bar.baz", or just a bare name.
    QualifiedName(const String& name) {// NOLINT
        CHECK(!name.empty());

        // split the string into atoms
        size_t start = 0;
        size_t pos = name.find(delimiter_, start);
        while (pos != String::npos) {
            auto atom = name.substr(start, pos - start);
            CHECK(!atom.empty());
            atoms_.emplace_back(atom);
            start = pos + 1;
            pos = name.find(delimiter_, start);
        }

        auto final_atom = name.substr(start);
        CHECK(!final_atom.empty());
        atoms_.emplace_back(final_atom);
        CacheAccessors();
    }

    QualifiedName(const char* name) : QualifiedName(String(name)) {}// NOLINT

    explicit QualifiedName(std::vector<String> atoms) : atoms_(std::move(atoms)) {
        for (const auto& atom: atoms_) {
            CHECK(!atom.empty()) << "atom cannot be empty";
            CHECK(atom.find(delimiter_) == String::npos)
                    << "delimiter not allowed in atom";
        }
        CacheAccessors();
    }

    // name must be a bare name(not dots)
    explicit QualifiedName(const QualifiedName& prefix, String name) {
        CHECK(!name.empty());
        CHECK(name.find(delimiter_) == String::npos);
        atoms_ = prefix.atoms_;
        atoms_.push_back(std::move(name));
        CacheAccessors();
    }

    // Is `this` a prefix of `other`?
    // For example, "foo.bar" is a prefix of "foo.bar.baz"
    NODISCARD bool IsPrefixOf(const QualifiedName& other) const {
        if (atoms_.size() > other.atoms_.size()) {
            return false;
        }

        for (int i = 0; i < atoms_.size(); ++i) {
            if (atoms_[i] != other.atoms_[i]) {
                return false;
            }
        }
        return true;
    }

    // The fully qualified name, like "foo.bar.baz"
    NODISCARD const String& GetQualifiedName() const {
        return qualified_name_;
    }

    // The leading qualifier, like "foo.bar"
    NODISCARD const String& GetPrefix() const {
        return prefix_;
    }

    // The base name, like "baz"
    NODISCARD const String& GetName() const {
        return name_;
    }

    NODISCARD const std::vector<String>& GetAtoms() const {
        return atoms_;
    }

    bool operator==(const QualifiedName& other) const {
        return qualified_name_ == other.qualified_name_;
    }

    bool operator!=(const QualifiedName& other) const {
        return !operator==(other);
    }

private:
    template<typename T, typename = T::iterator>
    String join(char delimiter, const T& v) {
        String res;
        size_t reserve = 0;
        for (const auto& e: v) {
            reserve += e.size() + 1;
        }
        res.reserve(reserve);

        for (int i = 0; i < v.size(); ++i) {
            if (i != 0) {
                res.push_back(delimiter);
            }
            res.append(v[i]);
        }
        return res;
    }

    void CacheAccessors() {
        qualified_name_ = join(delimiter_, atoms_);
        if (atoms_.size() > 1) {
            ArrayView view(atoms_);
            const auto prefix_view = view.slice(0, view.size() - 1);
            prefix_ = join(delimiter_, prefix_view);
        }

        if (!atoms_.empty()) {
            name_ = atoms_.back();
        }
    }

    // default delimiter
    static constexpr char delimiter_ = '.';

    // The actual list of names, like "{foo, bar, baz}"
    std::vector<String> atoms_;

    // Cached accessors, derived from `atoms_`.
    String qualified_name_;
    String name_;
    String prefix_;
};

}// namespace aethermind

// hash function
namespace std {
template<>
struct hash<aethermind::QualifiedName> {
    size_t operator()(const aethermind::QualifiedName& name) const noexcept {
        return std::hash<aethermind::String>()(name.GetQualifiedName());
    }
};
}// namespace std

#endif//AETHERMIND_UTILS_QUALIFIED_NAME_H
