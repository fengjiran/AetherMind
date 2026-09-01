#ifndef AETHERMIND_UTILS_QUALIFIED_NAME_H
#define AETHERMIND_UTILS_QUALIFIED_NAME_H

/// @file
/// @brief Representation and decomposition of dotted qualified names.

#include "container/array_view.h"
#include "container/string.h"

namespace aethermind {

/// @brief Represents a name composed of dot-separated non-empty components.
///
/// The class caches the complete name, its prefix, and its final component.
/// Constructed names are immutable through this interface, so returned
/// references remain valid while the object is alive and is not assigned to.
class QualifiedName {
public:
    /// @brief Creates an empty qualified name.
    QualifiedName() = default;

    /// @brief Parses a dotted or bare name into components.
    /// @param name Non-empty dotted name, such as `foo.bar.baz`, or a bare name.
    /// @note Every component must be non-empty. Invalid input triggers
    ///       `AM_CHECK` and terminates the process.
    QualifiedName(const String& name) { // NOLINT
        AM_CHECK(!name.empty());

        // Split the name into components so all cached views share one form.
        size_t start = 0;
        size_t pos = name.find(delimiter_, start);
        while (pos != String::npos) {
            auto atom = name.substr(start, pos - start);
            AM_CHECK(!atom.empty());
            atoms_.emplace_back(atom);
            start = pos + 1;
            pos = name.find(delimiter_, start);
        }

        auto final_atom = name.substr(start);
        AM_CHECK(!final_atom.empty());
        atoms_.emplace_back(final_atom);
        CacheAccessors();
    }

    /// @brief Parses a null-terminated dotted or bare name.
    /// @param name Non-empty name accepted by the `String` constructor.
    /// @note Every component must be non-empty. Invalid input triggers
    ///       `AM_CHECK` and terminates the process.
    QualifiedName(const char* name) : QualifiedName(String(name)) {} // NOLINT

    /// @brief Constructs a qualified name from individual components.
    /// @param atoms Non-empty components. Components must not contain `.`.
    /// @note Invalid components trigger `AM_CHECK` and terminate the process.
    explicit QualifiedName(std::vector<String> atoms) : atoms_(std::move(atoms)) {
        for (const auto& atom: atoms_) {
            AM_CHECK(!atom.empty(), "atom cannot be empty");
            AM_CHECK(atom.find(delimiter_) == String::npos, "delimiter not allowed in atom");
        }
        CacheAccessors();
    }

    /// @brief Appends one component to an existing qualified name.
    /// @param prefix Existing prefix; it may be empty.
    /// @param name Non-empty component that must not contain `.`.
    /// @note Invalid input triggers `AM_CHECK` and terminates the process.
    explicit QualifiedName(const QualifiedName& prefix, String name) {
        AM_CHECK(!name.empty());
        AM_CHECK(name.find(delimiter_) == String::npos);
        atoms_ = prefix.atoms_;
        atoms_.push_back(std::move(name));
        CacheAccessors();
    }

    /// @brief Checks whether this name is a component-wise prefix of another.
    /// @param other Name to compare with.
    /// @return True when every component in this name matches the corresponding
    ///         leading component of `other`.
    AM_NODISCARD bool IsPrefixOf(const QualifiedName& other) const {
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

    /// @brief Returns the complete dotted name.
    /// @return A reference to the cached complete name.
    AM_NODISCARD const String& GetQualifiedName() const {
        return qualified_name_;
    }

    /// @brief Returns the dotted prefix without the final component.
    /// @return A reference to the cached prefix, or an empty string for a
    ///         single-component name.
    AM_NODISCARD const String& GetPrefix() const {
        return prefix_;
    }

    /// @brief Returns the final component of the name.
    /// @return A reference to the cached final component, or an empty string for
    ///         an empty qualified name.
    AM_NODISCARD const String& GetName() const {
        return name_;
    }

    /// @brief Returns the individual name components in order.
    /// @return A reference to the cached component vector.
    AM_NODISCARD const std::vector<String>& GetAtoms() const {
        return atoms_;
    }

    /// @brief Compares two qualified names by their complete dotted form.
    /// @param other Name to compare with.
    /// @return True when both complete names are equal.
    bool operator==(const QualifiedName& other) const {
        return qualified_name_ == other.qualified_name_;
    }

    /// @brief Checks whether two qualified names differ.
    /// @param other Name to compare with.
    /// @return True when the complete names are different.
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
            const auto prefix_view = make_array_view(atoms_.data(), atoms_.size() - 1);
            prefix_ = join(delimiter_, prefix_view);
        }

        if (!atoms_.empty()) {
            name_ = atoms_.back();
        }
    }

    // The delimiter is part of the serialized qualified-name representation.
    static constexpr char delimiter_ = '.';

    // Components in their original order.
    std::vector<String> atoms_;

    // Cached strings derived from `atoms_`.
    String qualified_name_;
    String name_;
    String prefix_;
};

} // namespace aethermind

/// @brief Provides a standard-library hash for `aethermind::QualifiedName`.
namespace std {
template<>
struct hash<aethermind::QualifiedName> {
    size_t operator()(const aethermind::QualifiedName& name) const noexcept {
        return std::hash<aethermind::String>()(name.GetQualifiedName());
    }
};
} // namespace std

#endif // AETHERMIND_UTILS_QUALIFIED_NAME_H
