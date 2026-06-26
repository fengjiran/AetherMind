#include "aethermind/model/graph/state_alias_plan.h"

#include <algorithm>

namespace aethermind {

bool StateAliasPlan::empty() const noexcept {
    return aliases.empty();
}

size_t StateAliasPlan::size() const noexcept {
    return aliases.size();
}

std::span<const ResolvedStateAlias> StateAliasPlan::ForStep(size_t step_index) const noexcept {
    if (aliases.empty()) {
        return {};
    }

    // Aliases are sorted by step_index. Find the contiguous range.
    const auto lower = std::lower_bound(
            aliases.begin(), aliases.end(), step_index,
            [](const ResolvedStateAlias& a, size_t idx) noexcept {
                return a.step_index < idx;
            });
    if (lower == aliases.end() || lower->step_index != step_index) {
        return {};
    }

    const auto upper = std::upper_bound(
            lower, aliases.end(), step_index,
            [](size_t idx, const ResolvedStateAlias& a) noexcept {
                return idx < a.step_index;
            });

    const auto count = static_cast<size_t>(upper - lower);
    return {&*lower, count};
}

}// namespace aethermind
