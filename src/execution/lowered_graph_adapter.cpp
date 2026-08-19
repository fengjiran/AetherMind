#include "execution/lowered_graph_adapter.h"

#include <algorithm>
#include <tuple>

namespace aethermind {

StatusOr<StateAliasPlan> ResolveStateAliasesForExecution(const LoweredGraph& lowered) {
    // The compiler finalizer establishes these invariants. Recheck at the
    // trust boundary so an invalid artifact is surfaced as Internal rather
    // than passed to runtime state binding.
    AM_RETURN_IF_ERROR(ValidateLoweredGraph(lowered));

    StateAliasPlan plan;
    plan.aliases.reserve(lowered.state_aliases().size());
    for (const LoweredStateAlias& alias: lowered.state_aliases()) {
        plan.aliases.push_back({
                .step_index = alias.step_index,
                .input_port = alias.input_port,
                .output_port = alias.output_port,
        });
    }
    std::ranges::sort(plan.aliases,
                      [](const ResolvedStateAlias& lhs,
                         const ResolvedStateAlias& rhs) noexcept {
                          return std::tie(lhs.step_index, lhs.input_port, lhs.output_port) <
                                 std::tie(rhs.step_index, rhs.input_port, rhs.output_port);
                      });
    return plan;
}

}// namespace aethermind
