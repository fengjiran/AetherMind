#include "op_removability_internal.h"
#include "aethermind/operators/operator_schema.h"

namespace aethermind::detail {

bool IsDceRemovableOp(OpType op_type) noexcept {
    const auto schema = GetOperatorSchema(op_type);
    if (!schema.ok()) {
        return false;
    }
    return !schema->traits.has_side_effects && !HasStatefulOutput(*schema);
}

} // namespace aethermind::detail