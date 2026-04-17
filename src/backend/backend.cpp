#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_selector.h"
#include "aethermind/operators/op_type.h"

namespace aethermind {

// StatusOr<ResolvedKernel> Backend::ResolveKernelInfo(
//         OpType op_type,
//         const KernelSelector& selector) const noexcept {
//     const KernelFunc fn = ResolveKernel(op_type, selector);
//     if (fn == nullptr) {
//         return Status::NotFound(
//                 "No matching kernel registered: op_type=" +
//                 std::string(ToString(op_type)) +
//                 ", selector=" + ToString(selector));
//     }
//
//     return ResolvedKernel{
//             .op_type = op_type,
//             .fn = fn,
//             .attrs = {},
//             .debug_name = nullptr,
//     };
// }

}// namespace aethermind
