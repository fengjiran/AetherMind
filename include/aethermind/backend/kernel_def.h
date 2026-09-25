#ifndef AETHERMIND_BACKEND_KERNEL_DEF_H
#define AETHERMIND_BACKEND_KERNEL_DEF_H

/// @file kernel_def.h
/// @brief Backend kernel definition and its validation.
///
/// A `KernelDef` binds an `OpType` to a concrete `KernelFunc` together
/// with its selector, optional CPU feature requirements, and metadata builders.
/// The registry validates every definition with `ValidateKernelDef` before
/// registration.

#include "aethermind/backend/cpu/cpu_capabilities.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/backend/packed_weights.h"
#include "aethermind/base/kernel_selector.h"
#include "aethermind/operators/op_type.h"

#include <bit>
#include <string_view>

namespace aethermind {

/// @brief Backend kernel definition for registration and selection.
///
/// Defines one kernel implementation: its operator type, eligibility
/// selector, optional CPU requirements, entry point, and builders for
/// params and metadata. The definition is validated by
/// `ValidateKernelDef` before it enters the registry.
struct KernelDef {
    /// Operator type handled by this kernel.
    OpType op_type = OpType::kUnknown;

    /// Selector that determines kernel eligibility (device, dtype, layout).
    KernelSelector selector{};

    /// Exact opaque weight layout consumed when selector.weight_format is
    /// kPacked. Plain definitions must leave this empty.
    PackingRecipe packing_recipe{};

    /// CPU execution requirements. They are intentionally separate from the
    /// selector because an instruction set is not a total ordering.
    CpuFeatureSet cpu_requirements{};

    /// Type-erased kernel entry point. Must be non-null for a valid definition.
    KernelFunc kernel_func = nullptr;

    /// Priority for selection; higher value wins, first-registered wins on tie.
    int priority = 0;

    /// Byte size of the params struct; must be 0 when `params_builder` is null.
    size_t params_size = 0;

    /// Optional builder for type-erased kernel params; null when param-less.
    KernelParamsBuilder params_builder = nullptr;

    /// Optional builder for immutable metadata derived from `OpParams`.
    KernelMetadataBuilder metadata_builder = nullptr;

    /// Human-readable kernel name for diagnostics. Borrows storage owned by
    /// the caller: it must reference a string literal or other
    /// static-duration constant, because the registry stores the view without
    /// copying and `ResolvedKernel` hands the same pointer to execution.
    std::string_view name{};
};

/// @brief Validates a kernel definition's invariants.
///
/// Checks operator type, entry point, name, device type, CPU-only feature
/// requirements, and the consistency between `params_builder` and
/// `params_size`.
///
/// @param def Descriptor to validate.
/// @return `Ok` when all invariants hold, otherwise `InvalidArgument` with
///         a diagnostic message.
inline Status ValidateKernelDef(const KernelDef& def) noexcept {
    if (def.op_type == OpType::kUnknown) {
        return Status::InvalidArgument("Kernel definition op_type cannot be kUnknown");
    }

    if (def.kernel_func == nullptr) {
        return Status::InvalidArgument("Kernel definition function cannot be null");
    }

    if (def.name.empty()) {
        return Status::InvalidArgument("Kernel definition name cannot be empty");
    }

    if (def.selector.device_type == DeviceType::kUndefined) {
        return Status::InvalidArgument(
                "Kernel definition device_type cannot be kUndefined");
    }

    if (def.selector.weight_format == WeightFormat::kPacked) {
        if (def.packing_recipe.layout.empty() ||
            def.packing_recipe.alignment < alignof(void*) ||
            !std::has_single_bit(def.packing_recipe.alignment)) {
            return Status::InvalidArgument("Packed kernel definition requires "
                                           "a named recipe with power-of-two alignment");
        }
    } else if (!def.packing_recipe.layout.empty() ||
               def.packing_recipe.alignment != 0) {
        return Status::InvalidArgument("Plain kernel definition cannot "
                                       "declare a packing recipe");
    }

    if (def.selector.device_type != DeviceType::kCPU &&
        !def.cpu_requirements.empty()) {
        return Status::InvalidArgument("Only CPU kernel definitions "
                                       "may declare CPU feature requirements");
    }

    if (def.params_builder != nullptr) {
        if (def.params_size == 0) {
            return Status::InvalidArgument("params_size must be > 0 "
                                           "when params_builder is set");
        }

        if (def.params_size > kMaxKernelParamsSize) {
            return Status::InvalidArgument("params_size exceeds kMaxKernelParamsSize");
        }
    }

    if (def.params_builder == nullptr && def.params_size != 0) {
        return Status::InvalidArgument(
                "params_size must be zero when params_builder is null");
    }

    return Status::Ok();
}

} // namespace aethermind

#endif
