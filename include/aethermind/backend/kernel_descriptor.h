#ifndef AETHERMIND_BACKEND_KERNEL_DESCRIPTOR_H
#define AETHERMIND_BACKEND_KERNEL_DESCRIPTOR_H

/// @file kernel_descriptor.h
/// @brief Backend kernel descriptor and its validation.
///
/// A `KernelDescriptor` binds an `OpType` to a concrete `KernelFunc` together
/// with its selector, optional CPU feature requirements, and metadata builders.
/// The registry validates every descriptor with `ValidateKernelDescriptor` before
/// registration.

#include "aethermind/backend/cpu/cpu_capabilities.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/kernel_selector.h"
#include "aethermind/operators/op_type.h"

#include <string>

namespace aethermind {

/// @brief Backend kernel descriptor for registration and selection.
///
/// Describes one kernel implementation: its operator type, eligibility
/// selector, optional CPU requirements, entry point, and builders for
/// params and metadata. The descriptor is validated by
/// `ValidateKernelDescriptor` before it enters the registry.
struct KernelDescriptor {
    /// Operator type handled by this kernel.
    OpType op_type = OpType::kUnknown;

    /// Selector that determines kernel eligibility (device, dtype, layout).
    KernelSelector selector{};

    /// CPU execution requirements. They are intentionally separate from the
    /// selector because an instruction set is not a total ordering.
    CpuFeatureSet cpu_requirements{};

    /// Type-erased kernel entry point. Must be non-null for a valid descriptor.
    KernelFunc kernel_func = nullptr;

    /// Priority for selection; higher value wins, first-registered wins on tie.
    int priority = 0;

    /// Byte size of the params struct; must be 0 when `params_builder` is null.
    size_t params_size = 0;

    /// Optional builder for type-erased kernel params; null when param-less.
    KernelParamsBuilder params_builder = nullptr;

    /// Optional builder for immutable metadata derived from `OpParams`.
    KernelMetadataBuilder metadata_builder = nullptr;

    /// Human-readable kernel name for diagnostics and registry lookup.
    std::string name{};
};

/// @brief Validates a kernel descriptor's invariants.
///
/// Checks operator type, entry point, name, device type, CPU-only feature
/// requirements, and the consistency between `params_builder` and
/// `params_size`.
///
/// @param descriptor Descriptor to validate.
/// @return `Ok` when all invariants hold, otherwise `InvalidArgument` with
///         a diagnostic message.
AM_NODISCARD inline Status ValidateKernelDescriptor(const KernelDescriptor& descriptor) noexcept {
    if (descriptor.op_type == OpType::kUnknown) {
        return Status::InvalidArgument("Kernel descriptor op_type cannot be kUnknown");
    }

    if (descriptor.kernel_func == nullptr) {
        return Status::InvalidArgument("Kernel descriptor function cannot be null");
    }

    if (descriptor.name.empty()) {
        return Status::InvalidArgument("Kernel descriptor name cannot be empty");
    }

    if (descriptor.selector.device_type == DeviceType::kUndefined) {
        return Status::InvalidArgument(
                "Kernel descriptor device_type cannot be kUndefined");
    }

    if (descriptor.selector.device_type != DeviceType::kCPU &&
        !descriptor.cpu_requirements.empty()) {
        return Status::InvalidArgument(
                "Only CPU kernel descriptors may declare CPU feature requirements");
    }

    if (descriptor.params_builder != nullptr) {
        if (descriptor.params_size == 0) {
            return Status::InvalidArgument(
                    "params_size must be > 0 when params_builder is set");
        }

        if (descriptor.params_size > kMaxKernelParamsSize) {
            return Status::InvalidArgument("params_size exceeds kMaxKernelParamsSize");
        }
    }

    if (descriptor.params_builder == nullptr && descriptor.params_size != 0) {
        return Status::InvalidArgument(
                "params_size must be zero when params_builder is null");
    }

    return Status::Ok();
}

} // namespace aethermind

#endif
