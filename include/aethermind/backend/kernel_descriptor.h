#ifndef AETHERMIND_BACKEND_KERNEL_DESCRIPTOR_H
#define AETHERMIND_BACKEND_KERNEL_DESCRIPTOR_H

#include "aethermind/backend/kernel_selector.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/op_type.h"

namespace aethermind {

struct KernelDescriptor {
    OpType op_type = OpType::kUnknown;
    KernelSelector selector{};
    KernelFunc kernel_func = nullptr;
    const char* name = nullptr;
    int priority = 0;
};

AM_NODISCARD inline bool IsValidKernelDescriptor(
        const KernelDescriptor& desc) noexcept {
    return desc.op_type != OpType::kUnknown &&
           desc.kernel_func != nullptr &&
           desc.name != nullptr &&
           desc.selector.device_type != kUndefined;
}

AM_NODISCARD inline Status ValidateKernelDescriptor(const KernelDescriptor& descriptor) noexcept {
    if (descriptor.op_type == OpType::kUnknown) {
        return Status::InvalidArgument("Kernel descriptor op_type cannot be kUnknown");
    }

    if (descriptor.kernel_func == nullptr) {
        return Status::InvalidArgument("Kernel descriptor function cannot be null");
    }

    if (descriptor.name == nullptr) {
        return Status::InvalidArgument("Kernel descriptor name cannot be null");
    }

    if (descriptor.selector.device_type == DeviceType::kUndefined) {
        return Status::InvalidArgument("Kernel descriptor device_type cannot be kUndefined");
    }

    return Status::Ok();
}

}// namespace aethermind

#endif
