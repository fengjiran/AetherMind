#ifndef AETHERMIND_BACKEND_KERNEL_STATIC_REGISTRATION_H
#define AETHERMIND_BACKEND_KERNEL_STATIC_REGISTRATION_H

#include "aethermind/backend/kernel_registry.h"
#include "utils/logging.h"

namespace aethermind::kernel_registration_detail {

inline Status RegisterKernel(const KernelDescriptor& descriptor) {
    return KernelRegistry::Global().Register(descriptor);
}

}// namespace aethermind::kernel_registration_detail

#define AM_REGISTER_KERNEL(unique_name, ...)                                                                       \
    namespace {                                                                                                    \
    [[maybe_unused]] static const bool _am_kernel_reg_##unique_name = []() -> bool {                               \
        const ::aethermind::Status status = ::aethermind::kernel_registration_detail::RegisterKernel(__VA_ARGS__); \
        AM_CHECK(status.ok(), "Kernel registration failed: {}", status.ToString().c_str());                        \
        return true;                                                                                               \
    }();                                                                                                           \
    }

#endif
