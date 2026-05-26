#ifndef AETHERMIND_BACKEND_KERNEL_REQUEST_H
#define AETHERMIND_BACKEND_KERNEL_REQUEST_H

#include "aethermind/backend/kernel_selector.h"
#include "aethermind/base/status.h"
#include "aethermind/operators/op_type.h"
#include "macros.h"

#include <string>

namespace aethermind {

struct KernelRequest {
    OpType op_type = OpType::kUnknown;
    KernelSelector selector{};
};

AM_NODISCARD inline Status ValidateKernelRequest(const KernelRequest& request) noexcept {
    if (request.op_type == OpType::kUnknown) {
        return Status::InvalidArgument("Kernel request op_type cannot be kUnknown");
    }

    if (request.selector.device_type == DeviceType::kUndefined) {
        return Status::InvalidArgument("Kernel request device_type cannot be kUndefined");
    }

    return Status::Ok();
}

AM_NODISCARD inline std::string ToString(const KernelRequest& request) {
    return std::string("KernelRequest{op_type=") +
           std::string(ToString(request.op_type)) +
           ", selector=" + ToString(request.selector) + "}";
}

}// namespace aethermind

#endif
