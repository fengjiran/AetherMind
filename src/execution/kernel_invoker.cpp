#include "aethermind/execution/kernel_invoker.h"

#include <cstddef>

namespace aethermind {

Status InvokeKernel(const ResolvedKernel& kernel,
                    KernelContext& context,
                    std::span<const TensorView> inputs,
                    std::span<const MutableTensorView> outputs) noexcept {
    if (kernel.fn == nullptr) {
        return Status::FailedPrecondition("Cannot invoke a null kernel function");
    }
    if (kernel.params_builder == nullptr) {
        if (kernel.params_size != 0) {
            return Status::FailedPrecondition(
                    "Kernel params_size must be zero when params_builder is null");
        }
        return kernel.fn(context);
    }
    if (kernel.params_size == 0 || kernel.params_size > kMaxKernelParamsSize) {
        return Status::FailedPrecondition("Kernel params_size is outside the supported range");
    }

    alignas(std::max_align_t) std::byte params_storage[kMaxKernelParamsSize];
    AM_RETURN_IF_ERROR(kernel.params_builder(inputs, outputs, params_storage));
    context.kernel_params = params_storage;
    return kernel.fn(context);
}

}// namespace aethermind
