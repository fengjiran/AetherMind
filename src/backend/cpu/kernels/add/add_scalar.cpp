// Scalar dispatch for the CPU Add kernel.
//
// Implements AddKernel_Scalar by dispatching on the dtype stored in
// AddKernelArgs and forwarding to the template-based implementations in
// add_scalar_impl.h. The template code is TU-local; explicit instantiations
// for all five supported dtypes live here.

#include "add_internal.h"
#include "add_scalar_impl.h"

namespace aethermind::cpu::detail {

Status AddKernel_Scalar(const AddKernelArgs& args) noexcept {
    const DataType dtype = args.dtype;

    if (dtype == DataType::Float32()) {
        return ExecuteTyped<float>(args);
    }
    if (dtype == DataType::Double()) {
        return ExecuteTyped<double>(args);
    }
    if (dtype == DataType::BFloat(16)) {
        return ExecuteTyped<BFloat16>(args);
    }
    if (dtype == DataType::Int(32)) {
        return ExecuteTyped<int32_t>(args);
    }
    if (dtype == DataType::Int(64)) {
        return ExecuteTyped<int64_t>(args);
    }
    return Status::InvalidArgument(
            "CpuAddKernel scalar dispatch received unsupported dtype");
}

}// namespace aethermind::cpu::detail
