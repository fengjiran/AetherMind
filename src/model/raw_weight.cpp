#include "aethermind/model/raw_weight.h"
#include "utils/overflow_check.h"

namespace aethermind {

Status ValidateRawWeightView(const RawWeightView& view) {
    if (!view.IsValid()) {
        return Status::InvalidArgument(
                "RawWeightView is not a valid view");
    }

    const int element_bytes = view.dtype.nbytes();
    if (element_bytes <= 0) {
        return Status::InvalidArgument(
                "RawWeightView requires a nonzero element byte size");
    }

    uint64_t numel = 1;
    for (const int64_t dim: view.shape) {
        if (dim < 0) {
            return Status::InvalidArgument(
                    "RawWeightView shape has a negative dimension");
        }

        if (CheckOverflowMul(numel, static_cast<uint64_t>(dim), &numel)) {
            return Status::InvalidArgument(
                    "RawWeightView shape product overflows");
        }
    }

    uint64_t expected_bytes = 0;
    if (CheckOverflowMul(numel, static_cast<uint64_t>(element_bytes),
                         &expected_bytes)) {
        return Status::InvalidArgument(
                "RawWeightView byte size overflows");
    }

    if (view.bytes != expected_bytes) {
        return Status::InvalidArgument(
                "RawWeightView byte size does not match its logical shape");
    }
    return Status::Ok();
}

} // namespace aethermind
