#ifndef AETHERMIND_BACKEND_KERNEL_KEY_H
#define AETHERMIND_BACKEND_KERNEL_KEY_H

#include "device.h"
#include "operator_name.h"
#include "utils/hash.h"

namespace aethermind {

// Migration-only key for the legacy OperatorName-based dispatch path.
// The new dispatch mainline is centered on OpType + KernelSelector and is
// intentionally introduced in parallel during the Batch 1 transition.
struct KernelKey {
    DeviceType device_type = DeviceType::kUndefined;
    OperatorName op_name;

    friend bool operator==(const KernelKey& lhs, const KernelKey& rhs) {
        return lhs.device_type == rhs.device_type && lhs.op_name == rhs.op_name;
    }

    friend bool operator!=(const KernelKey& lhs, const KernelKey& rhs) {
        return !(lhs == rhs);
    }
};

}// namespace aethermind

namespace std {
template<>
struct hash<aethermind::KernelKey> {
    size_t operator()(const aethermind::KernelKey& key) const noexcept {
        return aethermind::get_hash(key.device_type, key.op_name);
    }
};
}// namespace std

#endif
