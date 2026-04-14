#ifndef AETHERMIND_BACKEND_KERNEL_KEY_H
#define AETHERMIND_BACKEND_KERNEL_KEY_H

#include "device.h"
#include "operator_name.h"

namespace aethermind {

struct KernelKey {
    DeviceType device_type = DeviceType::kUndefined;
    OperatorName op_name{"", ""};

    friend bool operator==(const KernelKey& lhs, const KernelKey& rhs) {
        return lhs.device_type == rhs.device_type && lhs.op_name == rhs.op_name;
    }

    friend bool operator!=(const KernelKey& lhs, const KernelKey& rhs) {
        return !operator==(lhs, rhs);
    }
};

}// namespace aethermind

namespace std {
template<>
struct hash<aethermind::KernelKey> {
    size_t operator()(const aethermind::KernelKey& key) const noexcept {
        return std::hash<aethermind::DeviceType>()(key.device_type) ^
               (std::hash<aethermind::OperatorName>()(key.op_name) << 1);
    }
};
}// namespace std

#endif
