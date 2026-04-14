//
// Created by richard on 4/15/26.
//

#include "aethermind/backend/dispatcher_bridge.h"

namespace aethermind {

KernelKey MakeKernelKey(DeviceType device_type, const OperatorName& op_name) {
    return KernelKey{
            .device_type = device_type,
            .op_name = op_name,
    };
}

}// namespace aethermind