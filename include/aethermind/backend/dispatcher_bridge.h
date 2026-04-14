//
// Created by richard on 4/15/26.
//
#ifndef AETHERMIND_BACKEND_DISPATCHER_BRIDGE_H
#define AETHERMIND_BACKEND_DISPATCHER_BRIDGE_H

#include "aethermind/backend/kernel_key.h"

namespace aethermind {

KernelKey MakeKernelKey(DeviceType device_type, const OperatorName& op_name);

}// namespace aethermind
#endif
