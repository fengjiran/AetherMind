#include "aethermind/base/kernel_selector.h"

#include <string>

namespace aethermind {

std::string ToString(const KernelSelector& selector) {
    return std::string("KernelSelector{device=") +
           DeviceType2Str(selector.device_type) +
           ", activation_dtype=" +
           std::to_string(selector.act_dtype.bits()) +
           "bit" +
           ", weight_dtype=" +
           std::to_string(selector.weight_dtype.bits()) +
           "bit" +
           ", weight_format=" +
           ToString(selector.weight_format) +
           ", isa=" +
           ToString(selector.isa) +
           ", phase=" +
           ToString(selector.phase) +
           "}";
}

}// namespace aethermind