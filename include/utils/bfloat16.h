//
// Created by richard on 9/23/25.
//

#ifndef AETHERMIND_BFLOAT16_H
#define AETHERMIND_BFLOAT16_H

#include "utils/floating_point_utils.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <ostream>

namespace aethermind {
namespace details {

float bf16_to_fp32_value(uint16_t input);

uint16_t bf16_from_fp32_value(float);

}
}

#endif//AETHERMIND_BFLOAT16_H
