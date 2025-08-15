//
// Created by 赵丹 on 2025/8/15.
//

#ifndef AETHERMIND_ANY_H
#define AETHERMIND_ANY_H

#include "device.h"

namespace aethermind {

class Any {
    public:
private:
    union data {
        int64_t v_int;
        double v_float;
        bool v_bool;
    } data_;
};

}

#endif//AETHERMIND_ANY_H
