//
// Created by 赵丹 on 2025/8/11.
//

#ifndef AETHERMIND_DISPATCH_KEY_SET_H
#define AETHERMIND_DISPATCH_KEY_SET_H

#include "dispatch_key.h"

namespace aethermind {

class DispatchKeySet final {
public:
    DispatchKeySet() = default;


private:
    DispatchKeySet(uint64_t repr) : repr_(repr) {}
    uint64_t repr_ = 0;

};

}

#endif//AETHERMIND_DISPATCH_KEY_SET_H
