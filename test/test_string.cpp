//
// Created by 赵丹 on 2025/8/22.
//
#include "container/string.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

struct t1 {
    int a;
    char b;
};

using storage_type = std::aligned_storage_t<sizeof(t1), alignof(t1)>;

static_assert(sizeof(storage_type) == 8);

}