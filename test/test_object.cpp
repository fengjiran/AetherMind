//
// Created by 赵丹 on 2025/8/20.
//
#include "object.h"
#include <gtest/gtest.h>

using namespace aethermind;

namespace {

class TestA : public Object {};

TEST(object, ctors) {
    ObjectPtr<TestA> p = nullptr;
}

}