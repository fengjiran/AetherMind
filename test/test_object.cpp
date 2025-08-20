//
// Created by 赵丹 on 2025/8/20.
//
#include "object.h"
#include <gtest/gtest.h>

using namespace aethermind;

namespace {

class NumberObj : public Object {};

class IntObj : public NumberObj {
public:
    int64_t value;
    IntObj() : value(0) {}
    explicit IntObj(int64_t value) : value(value) {}
};

TEST(object, ctors) {
    ObjectPtr<NumberObj> p1;
    EXPECT_TRUE(!p1.defined());
    EXPECT_EQ(p1.use_count(), 0);

    ObjectPtr<NumberObj> p2 = nullptr;
    EXPECT_TRUE(!p2.defined());
    EXPECT_EQ(p2.use_count(), 0);

    auto p3 = make_object<NumberObj>();
    EXPECT_TRUE(p3.defined());
    EXPECT_EQ(p3.use_count(), 1);

    ObjectPtr<NumberObj> p4 = p3;
    EXPECT_TRUE(p4.defined());
    EXPECT_EQ(p4.use_count(), 2);

    p1 = p4;
    EXPECT_TRUE(p1.defined());
    EXPECT_EQ(p1.use_count(), 3);

    auto p5 = make_object<IntObj>(5);
    EXPECT_TRUE(p5.defined());
    EXPECT_EQ(p5.use_count(), 1);
    EXPECT_EQ(p5->value, 5);

    p4 = p5;
    EXPECT_TRUE(p4.defined());
    EXPECT_EQ(p4.use_count(), 2);
    EXPECT_EQ(static_cast<IntObj*>(p4.get())->value, 5);
}

class Class0 : public Object {};
class Class1 : public Object {
public:
    explicit Class1(int val) : val_(val) {}
    int val_;
};

class Class2 : public Object {
public:
    Class2(int val1, int val2) : val1_(val1), val2_(val2) {}
    int val1_;
    int val2_;
};

struct SomeBaseClass : Object {
    explicit SomeBaseClass(int val) : val_(val) {}
    int val_;
};

struct SomeChildClass : SomeBaseClass {
    explicit SomeChildClass(int v) : SomeBaseClass(v) {}
};

static_assert(std::is_same_v<Class0, ObjectPtr<Class0>::element_type>);

}// namespace