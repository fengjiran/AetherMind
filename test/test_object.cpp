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
    EXPECT_TRUE(p1.get() == nullptr);

    ObjectPtr<NumberObj> p2 = nullptr;
    EXPECT_TRUE(!p2.defined());
    EXPECT_EQ(p2.use_count(), 0);
    EXPECT_TRUE(p2.get() == nullptr);

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

static_assert(std::is_same_v<Class0, ObjectPtr<Class0>::element_type>);

struct SomeBaseClass : Object {
    explicit SomeBaseClass(int val) : val_(val) {}
    int val_;
};

struct SomeChildClass : SomeBaseClass {
    explicit SomeChildClass(int v) : SomeBaseClass(v) {}
};

class DestructableMock : public Object {
public:
    DestructableMock(bool* resourcesReleased, bool* wasDestructed)
        : resourcesReleased_(resourcesReleased), wasDestructed_(wasDestructed) {}

    ~DestructableMock() {
        *resourcesReleased_ = true;
        *wasDestructed_ = true;
    }

    void release_resources() {
        *resourcesReleased_ = true;
    }

private:
    bool* resourcesReleased_;
    bool* wasDestructed_;
};

class ChildDestructableMock final : public DestructableMock {
public:
    ChildDestructableMock(bool* resourcesReleased, bool* wasDestructed)
        : DestructableMock(resourcesReleased, wasDestructed) {}
};

using SomeClass = Class0;

class NullType1 final {
    static SomeClass singleton_;

public:
    static constexpr SomeClass* singleton() {
        return &singleton_;
    }
};

SomeClass NullType1::singleton_;

class NullType2 final {
    static SomeClass singleton_;

public:
    static constexpr SomeClass* singleton() {
        return &singleton_;
    }
};
SomeClass NullType2::singleton_;
static_assert(NullType1::singleton() != NullType2::singleton());

TEST(object, class0) {
    auto var = make_object<Class0>();
    EXPECT_EQ(var.get(), dynamic_cast<Class0*>(var.get()));
    Class0 cls;
}

TEST(object, class1) {
    const auto var = make_object<Class1>(5);
    EXPECT_EQ(var->val_, 5);
    EXPECT_EQ(var.get()->val_, 5);
    EXPECT_EQ((*var).val_, 5);
}

TEST(object, class2) {
    auto var = make_object<Class2>(5, 10);
    EXPECT_EQ(var->val1_, 5);
    EXPECT_EQ(var->val2_, 10);
}

TEST(object, assign_to_base_class_ptr) {
    ObjectPtr<SomeBaseClass> var = make_object<SomeChildClass>(5);
    EXPECT_EQ(var->val_, 5);
    EXPECT_EQ(var->use_count(), 1);
}

TEST(object, move_assign_ptr_to_same_object) {
    auto obj1 = make_object<SomeClass>();
    auto obj2 = make_object<SomeClass>();
    SomeClass* p1 = obj1.get();
    obj2 = std::move(obj1);
    EXPECT_EQ(obj2.get(), p1);
    EXPECT_EQ(obj2.use_count(), 1);
}

TEST(object, move_assign_then_old_inst_invalid) {
    auto obj1 = make_object<SomeClass>();
    auto obj2 = make_object<SomeClass>();
    obj2 = std::move(obj1);
    EXPECT_TRUE(!obj1.defined());
}

TEST(object, move_assign_to_self_then_point_to_same_obj) {
    auto obj1 = make_object<SomeClass>();
    SomeClass* p1 = obj1.get();
    obj1 = std::move(obj1);
    EXPECT_EQ(obj1.get(), p1);
    EXPECT_TRUE(obj1.defined());
}

TEST(object, move_assign_to_self_then_stay_invalid) {
    ObjectPtr<SomeClass> obj1;
    obj1 = std::move(obj1);
    EXPECT_TRUE(!obj1.defined());
}



}// namespace