//
// Created by 赵丹 on 2025/8/20.
//
#include "object.h"
#include <gtest/gtest.h>

using namespace aethermind;

class NumberObj : public Object {};

class IntObj : public NumberObj {
public:
    int64_t value;
    IntObj() : value(0) {}
    explicit IntObj(int64_t value) : value(value) {}
};

namespace {

TEST(object, ctors) {
    ObjectPtr<NumberObj> p1;
    EXPECT_TRUE(!p1.defined());
    EXPECT_EQ(p1.use_count(), 0);
    EXPECT_TRUE(p1.get() == NullTypeOf<NumberObj>::singleton());

    ObjectPtr<NumberObj> p2 = nullptr;
    EXPECT_TRUE(!p2.defined());
    EXPECT_EQ(p2.use_count(), 0);
    EXPECT_TRUE(p2.get() == NullTypeOf<NumberObj>::singleton());

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
    // std::weak_ptr<int> p;
    // p.expired();
}

class Class0 : public Object {};
class Class1 : public Object {
public:
    Class1() : val_(0) {}
    explicit Class1(int val) : val_(val) {}
    int val_;
};

class Class2 : public Object {
public:
    Class2() : val1_(0), val2_(0) {}
    Class2(int val1, int val2) : val1_(val1), val2_(val2) {}
    int val1_;
    int val2_;
};


struct SomeBaseClass : Object {
    SomeBaseClass() : val_(0) {}
    explicit SomeBaseClass(int val) : val_(val) {}
    int val_;
};

struct SomeChildClass : SomeBaseClass {
    SomeChildClass() : SomeBaseClass(0) {}
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

TEST(object, move_assign_then_new_inst_is_valid) {
    auto obj1 = make_object<SomeClass>();
    ObjectPtr<SomeClass> obj2;
    obj2 = std::move(obj1);
    EXPECT_TRUE(!obj1.defined());
    EXPECT_TRUE(obj2.defined());
}

TEST(object, move_assign_then_point_to_same_obj) {
    auto obj1 = make_object<SomeClass>();
    ObjectPtr<SomeClass> obj2;
    auto p1 = obj1.get();
    obj2 = std::move(obj1);
    EXPECT_EQ(obj2.get(), p1);
}

TEST(object, move_assign_from_invalid_ptr_then_new_obj_is_invalid) {
    ObjectPtr<SomeClass> obj1;
    ObjectPtr<SomeClass> obj2 = make_object<SomeClass>();
    EXPECT_TRUE(obj2.defined());
    obj2 = std::move(obj1);
    EXPECT_FALSE(obj2.defined());
}

TEST(object, move_assign_to_base_class_then_point_to_same_obj) {
    ObjectPtr<SomeChildClass> obj1 = make_object<SomeChildClass>(1);
    ObjectPtr<SomeBaseClass> obj2 = make_object<SomeBaseClass>(2);
    SomeBaseClass* obj1ptr = obj1.get();
    obj2 = std::move(obj1);
    EXPECT_EQ(obj1ptr, obj2.get());
    EXPECT_EQ(1, obj2->val_);
}

TEST(object, move_assign_to_base_class_then_old_inst_invalid) {
    ObjectPtr<SomeChildClass> obj1 = make_object<SomeChildClass>(1);
    ObjectPtr<SomeBaseClass> obj2 = make_object<SomeBaseClass>(2);
    obj2 = std::move(obj1);
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_FALSE(obj1.defined());
}

TEST(object, move_assign_to_base_class_then_new_inst_valid) {
    ObjectPtr<SomeChildClass> obj1 = make_object<SomeChildClass>(5);
    ObjectPtr<SomeBaseClass> obj2;
    obj2 = std::move(obj1);
    EXPECT_TRUE(obj2.defined());
}

TEST(object, given_invalid_ptr_move_assign_to_base_class_then_point_to_same_obj) {
    ObjectPtr<SomeChildClass> obj1 = make_object<SomeChildClass>(5);
    ObjectPtr<SomeBaseClass> obj2;
    SomeBaseClass* obj1ptr = obj1.get();
    obj2 = std::move(obj1);
    EXPECT_EQ(obj1ptr, obj2.get());
    EXPECT_EQ(5, obj2->val_);
}

TEST(object, given_invalid_ptr_move_assign_invalid_ptr_to_base_class_then_new_inst_valid) {
    ObjectPtr<SomeChildClass> obj1;
    ObjectPtr<SomeBaseClass> obj2 = make_object<SomeBaseClass>(2);
    EXPECT_TRUE(obj2.defined());
    obj2 = std::move(obj1);
    EXPECT_FALSE(obj2.defined());
}

TEST(object, given_nulltype_move_assign_to_diff_nulltype_then_has_new_nulltype) {
    ObjectPtr<SomeClass> obj1;
    ObjectPtr<SomeClass> obj2;
    obj2 = std::move(obj1);
    EXPECT_NE(NullType1::singleton(), NullType2::singleton());
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_NE(NullType1::singleton(), obj1.get());
    EXPECT_NE(NullType2::singleton(), obj2.get());
    EXPECT_FALSE(obj1.defined());
    EXPECT_FALSE(obj2.defined());
}

TEST(object, givenValidPtr_whenCopyAssigning_thenPointsToSameObject) {
    ObjectPtr<SomeClass> obj1 = make_object<SomeClass>();
    ObjectPtr<SomeClass> obj2 = make_object<SomeClass>();
    SomeClass* obj1ptr = obj1.get();
    obj2 = obj1;
    EXPECT_EQ(obj1ptr, obj2.get());
}

TEST(object, givenValidPtr_whenCopyAssigning_thenOldInstanceValid) {
    ObjectPtr<SomeClass> obj1 = make_object<SomeClass>();
    ObjectPtr<SomeClass> obj2 = make_object<SomeClass>();
    obj2 = obj1;
    EXPECT_TRUE(obj1.defined());
}

TEST(object, givenValidPtr_whenCopyAssigningToSelf_thenPointsToSameObject) {
    ObjectPtr<SomeClass> obj1 = make_object<SomeClass>();
    SomeClass* obj1ptr = obj1.get();
    obj1 = obj1;
    EXPECT_EQ(obj1ptr, obj1.get());
}

TEST(object, givenValidPtr_whenCopyAssigningToSelf_thenStaysValid) {
    ObjectPtr<SomeClass> obj1 = make_object<SomeClass>();
    obj1 = obj1;
    EXPECT_TRUE(obj1.defined());
}

TEST(object, givenInvalidPtr_whenCopyAssigningToSelf_thenStaysInvalid) {
    ObjectPtr<SomeClass> obj1;
    // NOLINTNEXTLINE(clang-diagnostic-self-assign-overloaded)
    obj1 = obj1;
    EXPECT_FALSE(obj1.defined());
}

TEST(object, givenInvalidPtr_whenCopyAssigning_thenNewInstanceIsValid) {
    ObjectPtr<SomeClass> obj1 = make_object<SomeClass>();
    ObjectPtr<SomeClass> obj2;
    obj2 = obj1;
    EXPECT_TRUE(obj2.defined());
}

TEST(object, givenValidPtr_whenCopyAssigningToBaseClass_thenPointsToSameObject) {
    ObjectPtr<SomeChildClass> child = make_object<SomeChildClass>(3);
    ObjectPtr<SomeBaseClass> base = make_object<SomeBaseClass>(10);
    base = child;
    EXPECT_EQ(3, base->val_);
}

TEST(object, givenValidPtr_whenCopyAssigningToBaseClass_thenOldInstanceInvalid) {
    ObjectPtr<SomeChildClass> obj1 = make_object<SomeChildClass>(3);
    ObjectPtr<SomeBaseClass> obj2 = make_object<SomeBaseClass>(10);
    obj2 = obj1;
    EXPECT_TRUE(obj1.defined());
}

TEST(object, givenInvalidPtr_whenCopyAssigningToBaseClass_thenNewInstanceIsValid) {
    ObjectPtr<SomeChildClass> obj1 = make_object<SomeChildClass>(5);
    ObjectPtr<SomeBaseClass> obj2;
    obj2 = obj1;
    EXPECT_TRUE(obj2.defined());
}

TEST(object, givenInvalidPtr_whenCopyAssigningToBaseClass_thenPointsToSameObject) {
    ObjectPtr<SomeChildClass> obj1 = make_object<SomeChildClass>(5);
    ObjectPtr<SomeBaseClass> obj2;
    SomeBaseClass* obj1ptr = obj1.get();
    obj2 = obj1;
    EXPECT_EQ(obj1ptr, obj2.get());
    EXPECT_EQ(5, obj2->val_);
}

TEST(object, givenPtr_whenCopyAssigningInvalidPtrToBaseClass_thenNewInstanceIsInvalid) {
    ObjectPtr<SomeChildClass> obj1;
    ObjectPtr<SomeBaseClass> obj2 = make_object<SomeBaseClass>(2);
    EXPECT_TRUE(obj2.defined());
    obj2 = obj1;
    EXPECT_FALSE(obj2.defined());
}

TEST(object, givenNullPtr_whenCopyAssigningToDifferentNullptr_thenHasNewNullptr) {
    ObjectPtr<SomeClass> obj1;
    ObjectPtr<SomeClass> obj2;
    obj2 = obj1;
    EXPECT_NE(NullType1::singleton(), NullType2::singleton());
    EXPECT_NE(NullType1::singleton(), obj1.get());
    EXPECT_NE(NullType2::singleton(), obj2.get());
    EXPECT_FALSE(obj1.defined());
    EXPECT_FALSE(obj2.defined());
}

TEST(object, givenPtr_whenMoveConstructing_thenPointsToSameObject) {
    ObjectPtr<SomeClass> obj1 = make_object<SomeClass>();
    SomeClass* obj1ptr = obj1.get();
    ObjectPtr<SomeClass> obj2 = std::move(obj1);
    EXPECT_EQ(obj1ptr, obj2.get());
}

TEST(object, givenPtr_whenMoveConstructing_thenOldInstanceInvalid) {
    ObjectPtr<SomeClass> obj1 = make_object<SomeClass>();
    ObjectPtr<SomeClass> obj2 = std::move(obj1);
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move,bugprone-use-after-move)
    EXPECT_FALSE(obj1.defined());
}

TEST(object, givenPtr_whenMoveConstructing_thenNewInstanceValid) {
    ObjectPtr<SomeClass> obj1 = make_object<SomeClass>();
    ObjectPtr<SomeClass> obj2 = std::move(obj1);
    EXPECT_TRUE(obj2.defined());
}

TEST(object, givenPtr_whenMoveConstructingFromInvalidPtr_thenNewInstanceInvalid) {
    ObjectPtr<SomeClass> obj1;
    ObjectPtr<SomeClass> obj2 = std::move(obj1);
    EXPECT_FALSE(obj2.defined());
}

TEST(object, givenPtr_whenMoveConstructingToBaseClass_thenPointsToSameObject) {
    ObjectPtr<SomeChildClass> child = make_object<SomeChildClass>(3);
    SomeBaseClass* objptr = child.get();
    ObjectPtr<SomeBaseClass> base = std::move(child);
    EXPECT_EQ(3, base->val_);
    EXPECT_EQ(objptr, base.get());
}

TEST(object, givenPtr_whenMoveConstructingToBaseClass_thenOldInstanceInvalid) {
    ObjectPtr<SomeChildClass> child = make_object<SomeChildClass>(3);
    ObjectPtr<SomeBaseClass> base = std::move(child);
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_FALSE(child.defined());
}

TEST(object, givenPtr_whenMoveConstructingToBaseClass_thenNewInstanceValid) {
    ObjectPtr<SomeChildClass> obj1 = make_object<SomeChildClass>(2);
    ObjectPtr<SomeBaseClass> obj2 = std::move(obj1);
    EXPECT_TRUE(obj2.defined());
}

TEST(object, givenPtr_whenMoveConstructingToBaseClassFromInvalidPtr_thenNewInstanceInvalid) {
    ObjectPtr<SomeChildClass> obj1;
    ObjectPtr<SomeBaseClass> obj2 = std::move(obj1);
    EXPECT_FALSE(obj2.defined());
}

TEST(object, givenNullPtr_whenMoveConstructingToDifferentNullptr_thenHasNewNullptr) {
    ObjectPtr<SomeClass> obj1;
    ObjectPtr<SomeClass> obj2 = std::move(obj1);
    EXPECT_NE(NullType1::singleton(), NullType2::singleton());
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_NE(NullType1::singleton(), obj1.get());
    EXPECT_NE(NullType2::singleton(), obj2.get());
    EXPECT_FALSE(obj1.defined());
    EXPECT_FALSE(obj2.defined());
}

TEST(object, givenPtr_whenCopyConstructing_thenPointsToSameObject) {
    ObjectPtr<SomeClass> obj1 = make_object<SomeClass>();
    SomeClass* obj1ptr = obj1.get();
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    ObjectPtr<SomeClass> obj2 = obj1;
    EXPECT_EQ(obj1ptr, obj2.get());
    EXPECT_TRUE(obj1.defined());
}

TEST(object, givenPtr_whenCopyConstructing_thenOldInstanceValid) {
    ObjectPtr<SomeClass> obj1 = make_object<SomeClass>();
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    ObjectPtr<SomeClass> obj2 = obj1;
    EXPECT_TRUE(obj1.defined());
}

TEST(object, givenPtr_whenCopyConstructing_thenNewInstanceValid) {
    ObjectPtr<SomeClass> obj1 = make_object<SomeClass>();
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    ObjectPtr<SomeClass> obj2 = obj1;
    EXPECT_TRUE(obj2.defined());
}

TEST(object, givenPtr_whenCopyConstructingFromInvalidPtr_thenNewInstanceInvalid) {
    ObjectPtr<SomeClass> obj1;
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    ObjectPtr<SomeClass> obj2 = obj1;
    EXPECT_FALSE(obj2.defined());
}

TEST(object, givenPtr_whenCopyConstructingToBaseClass_thenPointsToSameObject) {
    ObjectPtr<SomeChildClass> child = make_object<SomeChildClass>(3);
    SomeBaseClass* objptr = child.get();
    ObjectPtr<SomeBaseClass> base = child;
    EXPECT_EQ(3, base->val_);
    EXPECT_EQ(objptr, base.get());
}
//546

TEST(weak_object_ptr, basic_construction) {
    // 默认构造函数
    WeakObjectPtr<NumberObj> weak1;
    EXPECT_FALSE(weak1.defined());
    EXPECT_TRUE(weak1.expired());
    EXPECT_EQ(weak1.use_count(), 0);
    EXPECT_EQ(weak1.weak_use_count(), 0);

    // 从ObjectPtr构造
    auto obj = make_object<NumberObj>();
    WeakObjectPtr<NumberObj> weak2(obj);
    EXPECT_TRUE(weak2.defined());
    EXPECT_FALSE(weak2.expired());
    EXPECT_EQ(weak2.use_count(), 1);
    EXPECT_EQ(weak2.weak_use_count(), 2);
}

TEST(weak_object_ptr, copy_construction) {
    auto obj = make_object<NumberObj>();
    WeakObjectPtr<NumberObj> weak1(obj);

    // 拷贝构造
    WeakObjectPtr<NumberObj> weak2(weak1);
    EXPECT_TRUE(weak2.defined());
    EXPECT_EQ(weak2.use_count(), 1);
    EXPECT_EQ(weak2.weak_use_count(), 3);

    // 验证两个weak指针指向同一个对象
    EXPECT_EQ(weak1.unsafe_get(), weak2.unsafe_get());
}

TEST(weak_object_ptr, move_construction) {
    auto obj = make_object<NumberObj>();
    WeakObjectPtr<NumberObj> weak1(obj);

    // 移动构造
    WeakObjectPtr<NumberObj> weak2(std::move(weak1));
    EXPECT_TRUE(weak2.defined());
    EXPECT_FALSE(weak1.defined());// 原指针应该被置空
    EXPECT_EQ(weak2.use_count(), 1);
    EXPECT_EQ(weak2.weak_use_count(), 2);
}

TEST(weak_object_ptr, assignment_operators) {
    auto obj1 = make_object<NumberObj>();
    auto obj2 = make_object<NumberObj>();

    WeakObjectPtr<NumberObj> weak1(obj1);
    WeakObjectPtr<NumberObj> weak2(obj2);

    // 拷贝赋值
    weak2 = weak1;
    EXPECT_EQ(weak1.unsafe_get(), weak2.unsafe_get());
    EXPECT_EQ(weak1.use_count(), 1);
    EXPECT_EQ(weak1.weak_use_count(), 3);

    // 移动赋值
    WeakObjectPtr<NumberObj> weak3;
    weak3 = std::move(weak1);
    EXPECT_TRUE(weak3.defined());
    EXPECT_FALSE(weak1.defined());
}

TEST(weak_object_ptr, lock_operation) {
    auto obj = make_object<NumberObj>();
    WeakObjectPtr<NumberObj> weak(obj);

    // lock应该返回有效的ObjectPtr
    auto locked = weak.lock();
    EXPECT_TRUE(locked.defined());
    EXPECT_EQ(locked.use_count(), 2);// obj + locked

    // 释放原始对象后lock应该返回nullptr
    obj.reset();
    auto locked_after = weak.lock();
    EXPECT_TRUE(locked_after.defined());
    EXPECT_FALSE(weak.expired());
}

TEST(weak_object_ptr, expired_check) {
    auto obj = make_object<NumberObj>();
    WeakObjectPtr<NumberObj> weak(obj);

    EXPECT_FALSE(weak.expired());

    obj.reset();
    EXPECT_TRUE(weak.expired());
}

TEST(weak_object_ptr, reset_operation) {
    auto obj = make_object<NumberObj>();
    WeakObjectPtr<NumberObj> weak(obj);

    EXPECT_TRUE(weak.defined());

    weak.reset();
    EXPECT_FALSE(weak.defined());
    EXPECT_TRUE(weak.expired());
}

TEST(weak_object_ptr, release_operation) {
    auto obj = make_object<NumberObj>();
    WeakObjectPtr<NumberObj> weak(obj);

    auto* released = weak.release();
    EXPECT_EQ(released, obj.get());
    EXPECT_FALSE(weak.defined());

    // 需要手动管理释放的指针
    details::ObjectUnsafe::DecWeakRef(released);
}

TEST(weak_object_ptr, inheritance_conversion) {
    auto intObj = make_object<IntObj>(42);

    // 从派生类到基类的转换
    WeakObjectPtr<NumberObj> weak_base(intObj);
    EXPECT_TRUE(weak_base.defined());

    // 验证转换后的指针类型
    auto locked = weak_base.lock();
    EXPECT_TRUE(locked.get() == intObj.get());
}

TEST(weak_object_ptr, comparison_operators) {
    auto obj1 = make_object<NumberObj>();
    auto obj2 = make_object<NumberObj>();

    WeakObjectPtr<NumberObj> weak1(obj1);
    WeakObjectPtr<NumberObj> weak2(obj1);// 指向同一个对象
    WeakObjectPtr<NumberObj> weak3(obj2);// 指向不同对象

    EXPECT_EQ(weak1, weak2);
    EXPECT_NE(weak1, weak3);

    // 空指针比较
    WeakObjectPtr<NumberObj> weak4;
    WeakObjectPtr<NumberObj> weak5;
    EXPECT_EQ(weak4, weak5);
}

TEST(weak_object_ptr, swap_operation) {
    auto obj1 = make_object<NumberObj>();
    auto obj2 = make_object<NumberObj>();

    WeakObjectPtr<NumberObj> weak1(obj1);
    WeakObjectPtr<NumberObj> weak2(obj2);

    auto* ptr1 = weak1.unsafe_get();
    auto* ptr2 = weak2.unsafe_get();

    weak1.swap(weak2);

    EXPECT_EQ(weak1.unsafe_get(), ptr2);
    EXPECT_EQ(weak2.unsafe_get(), ptr1);
}

TEST(weak_object_ptr, reclaim_operation) {
    auto obj = make_object<NumberObj>();

    // 创建多个weak引用
    WeakObjectPtr<NumberObj> weak1(obj);
    WeakObjectPtr<NumberObj> weak2(obj);

    // 释放原始对象
    obj.reset();

    // 使用reclaim回收指针
    auto reclaimed = WeakObjectPtr<NumberObj>::reclaim(weak1.release());
    EXPECT_TRUE(reclaimed.defined());
    EXPECT_EQ(reclaimed.use_count(), 0);
    EXPECT_EQ(reclaimed.weak_use_count(), 2);
}

}// namespace