//
// Created by 赵丹 on 2025/8/14.
//
#include "type.h"
#include <gtest/gtest.h>

namespace {

using namespace aethermind;

TEST(SingletonOrSharedTypePtr, Empty) {
    SingletonOrSharedTypePtr<int> empty;
    EXPECT_FALSE(empty);
    EXPECT_EQ(nullptr, empty.get());
    EXPECT_EQ(empty, nullptr);
    std::shared_ptr<int> emptyShared;
    EXPECT_EQ(emptyShared, empty);
}

TEST(SingletonOrSharedTypePtr, NonEmpty) {
    auto shared = std::make_shared<int>(42);
    SingletonOrSharedTypePtr<int> p(shared);

    EXPECT_TRUE(shared);
    EXPECT_EQ(42, *shared);
    EXPECT_EQ(42, *p);
    EXPECT_TRUE(p);
    EXPECT_NE(nullptr, p.get());
    EXPECT_TRUE(p != nullptr);
    EXPECT_EQ(shared, p);
    EXPECT_EQ(shared.get(), p.get());
}

TEST(SingletonOrSharedTypePtr, Comparison) {
    SingletonOrSharedTypePtr<int> empty;
    auto shared = std::make_shared<int>(42);
    SingletonOrSharedTypePtr<int> p(shared);
    auto shared2 = std::make_shared<int>(3);
    SingletonOrSharedTypePtr<int> p2(shared2);

    EXPECT_NE(empty, p);
    EXPECT_NE(p, p2);
}

TEST(Type, init) {
    const Type* t1 = AnyType::Global().get();
    EXPECT_EQ(t1->kind(), TypeKind::AnyType);
    EXPECT_EQ(t1->str(), "Any");
    EXPECT_EQ(t1->annotation_str(), "Any");
    EXPECT_TRUE(t1->cast_to_raw_type<AnyType>() == t1);
    EXPECT_TRUE(t1->cast<AnyType>().get() == t1);

    auto printer = [](const Type& t) -> std::optional<std::string> {
        return t.str() + "_test";
    };
    EXPECT_EQ(t1->annotation_str(printer), "Any_test");

    // const Type& t2 = NumberType::GetInst();

    TypePtr t3 = IntType::Global();
    EXPECT_EQ(t3->kind(), TypeKind::IntType);
    EXPECT_EQ(t3->str(), "int");
    EXPECT_EQ(t3->annotation_str(), "int");

    const auto& t4 = *t3;
    EXPECT_EQ(t4.kind(), TypeKind::IntType);
    EXPECT_EQ(t4.str(), "int");
    EXPECT_EQ(t4.annotation_str(), "int");
}

}// namespace