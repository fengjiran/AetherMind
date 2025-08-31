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

TEST(SingletonOrSharedTypePtr, SingletonComparison) {
    EXPECT_NE(StringType::Global(), NoneType::Global());
    EXPECT_NE(StringType::Global(), DeviceObjType::Global());
    EXPECT_NE(NoneType::Global(), DeviceObjType::Global());

    TypePtr type = NoneType::Global();
    EXPECT_NE(type, StringType::Global());
    EXPECT_NE(type, DeviceObjType::Global());

    TypePtr t1 = AnyType::Global();
    TypePtr t2 = NoneType::Global();
    TypePtr t3 = NumberType::Global();
    TypePtr t4 = IntType::Global();

    EXPECT_EQ(t1->str(), "Any");
    EXPECT_EQ(t2->str(), "None");
    EXPECT_EQ(t3->str(), "Scalar");
    EXPECT_EQ(t4->str(), "int");

    EXPECT_TRUE(*t1 == *AnyType::Global());
    EXPECT_TRUE(*t2 == *NoneType::Global());
    EXPECT_TRUE(*t3 == *NumberType::Global());
    EXPECT_TRUE(*t4 == *IntType::Global());

    EXPECT_TRUE(t1 == AnyType::Global());
    EXPECT_TRUE(t2 == NoneType::Global());
    EXPECT_TRUE(t3 == NumberType::Global());
    EXPECT_TRUE(t4 == IntType::Global());

    EXPECT_TRUE(t1.get() == AnyType::Global().get());
    EXPECT_TRUE(t2.get() == NoneType::Global().get());
    EXPECT_TRUE(t3.get() == NumberType::Global().get());
    EXPECT_TRUE(t4.get() == IntType::Global().get());
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

    TypePtr t3 = IntType::Global();
    EXPECT_EQ(t3->kind(), TypeKind::IntType);
    EXPECT_EQ(t3->str(), "int");
    EXPECT_EQ(t3->annotation_str(), "int");

    const auto& t4 = *t3;
    EXPECT_EQ(t4.kind(), TypeKind::IntType);
    EXPECT_EQ(t4.str(), "int");
    EXPECT_EQ(t4.annotation_str(), "int");
}

TEST(Type, Union) {
    TypePtr t1 = AnyType::Global();
    TypePtr t2 = NoneType::Global();
    TypePtr t3 = NumberType::Global();
    TypePtr t4 = IntType::Global();
    TypePtr t5 = FloatType::Global();
    TypePtr t6 = ComplexType::Global();
    TypePtr t7 = StringType::Global();
    TypePtr t8 = DeviceObjType::Global();

    EXPECT_TRUE(t4->is_subtype_of(t3));

    TypePtr union_type_1 = UnionType::create({t1, t2, t4, t5, t5});
    EXPECT_EQ(union_type_1->kind(), TypeKind::OptionalType);
    // EXPECT_EQ(union_type_1->containedTypeSize(), 4);
}

}// namespace