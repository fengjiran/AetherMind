//
// Created by 赵丹 on 2025/8/14.
//
#include "type.h"
#include <gtest/gtest.h>

namespace {

using namespace aethermind;

TEST(Type, init) {
    const Type* t1 = AnyType::Global().get();
    EXPECT_EQ(t1->kind(), TypeKind::AnyType);
    EXPECT_EQ(t1->str(), "Any");
    EXPECT_EQ(t1->annotation_str(), "Any");
    auto printer = [](const Type& t) -> std::optional<std::string> {
        return t.str() + "_test";
    };
    EXPECT_EQ(t1->annotation_str(printer), "Any_test");

    // const Type& t2 = NumberType::GetInst();

    const Type* t3 = IntType::GetTypePtr();
    EXPECT_EQ(t3->kind(), TypeKind::IntType);
    EXPECT_EQ(t3->str(), "int");
    EXPECT_EQ(t3->annotation_str(), "int");
}

}