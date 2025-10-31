//
// Created by richard on 10/31/25.
//
#include "type_system/list_type.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

// 测试ListType的基本功能
TEST(ListTypeTest, BasicCreationAndProperties) {
    // 创建基本的列表类型
    auto int_type = IntType::Global();
    auto list_of_ints = ListType::Create(int_type);

    // 验证基本属性
    EXPECT_EQ(list_of_ints->kind(), TypeKind::ListType);
    EXPECT_TRUE(list_of_ints->GetElementType() == int_type);

    // 验证GetContainedTypes方法
    auto contained_types = list_of_ints->GetContainedTypes();
    EXPECT_EQ(contained_types.size(), 1);
    EXPECT_TRUE(contained_types[0] == int_type);

    // 验证GetContainedType方法
    EXPECT_TRUE(list_of_ints->GetContainedType(0) == int_type);
    EXPECT_EQ(list_of_ints->GetContainedTypeSize(), 1);
}

// 测试ListType的工厂方法
TEST(ListTypeTest, FactoryMethods) {
    // 测试OfNumbers工厂方法
    auto list_of_numbers = ListType::OfNumbers();
    EXPECT_EQ(list_of_numbers->kind(), TypeKind::ListType);
    EXPECT_TRUE(list_of_numbers->GetElementType() == NumberType::Global());

    // 测试OfInts工厂方法
    auto list_of_ints = ListType::OfInts();
    EXPECT_EQ(list_of_ints->kind(), TypeKind::ListType);
    EXPECT_TRUE(list_of_ints->GetElementType() == IntType::Global());

    // 测试OfFloats工厂方法
    auto list_of_floats = ListType::OfFloats();
    EXPECT_EQ(list_of_floats->kind(), TypeKind::ListType);
    EXPECT_TRUE(list_of_floats->GetElementType() == FloatType::Global());

    // 测试OfBools工厂方法
    auto list_of_bools = ListType::OfBools();
    EXPECT_EQ(list_of_bools->kind(), TypeKind::ListType);
    EXPECT_TRUE(list_of_bools->GetElementType() == BoolType::Global());

    // 测试OfComplexDoubles工厂方法
    auto list_of_complex = ListType::OfComplexDoubles();
    EXPECT_EQ(list_of_complex->kind(), TypeKind::ListType);
    EXPECT_TRUE(list_of_complex->GetElementType() == ComplexType::Global());

    // 测试OfStrings工厂方法
    auto list_of_strings = ListType::OfStrings();
    EXPECT_EQ(list_of_strings->kind(), TypeKind::ListType);
    EXPECT_TRUE(list_of_strings->GetElementType() == StringType::Global());
}

// 测试ListType的Get方法
TEST(ListTypeTest, GetMethod) {
    auto int_type = IntType::Global();

    // 使用Get方法创建列表类型
    auto list_of_ints = ListType::Get("List[int]", int_type);
    EXPECT_EQ(list_of_ints->kind(), TypeKind::ListType);

    // 验证列表元素类型
    auto casted_list = dynamic_cast<ListType*>(list_of_ints.get());
    EXPECT_TRUE(casted_list != nullptr);
    EXPECT_TRUE(casted_list->GetElementType() == int_type);
}

// 测试ListType的类型相等性
TEST(ListTypeTest, TypeEquality) {
    auto int_type = IntType::Global();
    auto float_type = FloatType::Global();

    // 相同元素类型的列表应该相等
    auto list1 = ListType::Create(int_type);
    auto list2 = ListType::Create(int_type);
    EXPECT_TRUE(list1->Equals(*list2));
    EXPECT_TRUE(*list1 == *list2);

    // 不同元素类型的列表不应该相等
    auto list3 = ListType::Create(float_type);
    EXPECT_FALSE(list1->Equals(*list3));
    EXPECT_FALSE(*list1 == *list3);

    // 与AnyListType比较
    auto any_list = AnyListType::Global();
    EXPECT_FALSE(list1->Equals(*any_list));
    EXPECT_FALSE(*list1 == *any_list);
}

// 测试AnyListType的基本功能
TEST(AnyListTypeTest, BasicProperties) {
    auto any_list = AnyListType::Global();

    // 验证基本属性
    EXPECT_EQ(any_list->kind(), TypeKind::AnyListType);

    // 验证单例特性
    auto any_list2 = AnyListType::Global();
    EXPECT_TRUE(any_list == any_list2);
}

// 测试ListType和AnyListType的子类型关系
TEST(ListTypeTest, SubtypeRelationship) {
    auto int_type = IntType::Global();
    auto list_of_ints = ListType::Create(int_type);
    auto any_list = AnyListType::Global();
    auto any_type = AnyType::Global();

    // 所有List[T]都应该是AnyList的子类型
    EXPECT_TRUE(list_of_ints->IsSubtypeOf(*any_list));

    // AnyList应该是Any的子类型
    EXPECT_TRUE(any_list->IsSubtypeOf(*any_type));

    // 列表之间的子类型关系应该基于元素类型
    auto float_type = FloatType::Global();
    auto list_of_floats = ListType::Create(float_type);

    // 除非元素类型有子类型关系，否则列表之间没有子类型关系
    EXPECT_FALSE(list_of_ints->IsSubtypeOf(*list_of_floats));
    EXPECT_FALSE(list_of_floats->IsSubtypeOf(*list_of_ints));
}

// 测试CreateWithContainedTypes方法
TEST(ListTypeTest, CreateWithContainedTypes) {
    auto int_type = IntType::Global();
    auto list_of_ints = ListType::Create(int_type);

    // 替换包含的类型
    auto float_type = FloatType::Global();
    std::vector<TypePtr> new_contained_types = {float_type};
    auto list_of_floats = list_of_ints->CreateWithContainedTypes(new_contained_types);

    // 验证新创建的类型
    EXPECT_EQ(list_of_floats->kind(), TypeKind::ListType);

    auto casted_list = dynamic_cast<ListType*>(list_of_floats.get());
    EXPECT_TRUE(casted_list != nullptr);
    EXPECT_TRUE(casted_list->GetElementType() == float_type);

    // 测试WithContainedTypes方法（这是Type基类的方法）
    auto bool_type = BoolType::Global();
    std::vector<TypePtr> bool_contained_types = {bool_type};
    auto list_of_bools = list_of_ints->WithContainedTypes(bool_contained_types);

    auto casted_bool_list = dynamic_cast<ListType*>(list_of_bools.get());
    EXPECT_TRUE(casted_bool_list != nullptr);
    EXPECT_TRUE(casted_bool_list->GetElementType() == bool_type);
}

// 测试str方法和字符串表示
TEST(ListTypeTest, StringRepresentation) {
    auto int_type = IntType::Global();
    auto list_of_ints = ListType::Create(int_type);
    auto any_list = AnyListType::Global();

    // 测试str方法（具体实现可能在.cpp文件中，但我们可以测试接口）
    EXPECT_FALSE(list_of_ints->str().empty());
    EXPECT_EQ(any_list->str(), "list");

    // 测试Annotation方法
    EXPECT_FALSE(list_of_ints->Annotation().empty());
    EXPECT_FALSE(any_list->Annotation().empty());

    // 测试ReprStr方法
    EXPECT_FALSE(list_of_ints->ReprStr().empty());
    EXPECT_FALSE(any_list->ReprStr().empty());
}

// 测试ListType的边界情况
TEST(ListTypeTest, EdgeCases) {
    // 测试嵌套列表
    auto int_type = IntType::Global();
    auto list_of_ints = ListType::Create(int_type);
    auto list_of_list_of_ints = ListType::Create(list_of_ints);

    EXPECT_EQ(list_of_list_of_ints->kind(), TypeKind::ListType);

    auto inner_list = dynamic_cast<ListType*>(list_of_list_of_ints->GetElementType().get());
    EXPECT_TRUE(inner_list != nullptr);
    EXPECT_TRUE(inner_list->GetElementType() == int_type);
}

// 测试IsSubtypeOfExt方法的详细信息输出
TEST(ListTypeTest, IsSubtypeOfExtWithWhyNot) {
    auto int_type = IntType::Global();
    auto float_type = FloatType::Global();
    auto list_of_ints = ListType::Create(int_type);
    auto list_of_floats = ListType::Create(float_type);
    auto any_list = AnyListType::Global();

    // 当子类型关系成立时
    std::stringstream why_not1;
    EXPECT_TRUE(list_of_ints->IsSubtypeOfExt(*any_list, &why_not1));
    EXPECT_TRUE(why_not1.str().empty());

    // 当子类型关系不成立时
    std::stringstream why_not2;
    EXPECT_FALSE(list_of_ints->IsSubtypeOfExt(*list_of_floats, &why_not2));
    // 具体的错误消息内容取决于实现，但应该不为空
}
}// namespace