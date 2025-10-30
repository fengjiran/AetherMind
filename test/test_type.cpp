//
// Created by 赵丹 on 2025/8/14.
//
// #include "type_system/tensor_type.h"
#include "type_system/union_type.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

// 基本类型测试
TEST(TypeSystem, BasicTypeProperties) {
    // 测试单例类型的全局实例
    EXPECT_TRUE(AnyType::Global());
    EXPECT_TRUE(NoneType::Global());
    EXPECT_TRUE(NumberType::Global());
    EXPECT_TRUE(IntType::Global());
    EXPECT_TRUE(FloatType::Global());
    EXPECT_TRUE(ComplexType::Global());
    EXPECT_TRUE(StringType::Global());
    EXPECT_TRUE(DeviceObjType::Global());

    // 测试类型的字符串表示
    EXPECT_EQ(AnyType::Global()->str(), "Any");
    EXPECT_EQ(NoneType::Global()->str(), "None");
    EXPECT_EQ(NumberType::Global()->str(), "Scalar");
    EXPECT_EQ(IntType::Global()->str(), "int");
    EXPECT_EQ(FloatType::Global()->str(), "float");
    EXPECT_EQ(ComplexType::Global()->str(), "complex");
    EXPECT_EQ(StringType::Global()->str(), "string");
    EXPECT_EQ(DeviceObjType::Global()->str(), "Device");

    // 测试类型的注释字符串表示
    EXPECT_EQ(AnyType::Global()->Annotation(), "Any");
    EXPECT_EQ(NoneType::Global()->Annotation(), "None");
    EXPECT_EQ(NumberType::Global()->Annotation(), "number");
    EXPECT_EQ(IntType::Global()->Annotation(), "int");
    EXPECT_EQ(FloatType::Global()->Annotation(), "float");
    EXPECT_EQ(ComplexType::Global()->Annotation(), "complex");
    EXPECT_EQ(StringType::Global()->Annotation(), "string");
    EXPECT_EQ(DeviceObjType::Global()->Annotation(), "Device");

    // 测试自定义类型打印机
    auto custom_printer = [](const Type& t) -> std::optional<std::string> {
        if (t.kind() == TypeKind::IntType) {
            return "CustomInt";
        }
        return std::nullopt;
    };

    EXPECT_EQ(IntType::Global()->Annotation(custom_printer), "CustomInt");
    EXPECT_EQ(FloatType::Global()->Annotation(custom_printer), "float");
}

// 类型相等性测试
TEST(TypeSystem, TypeEquality) {
    // 相同类型比较
    EXPECT_TRUE(*AnyType::Global() == *AnyType::Global());
    EXPECT_TRUE(*NoneType::Global() == *NoneType::Global());
    EXPECT_TRUE(*IntType::Global() == *IntType::Global());
    EXPECT_TRUE(*FloatType::Global() == *FloatType::Global());

    // 不同类型比较
    EXPECT_FALSE(*AnyType::Global() == *NoneType::Global());
    EXPECT_FALSE(*IntType::Global() == *FloatType::Global());
    EXPECT_FALSE(*NumberType::Global() == *StringType::Global());

    // 不等运算符测试
    EXPECT_TRUE(*AnyType::Global() != *NoneType::Global());
    EXPECT_FALSE(*IntType::Global() != *IntType::Global());

    // TypePtr比较
    TypePtr t1 = AnyType::Global();
    TypePtr t2 = AnyType::Global();
    TypePtr t3 = NoneType::Global();

    EXPECT_TRUE(*t1 == *t2);
    EXPECT_FALSE(*t1 == *t3);
    EXPECT_TRUE(t1 == t2);
    EXPECT_FALSE(t1 == t3);
}

// 类型转换和投射测试
TEST(TypeSystem, TypeCasting) {
    // 向下转型测试
    Type* any_type = AnyType::Global().get();
    auto* cast_any = any_type->CastToRawTypePtr<AnyType>();
    EXPECT_TRUE(cast_any != nullptr);
    EXPECT_TRUE(cast_any == any_type);

    // 无效转型测试
    auto* cast_int = any_type->CastToRawTypePtr<IntType>();
    EXPECT_TRUE(cast_int == nullptr);

    // 使用cast()方法测试
    TypePtr any_ptr = AnyType::Global();
    auto any_cast = any_ptr->CastTo<AnyType>();
    EXPECT_TRUE(any_cast);

    auto int_cast = any_ptr->CastTo<IntType>();
    EXPECT_FALSE(int_cast);

    // expect()方法测试 - 成功情况
    TypePtr int_ptr = IntType::Global();
    auto int_expect = int_ptr->Expect<IntType>();
    EXPECT_TRUE(int_expect);

    // expectRef()方法测试
    const Type& float_ref = *FloatType::Global();
    const auto& float_expect_ref = float_ref.ExpectRef<FloatType>();
    EXPECT_EQ(float_expect_ref.str(), "float");
}

// 子类型关系测试
TEST(TypeSystem, SubtypeRelationships) {
    // IntType应该是NumberType的子类型
    EXPECT_TRUE(IntType::Global()->IsSubtypeOf(*NumberType::Global()));
    EXPECT_FALSE(NumberType::Global()->IsSubtypeOf(*IntType::Global()));

    // FloatType应该是NumberType的子类型
    EXPECT_TRUE(FloatType::Global()->IsSubtypeOf(*NumberType::Global()));
    EXPECT_FALSE(NumberType::Global()->IsSubtypeOf(*FloatType::Global()));

    // ComplexType应该是NumberType的子类型
    EXPECT_TRUE(ComplexType::Global()->IsSubtypeOf(*NumberType::Global()));
    EXPECT_FALSE(NumberType::Global()->IsSubtypeOf(*ComplexType::Global()));

    // 相同类型应该是彼此的子类型
    EXPECT_TRUE(IntType::Global()->IsSubtypeOf(*IntType::Global()));

    // 不同分支的类型不应该有子类型关系
    EXPECT_FALSE(IntType::Global()->IsSubtypeOf(*FloatType::Global()));
    EXPECT_FALSE(FloatType::Global()->IsSubtypeOf(*IntType::Global()));
    EXPECT_FALSE(StringType::Global()->IsSubtypeOf(*NumberType::Global()));

    // 使用TypePtr测试子类型关系
    TypePtr int_ptr = IntType::Global();
    TypePtr num_ptr = NumberType::Global();
    EXPECT_TRUE(int_ptr->IsSubtypeOf(*num_ptr));
    EXPECT_FALSE(num_ptr->IsSubtypeOf(*int_ptr));

    // 带why_not参数的测试
    std::ostringstream why_not;
    bool result = FloatType::Global()->IsSubtypeOfExt(*StringType::Global(), &why_not);
    EXPECT_FALSE(result);
    // 检查是否有错误信息被写入
    // EXPECT_FALSE(why_not.str().empty());
}

// UnionType测试
TEST(TypeSystem, UnionType) {
    GTEST_SKIP();
    // 创建UnionType
    std::vector<TypePtr> types = {
            IntType::Global(),
            FloatType::Global(),
            StringType::Global()};
    auto union_type = UnionType::Create(types);

    EXPECT_TRUE(union_type);
    EXPECT_EQ(union_type->kind(), TypeKind::UnionType);
    EXPECT_TRUE(union_type->IsUnionType());

    // 检查包含的类型
    auto contained_types = union_type->GetContainedTypes();
    EXPECT_EQ(contained_types.size(), 3);

    // 检查canHoldType方法
    EXPECT_TRUE(union_type->canHoldType(*IntType::Global()));
    EXPECT_TRUE(union_type->canHoldType(*FloatType::Global()));
    EXPECT_TRUE(union_type->canHoldType(*StringType::Global()));
    EXPECT_FALSE(union_type->canHoldType(*NoneType::Global()));

    // 测试hasFreeVariables
    EXPECT_FALSE(union_type->HasFreeVars());

    // 测试空的UnionType
    std::vector<TypePtr> empty_types;
    auto empty_union = UnionType::Create(empty_types);
    EXPECT_TRUE(empty_union);
    EXPECT_EQ(empty_union->GetContainedTypes().size(), 0);
}

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
    EXPECT_EQ(t1->Annotation(), "Any");
    EXPECT_TRUE(t1->CastToRawTypePtr<AnyType>() == t1);
    EXPECT_TRUE(t1->CastTo<AnyType>().get() == t1);

    auto printer = [](const Type& t) -> std::optional<std::string> {
        return t.str() + "_test";
    };
    EXPECT_EQ(t1->Annotation(printer), "Any_test");

    TypePtr t3 = IntType::Global();
    EXPECT_EQ(t3->kind(), TypeKind::IntType);
    EXPECT_EQ(t3->str(), "int");
    EXPECT_EQ(t3->Annotation(), "int");

    const auto& t4 = *t3;
    EXPECT_EQ(t4.kind(), TypeKind::IntType);
    EXPECT_EQ(t4.str(), "int");
    EXPECT_EQ(t4.Annotation(), "int");
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

    EXPECT_TRUE(t4->IsSubtypeOf(t3));

    TypePtr union_type_1 = UnionType::Create({t1, t2, t4, t5, t5});
    EXPECT_EQ(union_type_1->kind(), TypeKind::OptionalType);
    // EXPECT_EQ(union_type_1->containedTypeSize(), 4);
}

}// namespace