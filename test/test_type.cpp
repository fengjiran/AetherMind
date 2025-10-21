//
// Created by 赵丹 on 2025/8/14.
//
#include "type_system/tensor_type.h"
#include "type_system/union_type.h"

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

// ShapeSymbol 测试组
TEST(ShapeSymbolTest, BasicOperations) {
    // 测试默认构造函数
    ShapeSymbol default_sym;
    EXPECT_FALSE(default_sym.IsStatic());
    EXPECT_EQ(default_sym.value(), -1);

    // 测试创建静态大小
    ShapeSymbol static_sym = ShapeSymbol::CreateFromValue(10);
    EXPECT_TRUE(static_sym.IsStatic());
    EXPECT_EQ(static_sym.value(), 10);
    EXPECT_EQ(static_sym.GetStaticValue(), 10);

    // 测试创建动态符号
    ShapeSymbol dynamic_sym = ShapeSymbol::Create();
    EXPECT_FALSE(dynamic_sym.IsStatic());
    EXPECT_LT(dynamic_sym.value(), 0);

    // 测试创建多个动态符号，应该有不同的值
    ShapeSymbol dyn1 = ShapeSymbol::Create();
    ShapeSymbol dyn2 = ShapeSymbol::Create();
    EXPECT_NE(dyn1.value(), dyn2.value());

    // 测试比较运算符
    ShapeSymbol static1 = ShapeSymbol::CreateFromValue(5);
    ShapeSymbol static2 = ShapeSymbol::CreateFromValue(5);
    EXPECT_EQ(static1, static2);
    EXPECT_FALSE(static1 < static2);

    ShapeSymbol static3 = ShapeSymbol::CreateFromValue(10);
    EXPECT_LT(static1, static3);
    EXPECT_FALSE(static3 < static1);
}

TEST(ShapeSymbolTest, MergePrimitive) {
    // 测试合并相同的静态符号
    ShapeSymbol s1 = ShapeSymbol::CreateFromValue(42);
    ShapeSymbol s2 = ShapeSymbol::CreateFromValue(42);
    ShapeSymbol merged = MergePrimitiveValue(s1, s2);
    EXPECT_TRUE(merged.IsStatic());
    EXPECT_EQ(merged.GetStaticValue(), 42);

    // 测试合并不同的静态符号
    ShapeSymbol s3 = ShapeSymbol::CreateFromValue(10);
    ShapeSymbol merged2 = MergePrimitiveValue(s1, s3);
    EXPECT_FALSE(merged2.IsStatic());

    // 测试合并静态和动态符号
    ShapeSymbol dyn = ShapeSymbol::Create();
    ShapeSymbol merged3 = MergePrimitiveValue(s1, dyn);
    EXPECT_FALSE(merged3.IsStatic());

    // 测试合并两个动态符号
    ShapeSymbol dyn2 = ShapeSymbol::Create();
    ShapeSymbol merged4 = MergePrimitiveValue(dyn, dyn2);
    EXPECT_FALSE(merged4.IsStatic());
}

// SymbolicShape 测试组
TEST(SymbolicShapeTest, Constructors) {
    // 测试默认构造函数（无秩）
    SymbolicShape unranked;
    EXPECT_FALSE(unranked.rank().has_value());
    EXPECT_FALSE(unranked.shape().has_value());

    // 测试已知秩但未知维度的构造函数
    SymbolicShape rank_3(3);
    EXPECT_TRUE(rank_3.rank().has_value());
    EXPECT_EQ(rank_3.rank().value(), 3);
    EXPECT_TRUE(rank_3.shape().has_value());
    EXPECT_EQ(rank_3.shape().value().size(), 3);
    EXPECT_FALSE(rank_3.IsComplete());

    // 测试从部分已知维度构造
    std::vector<std::optional<int64_t>> partial_dims = {10, std::nullopt, 20};
    SymbolicShape partial(partial_dims);
    EXPECT_TRUE(partial.rank().has_value());
    EXPECT_EQ(partial.rank().value(), 3);
    EXPECT_TRUE(partial.shape().has_value());
    EXPECT_TRUE(partial.shape().value()[0].IsStatic());
    EXPECT_FALSE(partial.shape().value()[1].IsStatic());
    EXPECT_TRUE(partial.shape().value()[2].IsStatic());
    EXPECT_FALSE(partial.IsComplete());

    // 测试从具体形状构造
    std::vector<int64_t> concrete_dims = {2, 3, 4};
    SymbolicShape concrete(IntArrayView{concrete_dims});
    EXPECT_TRUE(concrete.rank().has_value());
    EXPECT_EQ(concrete.rank().value(), 3);
    EXPECT_TRUE(concrete.IsComplete());

    // 测试从ShapeSymbol向量构造
    std::vector<ShapeSymbol> symbols = {
            ShapeSymbol::CreateFromValue(5),
            ShapeSymbol::Create(),
            ShapeSymbol::CreateFromValue(6)};
    SymbolicShape from_symbols(symbols);
    EXPECT_TRUE(from_symbols.rank().has_value());
    EXPECT_EQ(from_symbols.rank().value(), 3);
}

TEST(SymbolicShapeTest, Accessors) {
    std::vector<int64_t> dims = {2, 3, 4};
    SymbolicShape shape(IntArrayView{dims});

    // 测试operator[]访问
    EXPECT_EQ(shape[0].GetStaticValue(), 2);
    EXPECT_EQ(shape[1].GetStaticValue(), 3);
    EXPECT_EQ(shape[2].GetStaticValue(), 4);

    // 测试at()访问
    EXPECT_EQ(shape.at(0).GetStaticValue(), 2);
    EXPECT_EQ(shape.at(1).GetStaticValue(), 3);
    EXPECT_EQ(shape.at(2).GetStaticValue(), 4);

    // 测试symbolic_dims方法
    auto sym_dims = shape.GetSymbolicDims();
    EXPECT_TRUE(sym_dims.has_value());
    EXPECT_EQ(sym_dims.value().size(), 3);
    EXPECT_FALSE(sym_dims.value()[0]);// 第一个维度是静态的
    EXPECT_FALSE(sym_dims.value()[1]);// 第二个维度是静态的
    EXPECT_FALSE(sym_dims.value()[2]);// 第三个维度是静态的

    // 测试边界情况 - 无秩形状访问
    SymbolicShape unranked;
    EXPECT_THROW(unranked[0], Error);
    EXPECT_THROW(UNUSED(unranked.at(0)), Error);

    // 测试边界情况 - 越界访问
    EXPECT_THROW(UNUSED(shape.at(10)), Error);
}

TEST(SymbolicShapeTest, Merge) {
    // 测试合并两个相同的具体形状
    SymbolicShape shape1(IntArrayView({2, 3, 4}));
    SymbolicShape shape2(IntArrayView({2, 3, 4}));
    SymbolicShape merged = shape1.Merge(shape2);
    EXPECT_TRUE(merged.IsComplete());
    EXPECT_TRUE(merged.rank().has_value());
    EXPECT_EQ(merged.rank().value(), 3);
    EXPECT_EQ(merged[0].GetStaticValue(), 2);

    // 测试合并不同的具体形状
    SymbolicShape shape3(IntArrayView({2, 4, 4}));
    SymbolicShape merged2 = shape1.Merge(shape3);
    EXPECT_FALSE(merged2.IsComplete());
    EXPECT_TRUE(merged2.rank().has_value());
    EXPECT_EQ(merged2.rank().value(), 3);

    // 测试合并无秩和有秩形状
    SymbolicShape unranked;
    SymbolicShape merged3 = shape1.Merge(unranked);
    EXPECT_FALSE(merged3.rank().has_value());

    // 测试合并不同秩的形状
    SymbolicShape rank_2(2);
    SymbolicShape merged4 = shape1.Merge(rank_2);
    EXPECT_FALSE(merged4.rank().has_value());
}

// Stride 测试组
TEST(StrideTest, BasicOperations) {
    // 测试默认构造函数
    Stride default_stride;
    EXPECT_FALSE(default_stride.IsComplete());
    EXPECT_FALSE(default_stride.stride_idx().has_value());
    EXPECT_FALSE(default_stride.is_contiguous().has_value());
    EXPECT_FALSE(default_stride.stride().has_value());

    // 测试完全指定的构造函数
    Stride complete(1, true, 10);
    EXPECT_TRUE(complete.IsComplete());
    EXPECT_TRUE(complete.stride_idx().has_value());
    EXPECT_EQ(complete.stride_idx().value(), 1);
    EXPECT_TRUE(complete.is_contiguous().has_value());
    EXPECT_TRUE(complete.is_contiguous().value());
    EXPECT_TRUE(complete.stride().has_value());
    EXPECT_EQ(complete.stride().value(), 10);

    // 测试部分指定的构造函数
    Stride partial(2, std::nullopt, 20);
    EXPECT_FALSE(partial.IsComplete());
    EXPECT_TRUE(partial.stride_idx().has_value());
    EXPECT_FALSE(partial.is_contiguous().has_value());
    EXPECT_TRUE(partial.stride().has_value());

    // 测试比较运算符
    Stride complete2(1, true, 10);
    EXPECT_EQ(complete, complete2);

    Stride different(1, false, 10);
    EXPECT_NE(complete, different);
}

TEST(StrideTest, MergePrimitive) {
    // 测试合并两个完整且相同的Stride
    std::optional<Stride> s1 = Stride(1, true, 10);
    std::optional<Stride> s2 = Stride(1, true, 10);
    auto merged = MergePrimitiveValue(s1, s2);
    EXPECT_TRUE(merged.has_value());
    EXPECT_TRUE(merged.value().IsComplete());
    EXPECT_EQ(merged.value().stride_idx().value(), 1);

    // 测试合并两个完整但不同的Stride
    std::optional<Stride> s3 = Stride(2, true, 10);
    auto merged2 = MergePrimitiveValue(s1, s3);
    EXPECT_TRUE(merged2.has_value());
    EXPECT_FALSE(merged2.value().IsComplete());

    // 测试合并空和非空Stride
    std::optional<Stride> none;
    auto merged3 = MergePrimitiveValue(s1, none);
    EXPECT_FALSE(merged3.has_value());

    // 测试合并两个空Stride
    auto merged4 = MergePrimitiveValue(none, none);
    EXPECT_FALSE(merged4.has_value());
}

// VaryingShape 测试组 (以int64_t为例)
TEST(VaryingShapeTest, BasicOperations) {
    // 测试默认构造函数（无秩）
    VaryingShape<int64_t> unranked;
    EXPECT_FALSE(unranked.size().has_value());
    EXPECT_FALSE(unranked.shape().has_value());

    // 测试指定秩但无具体值的构造函数
    VaryingShape<int64_t> rank_3(3);
    EXPECT_TRUE(rank_3.size().has_value());
    EXPECT_EQ(rank_3.size().value(), 3);
    EXPECT_FALSE(rank_3[0].has_value());

    // 测试从具体值向量构造
    std::vector<int64_t> values = {2, 3, 4};
    VaryingShape<int64_t> concrete(values);
    EXPECT_TRUE(concrete.size().has_value());
    EXPECT_EQ(concrete.size().value(), 3);
    EXPECT_TRUE(concrete[0].has_value());
    EXPECT_EQ(concrete[0].value(), 2);

    // 测试从ArrayView构造
    IntArrayView view(values);
    VaryingShape<int64_t> from_view(view);
    EXPECT_TRUE(from_view.size().has_value());
    EXPECT_EQ(from_view.size().value(), 3);
    EXPECT_TRUE(from_view[0].has_value());
    EXPECT_EQ(from_view[0].value(), 2);

    // 测试从可选元素列表构造
    std::vector<std::optional<int64_t>> optional_values = {5, std::nullopt, 6};
    VaryingShape<int64_t> partial(optional_values);
    EXPECT_TRUE(partial.size().has_value());
    EXPECT_EQ(partial.size().value(), 3);
    EXPECT_TRUE(partial[0].has_value());
    EXPECT_FALSE(partial[1].has_value());
    EXPECT_TRUE(partial[2].has_value());
}

TEST(VaryingShapeTest, Accessors) {
    std::vector<std::optional<int64_t>> values = {5, std::nullopt, 6};
    VaryingShape<int64_t> shape(values);

    // 测试operator[]访问
    EXPECT_TRUE(shape[0].has_value());
    EXPECT_EQ(shape[0].value(), 5);
    EXPECT_FALSE(shape[1].has_value());
    EXPECT_TRUE(shape[2].has_value());
    EXPECT_EQ(shape[2].value(), 6);

    // 测试边界情况 - 无秩形状访问
    VaryingShape<int64_t> unranked;
    EXPECT_THROW(unranked[0], Error);
}

TEST(VaryingShapeTest, ConcreteSizes) {
    // 测试完全指定的形状
    std::vector<int64_t> values = {2, 3, 4};
    VaryingShape<int64_t> concrete(values);
    auto concrete_sizes = concrete.GetConcreteValue();
    EXPECT_TRUE(concrete_sizes.has_value());
    EXPECT_EQ(concrete_sizes.value().size(), 3);
    EXPECT_EQ(concrete_sizes.value()[0], 2);

    // 测试部分指定的形状
    std::vector<std::optional<int64_t>> partial_values = {5, std::nullopt, 6};
    VaryingShape<int64_t> partial(partial_values);
    auto partial_sizes = partial.GetConcreteValue();
    EXPECT_FALSE(partial_sizes.has_value());

    // 测试无秩形状
    VaryingShape<int64_t> unranked;
    auto unranked_sizes = unranked.GetConcreteValue();
    EXPECT_FALSE(unranked_sizes.has_value());
}

TEST(VaryingShapeTest, Merge) {
    // 测试合并两个相同的形状
    VaryingShape<int64_t> shape1(std::vector<int64_t>{2, 3, 4});
    VaryingShape<int64_t> shape2(std::vector<int64_t>{2, 3, 4});
    VaryingShape<int64_t> merged = shape1.Merge(shape2);
    EXPECT_TRUE(merged.size().has_value());
    EXPECT_EQ(merged.size().value(), 3);
    EXPECT_TRUE(merged[0].has_value());
    EXPECT_EQ(merged[0].value(), 2);

    // 测试合并不同秩的形状
    VaryingShape<int64_t> rank_2(2);
    VaryingShape<int64_t> merged2 = shape1.Merge(rank_2);
    EXPECT_FALSE(merged2.size().has_value());

    // 测试合并无秩和有秩形状
    VaryingShape<int64_t> unranked;
    VaryingShape<int64_t> merged3 = shape1.Merge(unranked);
    EXPECT_FALSE(merged3.size().has_value());
}

// VaryingShape<ShapeSymbol> 特殊测试
TEST(VaryingShapeShapeSymbolTest, BasicOperations) {
    // 测试从ShapeSymbol向量构造
    std::vector<ShapeSymbol> symbols = {
            ShapeSymbol::CreateFromValue(5),
            ShapeSymbol::Create(),
            ShapeSymbol::CreateFromValue(6)};
    VaryingShape<ShapeSymbol> shape(symbols);

    EXPECT_TRUE(shape.size().has_value());
    EXPECT_EQ(shape.size().value(), 3);
    EXPECT_TRUE(shape[0].has_value());
    EXPECT_TRUE(shape[0].value().IsStatic());
    EXPECT_FALSE(shape[1].value().IsStatic());
}

// VaryingShape<Stride> 特殊测试
TEST(VaryingShapeStrideTest, BasicOperations) {
    // 测试从Stride向量构造
    std::vector<Stride> strides = {
            Stride(0, true, 1),
            Stride(1, false, 3),
            Stride(2, true, 6)};
    VaryingShape<Stride> shape(strides);

    EXPECT_TRUE(shape.size().has_value());
    EXPECT_EQ(shape.size().value(), 3);
    EXPECT_TRUE(shape[0].has_value());
    EXPECT_TRUE(shape[0].value().IsComplete());
}

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
    auto union_type = UnionType::create(types);

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
    auto empty_union = UnionType::create(empty_types);
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

    TypePtr union_type_1 = UnionType::create({t1, t2, t4, t5, t5});
    EXPECT_EQ(union_type_1->kind(), TypeKind::OptionalType);
    // EXPECT_EQ(union_type_1->containedTypeSize(), 4);
}

}// namespace