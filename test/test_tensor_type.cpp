//
// Created by richard on 10/22/25.
//
#include "type_system/tensor_type.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

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


// 测试TensorType的基本属性访问
TEST(TensorTypeTest, BasicProperties) {
    // 创建一个完整的TensorType
    // DataType dtype = DataType::Make<int32_t>();
    DataType dtype = DataType::Int(32);
    Device device(DeviceType::kCPU, 0);
    std::vector<int64_t> shape = {2, 3, 4};
    auto tensor_type = TensorType::CreateContiguous(dtype, device, shape);

    // 验证基本属性
    EXPECT_TRUE(tensor_type->dtype().has_value());
    EXPECT_EQ(tensor_type->dtype().value(), dtype);
    EXPECT_TRUE(tensor_type->device().has_value());
    EXPECT_EQ(tensor_type->device().value(), device);
    EXPECT_TRUE(tensor_type->ndim().has_value());
    EXPECT_EQ(tensor_type->ndim().value(), 3);
    EXPECT_FALSE(tensor_type->undefined().value_or(false));
    EXPECT_FALSE(tensor_type->is_inferred());
    EXPECT_TRUE(tensor_type->IsComplete());
    EXPECT_EQ(tensor_type->str(), "Tensor");
    EXPECT_EQ(tensor_type->ReprStr(), "Tensor");

    // 验证numel计算
    EXPECT_TRUE(tensor_type->numel().has_value());
    EXPECT_EQ(tensor_type->numel().value(), 24);// 2*3*4=24
}

// 测试TensorType的创建方法
TEST(TensorTypeTest, CreateMethods) {
    // 测试CreateContiguous方法
    DataType dtype = DataType::Make<float>();
    Device device(DeviceType::kCPU, 0);
    std::vector<int64_t> shape = {2, 3};
    auto contiguous_type = TensorType::CreateContiguous(dtype, device, shape);

    EXPECT_TRUE(contiguous_type->dtype().has_value());
    EXPECT_EQ(contiguous_type->dtype().value(), dtype);
    EXPECT_TRUE(contiguous_type->device().has_value());
    EXPECT_EQ(contiguous_type->device().value(), device);
    EXPECT_TRUE(contiguous_type->ndim().has_value());
    EXPECT_EQ(contiguous_type->ndim().value(), 2);

    // 测试Create方法（使用SymbolicShape）
    SymbolicShape sym_shape(shape);
    VaryingShape<Stride> strides;
    auto symbolic_type = TensorType::Create(dtype, device, sym_shape, strides, false);

    EXPECT_TRUE(symbolic_type->dtype().has_value());
    EXPECT_EQ(symbolic_type->dtype().value(), dtype);
    EXPECT_TRUE(symbolic_type->device().has_value());
    EXPECT_EQ(symbolic_type->device().value(), device);

    // 测试Create方法（使用VaryingShape）
    VaryingShape<int64_t> varying_shape(shape);
    VaryingShape<int64_t> varying_strides(std::vector<int64_t>{3, 1});// 对于3D形状

    auto varying_type = TensorType::Create(dtype, device, varying_shape, varying_strides, false);

    EXPECT_TRUE(varying_type->dtype().has_value());
    EXPECT_EQ(varying_type->dtype().value(), dtype);

    // 测试创建维度未知的TensorType
    auto dim_type = TensorType::Create(dtype, std::nullopt, 4, false);
    EXPECT_TRUE(dim_type->ndim().has_value());
    EXPECT_EQ(dim_type->ndim().value(), 4);
    EXPECT_FALSE(dim_type->device().has_value());
}

// 测试TensorType的With方法
TEST(TensorTypeTest, WithMethods) {
    DataType dtype = DataType::Make<int32_t>();
    Device device(DeviceType::kCPU, 0);
    std::vector<int64_t> shape = {2, 3};
    auto original = TensorType::CreateContiguous(dtype, device, shape);

    // 测试WithRequiresGrad
    auto with_grad = original->WithRequiresGrad(true);
    EXPECT_TRUE(with_grad->RequiresGrad().has_value());
    EXPECT_TRUE(with_grad->RequiresGrad().value());

    // 测试WithDataType
    DataType new_dtype = DataType::Make<float>();
    auto with_dtype = original->WithDataType(new_dtype);
    EXPECT_TRUE(with_dtype->dtype().has_value());
    EXPECT_EQ(with_dtype->dtype().value(), new_dtype);

    // 测试WithDevice
    Device new_device = Device::CUDA();
    auto with_device = original->WithDevice(new_device);
    EXPECT_TRUE(with_device->device().has_value());
    EXPECT_EQ(with_device->device().value(), new_device);

    // GTEST_SKIP();
    // 测试WithShape
    std::vector<int64_t> new_shape = {4, 5};
    auto with_shape = original->WithShape(new_shape);
    EXPECT_TRUE(with_shape->ndim().has_value());
    EXPECT_EQ(with_shape->ndim().value(), 2);

    // 测试WithDim
    auto with_dim = original->WithDim(4);
    EXPECT_TRUE(with_dim->ndim().has_value());
    EXPECT_EQ(with_dim->ndim().value(), 4);

    // 测试WithUndefined
    auto with_undefined = original->WithUndefined();
    EXPECT_TRUE(with_undefined->undefined().has_value());
    EXPECT_TRUE(with_undefined->undefined().value());
}

// 测试TensorType的Merge方法
TEST(TensorTypeTest, MergeMethod) {
    DataType dtype = DataType::Make<float>();
    Device device(DeviceType::kCPU, 0);

    // 创建两个可以合并的TensorType
    auto type1 = TensorType::CreateContiguous(dtype, device, {2, 3, 4});
    auto type2 = TensorType::CreateContiguous(dtype, device, {2, 3, 5});

    // 合并它们
    auto merged = type1->Merge(*type2);

    // 验证合并结果
    EXPECT_TRUE(merged->dtype().has_value());
    EXPECT_EQ(merged->dtype().value(), dtype);
    EXPECT_TRUE(merged->device().has_value());
    EXPECT_EQ(merged->device().value(), device);
    EXPECT_TRUE(merged->ndim().has_value());
    EXPECT_EQ(merged->ndim().value(), 3);
    EXPECT_FALSE(merged->IsComplete());

    auto shape = merged->shape();
    EXPECT_EQ(shape[0].value(), 2);
    EXPECT_EQ(shape[1].value(), 3);
    EXPECT_FALSE(shape[2].has_value());

    // 测试不合并形状的情况
    auto merged_no_shape = type1->Merge(*type2, false);
    EXPECT_TRUE(merged_no_shape->dtype().has_value());
    EXPECT_EQ(merged_no_shape->dtype().value(), dtype);
}

// 测试TensorType的Contiguity方法
TEST(TensorTypeTest, ContiguityMethod) {
    DataType dtype = DataType::Make<int64_t>();
    Device device(DeviceType::kCPU, 0);
    std::vector<int64_t> shape = {2, 3, 4};

    // 创建一个连续的TensorType
    auto original = TensorType::CreateContiguous(dtype, device, shape);

    // 测试Contiguity方法
    auto contiguous = original->Contiguity();

    // 验证结果
    EXPECT_TRUE(contiguous->dtype().has_value());
    EXPECT_EQ(contiguous->dtype().value(), dtype);
    EXPECT_TRUE(contiguous->device().has_value());
    EXPECT_EQ(contiguous->device().value(), device);
    EXPECT_TRUE(contiguous->ndim().has_value());
    EXPECT_EQ(contiguous->ndim().value(), 3);
}

// 测试TensorType的GetContiguousStrideOf静态方法
TEST(TensorTypeTest, GetContiguousStrideOf) {
    std::vector<int64_t> shape = {10, 2, 3, 4};

    // 测试默认的连续内存格式
    auto strides_default = TensorType::GetContiguousStrideOf(shape);
    EXPECT_EQ(strides_default.size(), 4);
    EXPECT_EQ(strides_default[0], 24);// 3*4
    EXPECT_EQ(strides_default[1], 12);
    EXPECT_EQ(strides_default[2], 4);
    EXPECT_EQ(strides_default[3], 1);

    auto strides_channels_last = TensorType::GetContiguousStrideOf(shape, MemoryFormat::kChannelsLast);
    EXPECT_FALSE(strides_channels_last.empty());
}

// 测试TensorType的Equals方法
TEST(TensorTypeTest, EqualsMethod) {
    DataType dtype = DataType::Make<float>();
    Device device(DeviceType::kCPU, 0);
    std::vector<int64_t> shape = {2, 3};

    // 创建两个相同的TensorType
    auto type1 = TensorType::CreateContiguous(dtype, device, shape);
    auto type2 = TensorType::CreateContiguous(dtype, device, shape);

    // 验证它们相等
    EXPECT_TRUE(type1->Equals(*type2));

    // 创建一个不同的TensorType
    DataType different_dtype = DataType::Make<int32_t>();
    auto type3 = TensorType::CreateContiguous(different_dtype, device, shape);

    // 验证它们不相等
    EXPECT_FALSE(type1->Equals(*type3));

    // 测试与基类Type的比较
    TypePtr type_ptr = type1;
    EXPECT_TRUE(type_ptr->Equals(*type2));
}

// 测试TensorType的IsComplete方法
TEST(TensorTypeTest, IsCompleteMethod) {
    DataType dtype = DataType::Make<float>();
    Device device(DeviceType::kCPU, 0);
    std::vector<int64_t> shape = {2, 3};

    // 创建一个完整的TensorType
    auto complete_type = TensorType::CreateContiguous(dtype, device, shape);
    EXPECT_TRUE(complete_type->IsComplete());

    // 创建一个不完整的TensorType（缺少device）
    auto incomplete_type = TensorType::Create(dtype, std::nullopt,
                                              SymbolicShape(shape), VaryingShape<Stride>{}, false);
    EXPECT_FALSE(incomplete_type->IsComplete());

    // 创建一个只有维度信息的TensorType
    auto dim_only_type = complete_type->WithDimensionOnly();
    EXPECT_FALSE(dim_only_type->IsComplete());
}

// 测试TensorType的静态get方法
TEST(TensorTypeTest, StaticGetMethod) {
    // 获取静态实例
    const auto& static_instance = TensorType::get();

    // 验证实例
    EXPECT_TRUE(static_instance != nullptr);
    EXPECT_EQ(static_instance->str(), "Tensor");
    EXPECT_FALSE(static_instance->dtype().has_value());
    EXPECT_FALSE(static_instance->device().has_value());
    EXPECT_FALSE(static_instance->ndim().has_value());
}

// 测试TensorType的边界情况
TEST(TensorTypeTest, EdgeCases) {
    // 测试0维张量
    DataType dtype = DataType::Make<int32_t>();
    Device device(DeviceType::kCPU, 0);
    std::vector<int64_t> scalar_shape = {};
    auto scalar_type = TensorType::CreateContiguous(dtype, device, scalar_shape);

    EXPECT_TRUE(scalar_type->ndim().has_value());
    EXPECT_EQ(scalar_type->ndim().value(), 0);
    EXPECT_TRUE(scalar_type->numel().has_value());
    EXPECT_EQ(scalar_type->numel().value(), 1);

    // 测试可能未定义的张量类型
    auto possible_undefined = scalar_type->WithPossibleUndefined();
    EXPECT_FALSE(possible_undefined->undefined().has_value());

    // 测试大维度
    std::vector<int64_t> large_shape = {1000000, 1000};
    auto large_type = TensorType::CreateContiguous(dtype, device, large_shape);
    EXPECT_TRUE(large_type->ndim().has_value());
    EXPECT_EQ(large_type->ndim().value(), 2);

    // 测试numel计算是否会溢出（如果支持的话）
    try {
        EXPECT_TRUE(large_type->numel().has_value());
    } catch (const std::overflow_error&) {
        GTEST_SKIP() << "numel calculation overflowed";
    }
}

// 测试TensorType与其他类型的关系
TEST(TensorTypeTest, TypeRelations) {
    // 验证Kind枚举值
    EXPECT_EQ(TensorType::Kind, TypeKind::TensorType);

    // 测试创建从布尔类型转换的张量类型
    TypePtr tensor_from_bool = TensorType::CreateFromBoolType();
    EXPECT_TRUE(tensor_from_bool != nullptr);

    // 测试创建从数值类型转换的张量类型
    TypePtr int_type = IntType::Global();
    TypePtr tensor_from_int = TensorType::CreateFromNumberType(*int_type);
    EXPECT_TRUE(tensor_from_int != nullptr);
}

}// namespace