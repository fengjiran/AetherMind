#include "aethermind/base/tensor.h"
#include "aethermind/memory/cpu_allocator.h"
#include "scalar.h"

#include "../test_utils/tensor_factory.h"

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

namespace {
using namespace aethermind;
using namespace aethermind::test_utils;

inline Allocator& GetCPUAllocator() {
    static CPUAllocatorProvider provider;
    static std::unique_ptr<Allocator> alloc = provider.CreateAllocator(Device::CPU());
    return *alloc;
}

// ── integer dtype round-trips ──────────────────────────────────────

TEST(TensorScalarBridge, Int8RoundTrip) {
    Scalar s(static_cast<int8_t>(-7));
    Tensor t = Tensor::FromScalar(s, GetCPUAllocator());
    EXPECT_EQ(t.dtype(), DataType::Int(8));
    EXPECT_EQ(t.rank(), 0);
    EXPECT_EQ(t.numel(), 1);
    EXPECT_EQ(t.buffer().nbytes(), 1U);
    Scalar s2 = t.item();
    EXPECT_EQ(s2.type(), DataType::Int(8));
    EXPECT_EQ(s2.toChar(), -7);
}

TEST(TensorScalarBridge, Int16RoundTrip) {
    Scalar s(static_cast<int16_t>(-42));
    Tensor t = Tensor::FromScalar(s, GetCPUAllocator());
    EXPECT_EQ(t.dtype(), DataType::Int(16));
    EXPECT_EQ(t.buffer().nbytes(), 2U);
    Scalar s2 = t.item();
    EXPECT_EQ(s2.type(), DataType::Int(16));
    EXPECT_EQ(s2.toShort(), -42);
}

TEST(TensorScalarBridge, Int32RoundTrip) {
    Scalar s(static_cast<int32_t>(12345));
    Tensor t = Tensor::FromScalar(s, GetCPUAllocator());
    EXPECT_EQ(t.dtype(), DataType::Int(32));
    EXPECT_EQ(t.buffer().nbytes(), 4U);
    Scalar s2 = t.item();
    EXPECT_EQ(s2.type(), DataType::Int(32));
    EXPECT_EQ(s2.toInt(), 12345);
}

TEST(TensorScalarBridge, Int64RoundTrip) {
    Scalar s(static_cast<int64_t>(-9999999));
    Tensor t = Tensor::FromScalar(s, GetCPUAllocator());
    EXPECT_EQ(t.dtype(), DataType::Int(64));
    EXPECT_EQ(t.buffer().nbytes(), 8U);
    Scalar s2 = t.item();
    EXPECT_EQ(s2.type(), DataType::Int(64));
    EXPECT_EQ(s2.toLong(), -9999999);
}

TEST(TensorScalarBridge, UInt8RoundTrip) {
    // Scalar stores uint8_t as Int(8); use value fitting in signed range
    Scalar s(static_cast<uint8_t>(42));
    Tensor t = Tensor::FromScalar(s, GetCPUAllocator());
    EXPECT_EQ(t.dtype(), DataType::Int(8));
    EXPECT_EQ(t.buffer().nbytes(), 1U);
    Scalar s2 = t.item();
    EXPECT_EQ(s2.toByte(), 42);
}

TEST(TensorScalarBridge, UInt16RoundTrip) {
    // Scalar stores uint16_t as Int(16); use value fitting in signed range
    Scalar s(static_cast<uint16_t>(12345));
    Tensor t = Tensor::FromScalar(s, GetCPUAllocator());
    EXPECT_EQ(t.dtype(), DataType::Int(16));
    EXPECT_EQ(t.buffer().nbytes(), 2U);
    Scalar s2 = t.item();
    EXPECT_EQ(s2.toUInt16(), 12345);
}

TEST(TensorScalarBridge, UInt32RoundTrip) {
    Scalar s(static_cast<uint32_t>(123456789));
    Tensor t = Tensor::FromScalar(s, GetCPUAllocator());
    EXPECT_EQ(t.dtype(), DataType::Int(32));
    EXPECT_EQ(t.buffer().nbytes(), 4U);
    Scalar s2 = t.item();
    EXPECT_EQ(s2.toUInt32(), 123456789U);
}

TEST(TensorScalarBridge, UInt64RoundTrip) {
    Scalar s(static_cast<uint64_t>(18000000000000000000ULL));
    Tensor t = Tensor::FromScalar(s, GetCPUAllocator());
    EXPECT_EQ(t.dtype(), DataType::UInt(64));
    EXPECT_EQ(t.buffer().nbytes(), 8U);
    Scalar s2 = t.item();
    EXPECT_EQ(s2.toUInt64(), 18000000000000000000ULL);
}

// ── bool round-trip ───────────────────────────────────────────────

TEST(TensorScalarBridge, BoolRoundTrip) {
    Scalar s(true);
    Tensor t = Tensor::FromScalar(s, GetCPUAllocator());
    EXPECT_EQ(t.dtype(), DataType::Bool());
    EXPECT_EQ(t.itemsize(), 1U);
    EXPECT_EQ(t.buffer().nbytes(), 1U);
    Scalar s2 = t.item();
    EXPECT_EQ(s2.type(), DataType::Bool());
    EXPECT_TRUE(s2.toBool());

    Scalar sf(false);
    Tensor tf = Tensor::FromScalar(sf, GetCPUAllocator());
    EXPECT_FALSE(tf.item().toBool());
}

// ── floating-point round-trips ────────────────────────────────────

TEST(TensorScalarBridge, HalfRoundTrip) {
    Scalar s(Half(1.5f));
    Tensor t = Tensor::FromScalar(s, GetCPUAllocator());
    EXPECT_EQ(t.dtype(), DataType::Float(16));
    EXPECT_EQ(t.buffer().nbytes(), 2U);
    Scalar s2 = t.item();
    EXPECT_EQ(s2.type(), DataType::Float(16));
    EXPECT_FLOAT_EQ(s2.toFloat(), 1.5f);
}

TEST(TensorScalarBridge, Float32RoundTrip) {
    Scalar s(3.14f);
    Tensor t = Tensor::FromScalar(s, GetCPUAllocator());
    EXPECT_EQ(t.dtype(), DataType::Float32());
    EXPECT_EQ(t.buffer().nbytes(), 4U);
    Scalar s2 = t.item();
    EXPECT_EQ(s2.type(), DataType::Float32());
    EXPECT_FLOAT_EQ(s2.toFloat(), 3.14f);
}

TEST(TensorScalarBridge, DoubleRoundTrip) {
    Scalar s(2.718281828);
    Tensor t = Tensor::FromScalar(s, GetCPUAllocator());
    EXPECT_EQ(t.dtype(), DataType::Double());
    EXPECT_EQ(t.buffer().nbytes(), 8U);
    Scalar s2 = t.item();
    EXPECT_EQ(s2.type(), DataType::Double());
    EXPECT_DOUBLE_EQ(s2.toDouble(), 2.718281828);
}

TEST(TensorScalarBridge, BFloat16RoundTrip) {
    Scalar s(BFloat16(2.5f));
    Tensor t = Tensor::FromScalar(s, GetCPUAllocator());
    EXPECT_EQ(t.dtype(), DataType::BFloat(16));
    EXPECT_EQ(t.buffer().nbytes(), 2U);
    Scalar s2 = t.item();
    EXPECT_EQ(s2.type(), DataType::BFloat(16));
    EXPECT_FLOAT_EQ(s2.toFloat(), 2.5f);
}

TEST(TensorScalarBridge, Float8E4M3FNRoundTrip) {
    Scalar s(Float8_e4m3fn(0.75f));
    Tensor t = Tensor::FromScalar(s, GetCPUAllocator());
    EXPECT_EQ(t.dtype(), DataType::Float8E4M3FN());
    EXPECT_EQ(t.buffer().nbytes(), 1U);
    Scalar s2 = t.item();
    EXPECT_EQ(s2.type(), DataType::Float8E4M3FN());
    EXPECT_FLOAT_EQ(s2.toFloat(), 0.75f);
}

TEST(TensorScalarBridge, Float8E5M2RoundTrip) {
    Scalar s(Float8_e5m2(0.5f));
    Tensor t = Tensor::FromScalar(s, GetCPUAllocator());
    EXPECT_EQ(t.dtype(), DataType::Float8E5M2());
    EXPECT_EQ(t.buffer().nbytes(), 1U);
    Scalar s2 = t.item();
    EXPECT_EQ(s2.type(), DataType::Float8E5M2());
    EXPECT_FLOAT_EQ(s2.toFloat(), 0.5f);
}

// ── complex round-trips ────────────────────────────────────────────

TEST(TensorScalarBridge, ComplexHalfRoundTrip) {
    Scalar s(complex<Half>(Half(1.0f), Half(2.0f)));
    Tensor t = Tensor::FromScalar(s, GetCPUAllocator());
    EXPECT_EQ(t.dtype(), DataType::ComplexHalf());
    EXPECT_EQ(t.buffer().nbytes(), 4U);
    Scalar s2 = t.item();
    EXPECT_EQ(s2.type(), DataType::ComplexHalf());
    auto c = s2.toComplexFloat();
    EXPECT_FLOAT_EQ(c.real(), 1.0f);
    EXPECT_FLOAT_EQ(c.imag(), 2.0f);
}

TEST(TensorScalarBridge, ComplexFloatRoundTrip) {
    Scalar s(complex<float>(1.5f, -2.5f));
    Tensor t = Tensor::FromScalar(s, GetCPUAllocator());
    EXPECT_EQ(t.dtype(), DataType::ComplexFloat());
    EXPECT_EQ(t.buffer().nbytes(), 8U);
    Scalar s2 = t.item();
    EXPECT_EQ(s2.type(), DataType::ComplexFloat());
    auto c = s2.toComplexFloat();
    EXPECT_FLOAT_EQ(c.real(), 1.5f);
    EXPECT_FLOAT_EQ(c.imag(), -2.5f);
}

TEST(TensorScalarBridge, ComplexDoubleRoundTrip) {
    Scalar s(complex<double>(3.0, 4.0));
    Tensor t = Tensor::FromScalar(s, GetCPUAllocator());
    EXPECT_EQ(t.dtype(), DataType::ComplexDouble());
    EXPECT_EQ(t.buffer().nbytes(), 16U);
    Scalar s2 = t.item();
    EXPECT_EQ(s2.type(), DataType::ComplexDouble());
    auto c = s2.toComplexDouble();
    EXPECT_DOUBLE_EQ(c.real(), 3.0);
    EXPECT_DOUBLE_EQ(c.imag(), 4.0);
}

// ── FromScalar: basic metadata ─────────────────────────────────────

TEST(TensorScalarBridge, FromScalarProducesRankZeroTensor) {
    Scalar s(42);
    Tensor t = Tensor::FromScalar(s, GetCPUAllocator());
    EXPECT_TRUE(t.is_initialized());
    EXPECT_EQ(t.rank(), 0);
    EXPECT_EQ(t.numel(), 1);
    EXPECT_TRUE(t.is_contiguous());
    EXPECT_TRUE(t.is_rank_zero());
    EXPECT_TRUE(t.shape().empty());
    EXPECT_TRUE(t.strides().empty());
    EXPECT_EQ(t.byte_offset(), 0U);
    EXPECT_TRUE(t.storage_range_is_valid());
    EXPECT_EQ(t.max_touched_element_offset(), 0);
}

// ── item(): validation gates ───────────────────────────────────────

TEST(TensorScalarBridge, ItemOnDefaultTensorDeath) {
    Tensor t;
    EXPECT_FALSE(t.is_initialized());
    EXPECT_DEATH(static_cast<void>(t.item()), "Check failed");
}

TEST(TensorScalarBridge, ItemOnTwoElementTensorDeath) {
    Tensor t = MakeContiguousTensor({2});
    ASSERT_EQ(t.numel(), 2);
    EXPECT_DEATH(static_cast<void>(t.item()), "Check failed");
}

TEST(TensorScalarBridge, ItemOnZeroElementTensorDeath) {
    Tensor t = MakeEmptyTensor();
    ASSERT_EQ(t.numel(), 0);
    EXPECT_DEATH(static_cast<void>(t.item()), "Check failed");
}

TEST(TensorScalarBridge, ItemOnNonScalarDtypeDeath) {
    // Fixed-length vector dtype: Float(32, 4) = 16 bytes for 4 lanes
    auto buffer = test_utils::detail::MakeBuffer(16, 64);
    ShapeAndStride sas;
    sas.set_contiguous(IntArrayView{});
    DataType vec_dtype = DataType::Float(32, 4);
    Tensor t(std::move(buffer), 0, vec_dtype, sas);
    ASSERT_EQ(t.numel(), 1);
    ASSERT_FALSE(t.dtype().IsScalar());
    EXPECT_DEATH(static_cast<void>(t.item()), "Check failed");
}

// ── set_item(): basic round-trip ───────────────────────────────────

TEST(TensorScalarBridge, SetItemThenItemReturnsSameValue) {
    Scalar s1(42);
    Tensor t = Tensor::FromScalar(s1, GetCPUAllocator());

    Scalar s2(-99);
    t.set_item(s2);

    Scalar retrieved = t.item();
    EXPECT_EQ(retrieved.toLong(), -99);
    EXPECT_EQ(retrieved.type(), s1.type());
}

// ── set_item(): cross-type conversion ──────────────────────────────

TEST(TensorScalarBridge, SetItemCrossTypeConversion) {
    Scalar f32(1.5f);
    Tensor t = Tensor::FromScalar(f32, GetCPUAllocator());
    EXPECT_EQ(t.dtype(), DataType::Float32());

    Scalar i64(static_cast<int64_t>(42));
    t.set_item(i64);

    Scalar retrieved = t.item();
    EXPECT_EQ(retrieved.type(), DataType::Float32());
    EXPECT_FLOAT_EQ(retrieved.toFloat(), 42.0f);
}

TEST(TensorScalarBridge, SetItemFloatIntoBFloat16) {
    Scalar bf16(BFloat16(1.0f));
    Tensor t = Tensor::FromScalar(bf16, GetCPUAllocator());
    EXPECT_EQ(t.dtype(), DataType::BFloat(16));

    Scalar f32(3.5f);
    t.set_item(f32);

    Scalar retrieved = t.item();
    EXPECT_EQ(retrieved.type(), DataType::BFloat(16));
    EXPECT_NEAR(retrieved.toFloat(), 3.5f, 0.1f);
}

// ── set_item(): overflow conversion ────────────────────────────────

TEST(TensorScalarBridge, SetItemOutOfRangeIntegerThrows) {
    Scalar i16(static_cast<int16_t>(42));
    Tensor t = Tensor::FromScalar(i16, GetCPUAllocator());
    EXPECT_EQ(t.dtype(), DataType::Int(16));

    Scalar big(static_cast<int64_t>(100000));
    EXPECT_THROW(t.set_item(big), Error);
}

// ── set_item(): validation gates ───────────────────────────────────

TEST(TensorScalarBridge, SetItemOnDefaultTensorDeath) {
    Tensor t;
    EXPECT_DEATH(t.set_item(Scalar(1)), "Check failed");
}

TEST(TensorScalarBridge, SetItemOnTwoElementTensorDeath) {
    Tensor t = MakeContiguousTensor({2});
    EXPECT_DEATH(t.set_item(Scalar(1)), "Check failed");
}

TEST(TensorScalarBridge, SetItemOnNonScalarDtypeDeath) {
    auto buffer = test_utils::detail::MakeBuffer(16, 64);
    ShapeAndStride sas;
    sas.set_contiguous(IntArrayView{});
    DataType vec_dtype = DataType::Float(32, 4);
    Tensor t(std::move(buffer), 0, vec_dtype, sas);
    ASSERT_EQ(t.numel(), 1);
    ASSERT_FALSE(t.dtype().IsScalar());
    EXPECT_DEATH(t.set_item(Scalar(1.0f)), "Check failed");
}

// ── item()/set_item() on [1] vector (numel==1, rank==1) ────────────

TEST(TensorScalarBridge, ItemOnVectorShapeOneWorks) {
    Tensor t = MakeContiguousTensor({1}, DataType::Int(32));
    *static_cast<int32_t*>(t.mutable_data()) = 77;

    Scalar s = t.item();
    EXPECT_EQ(s.type(), DataType::Int(32));
    EXPECT_EQ(s.toInt(), 77);
}

TEST(TensorScalarBridge, SetItemOnVectorShapeOneWorks) {
    Tensor t = MakeContiguousTensor({1}, DataType::Int(32));
    t.set_item(Scalar(static_cast<int32_t>(123)));
    Scalar s = t.item();
    EXPECT_EQ(s.toInt(), 123);
}

// ── shallow copy Buffer sharing (aliasing) ─────────────────────────

TEST(TensorScalarBridge, CopyAssignmentSharesBufferAliasing) {
    Scalar s(static_cast<int32_t>(10));
    Tensor t1 = Tensor::FromScalar(s, GetCPUAllocator());

    Tensor t2 = t1;
    ASSERT_TRUE(t1.is_initialized());
    ASSERT_TRUE(t2.is_initialized());
    EXPECT_EQ(t1.data(), t2.data());

    t2.set_item(Scalar(static_cast<int32_t>(99)));
    Scalar s1 = t1.item();
    EXPECT_EQ(s1.toInt(), 99);
}

TEST(TensorScalarBridge, MoveAssignmentLeavesMovedFromUninitialized) {
    Scalar s(static_cast<int32_t>(5));
    Tensor t1 = Tensor::FromScalar(s, GetCPUAllocator());
    const void* original_data = t1.data();

    Tensor t2 = std::move(t1);
    ASSERT_TRUE(t2.is_initialized());
    EXPECT_EQ(t2.data(), original_data);
    EXPECT_EQ(t2.item().toInt(), 5);

    EXPECT_FALSE(t1.is_initialized());
}

}// namespace
