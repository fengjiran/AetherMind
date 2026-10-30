#include "aethermind/model/raw_weight.h"

#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>

namespace {

using namespace aethermind;

struct Storage final : RawStorage {
    std::array<std::byte, 128> data{};
};

RawWeightView MakeView(DataType dtype) {
    auto storage = std::make_shared<Storage>();
    return {.data = storage->data.data(),
            .bytes = 16,
            .dtype = dtype,
            .shape = {2, 2},
            .storage = std::move(storage)};
}

TEST(RawWeightValidation, RejectsScalableAndReservedLaneEncodings) {
    for (const auto dtype: {DataType(DLDataTypeCode::kFloat, 16, 2, true),
                            DataType(DLDataTypeCode::kFloat, 32, 2, true),
                            DataType(DLDataTypeCode::kBFloat, 16, 2, true),
                            DataType::Float(32, 65535)}) {
        SCOPED_TRACE(ToString(dtype));
        auto view = MakeView(dtype);
        EXPECT_FALSE(view.IsValid());
        EXPECT_EQ(ValidateRawWeightView(view).code(), StatusCode::kInvalidArgument);
        view.shape = {0, 2};
        view.bytes = 0;
        EXPECT_EQ(ValidateRawWeightView(view).code(), StatusCode::kInvalidArgument);
    }
}

TEST(RawWeightValidation, RejectsZeroLaneElementDTypes) {
    // The integer-field constructor can also encode zero lanes by truncation.
    for (const auto dtype: {DataType::Handle(64, 0), DataType::Float(32, 65536)}) {
        auto view = MakeView(dtype);
        view.bytes = 0;
        EXPECT_FALSE(view.IsValid());
        EXPECT_EQ(ValidateRawWeightView(view).code(), StatusCode::kInvalidArgument);
    }
}

TEST(RawWeightValidation, RejectsUndefinedAndVoidElements) {
    for (const auto dtype: {DataType{}, DataType::Handle(0, 0), DataType::Handle(0, 1)}) {
        auto view = MakeView(dtype);
        view.bytes = 0;
        EXPECT_FALSE(view.IsValid());
        EXPECT_EQ(ValidateRawWeightView(view).code(), StatusCode::kInvalidArgument);
    }
}

TEST(RawWeightValidation, AcceptsEmptyWeightsWithNonzeroElementSize) {
    auto view = MakeView(DataType::Float32());
    view.data = nullptr;
    view.bytes = 0;
    view.shape = {0, 4};
    EXPECT_TRUE(view.IsValid());
    EXPECT_TRUE(ValidateRawWeightView(view).ok());
}

TEST(RawWeightValidation, CountsFixedVectorLanesInElementByteSize) {
    auto view = MakeView(DataType::Float(32, 4));
    view.shape = {2, 3};
    view.bytes = 96;
    EXPECT_TRUE(view.IsValid());
    EXPECT_TRUE(ValidateRawWeightView(view).ok());
    view.bytes = 24;
    EXPECT_EQ(ValidateRawWeightView(view).code(), StatusCode::kInvalidArgument);
}

TEST(RawWeightValidation, AcceptsScalarWeight) {
    auto view = MakeView(DataType::Float32());
    view.shape.clear();
    view.bytes = sizeof(float);
    EXPECT_TRUE(view.IsValid());
    EXPECT_TRUE(ValidateRawWeightView(view).ok());
}

} // namespace
