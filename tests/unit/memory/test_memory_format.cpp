//
// Created by richard on 10/13/25.
//
#include "memory_format.h"
#include <gtest/gtest.h>

using namespace aethermind;

namespace {

// 测试MemoryFormat枚举值
TEST(MemoryFormatTest, EnumValues) {
    EXPECT_EQ(static_cast<uint8_t>(MemoryFormat::kContiguous), 0);
    EXPECT_EQ(static_cast<uint8_t>(MemoryFormat::kPreserve), 1);
    EXPECT_EQ(static_cast<uint8_t>(MemoryFormat::kChannelsLast), 2);
    EXPECT_EQ(static_cast<uint8_t>(MemoryFormat::kChannelsLast3d), 3);
    EXPECT_EQ(static_cast<uint8_t>(MemoryFormat::NumOptions), 4);
}

// 测试get_contiguous_memory_format函数
TEST(MemoryFormatTest, GetContiguousMemoryFormat) {
    EXPECT_EQ(GetContiguousMemoryFormat(), MemoryFormat::kContiguous);
}

// 测试get_channels_last_strides_2d函数 - 4维情况
TEST(MemoryFormatTest, GetChannelsLastStrides2D_4D) {
    std::vector<int64_t> shape = {2, 3, 4, 5};             // N, C, H, W
    std::vector<int64_t> expected_strides = {60, 1, 15, 3};// NCHW格式的步长

    std::vector<int64_t> actual_strides = GetChannelsLastStrides2d(shape);

    EXPECT_EQ(actual_strides.size(), 4);
    EXPECT_EQ(actual_strides[0], expected_strides[0]);
    EXPECT_EQ(actual_strides[1], expected_strides[1]);
    EXPECT_EQ(actual_strides[2], expected_strides[2]);
    EXPECT_EQ(actual_strides[3], expected_strides[3]);
}

// 测试get_channels_last_strides_3d函数 - 5维情况
TEST(MemoryFormatTest, GetChannelsLastStrides3D_5D) {
    std::vector<int64_t> shape = {2, 3, 4, 5, 6};               // N, C, D, H, W
    std::vector<int64_t> expected_strides = {360, 1, 90, 18, 3};// NCDHW格式的步长

    std::vector<int64_t> actual_strides = GetChannelsLastStrides3d(shape);

    EXPECT_EQ(actual_strides.size(), 5);
    EXPECT_EQ(actual_strides[0], expected_strides[0]);
    EXPECT_EQ(actual_strides[1], expected_strides[1]);
    EXPECT_EQ(actual_strides[2], expected_strides[2]);
    EXPECT_EQ(actual_strides[3], expected_strides[3]);
    EXPECT_EQ(actual_strides[4], expected_strides[4]);
}

// 测试is_channels_last_strides_2d函数 - 有效情况
TEST(MemoryFormatTest, IsChannelsLastStrides2D_Valid) {
    std::vector<int64_t> shape = {2, 3, 4, 5};    // N, C, H, W
    std::vector<int64_t> strides = {60, 1, 15, 3};// 有效的NHWC格式步长

    EXPECT_TRUE(IsChannelsLastStrides2d(shape, strides));
}

// 测试is_channels_last_strides_2d函数 - 无效情况
TEST(MemoryFormatTest, IsChannelsLastStrides2D_Invalid) {
    std::vector<int64_t> shape = {2, 3, 4, 5};    // N, C, H, W
    std::vector<int64_t> strides = {60, 20, 5, 1};// 标准NCHW格式步长

    EXPECT_FALSE(IsChannelsLastStrides2d(shape, strides));

    // 测试不支持的维度大小
    std::vector<int64_t> shape_5d = {2, 3, 4, 5, 6};// 5维形状
    EXPECT_FALSE(IsChannelsLastStrides2d(shape_5d, strides));
}

// 测试is_channels_last_strides_3d函数 - 有效情况
TEST(MemoryFormatTest, IsChannelsLastStrides3D_Valid) {
    std::vector<int64_t> shape = {2, 3, 4, 5, 6};      // N, C, D, H, W
    std::vector<int64_t> strides = {360, 1, 90, 18, 3};// 有效的NDHWC格式步长

    EXPECT_TRUE(IsChannelsLastStrides3d(shape, strides));
}

// 测试is_channels_last_strides_3d函数 - 无效情况
TEST(MemoryFormatTest, IsChannelsLastStrides3D_Invalid) {
    std::vector<int64_t> shape = {2, 3, 4, 5, 6};       // N, C, D, H, W
    std::vector<int64_t> strides = {360, 120, 30, 6, 1};// 标准NCDHW格式步长

    EXPECT_FALSE(IsChannelsLastStrides3d(shape, strides));

    // 测试不支持的维度大小
    std::vector<int64_t> shape_4d = {2, 3, 4, 5};// 4维形状
    EXPECT_FALSE(IsChannelsLastStrides3d(shape_4d, strides));
}

// 测试operator<<函数
TEST(MemoryFormatTest, OperatorStream) {
    std::ostringstream oss;

    oss << MemoryFormat::kPreserve;
    EXPECT_EQ(oss.str(), "Preserve");
    oss.str("");

    oss << MemoryFormat::kContiguous;
    EXPECT_EQ(oss.str(), "Contiguous");
    oss.str("");

    oss << MemoryFormat::kChannelsLast;
    EXPECT_EQ(oss.str(), "ChannelsLast");
    oss.str("");

    oss << MemoryFormat::kChannelsLast3d;
    EXPECT_EQ(oss.str(), "ChannelsLast3d");
}

// 测试模板函数与不同类型
TEST(MemoryFormatTest, TemplateFunctionsWithDifferentTypes) {
    // 测试size_t类型
    std::vector<size_t> shape_size_t = {2, 3, 4, 5};
    std::vector<size_t> strides_size_t = GetChannelsLastStrides2d<size_t>(shape_size_t);
    EXPECT_EQ(strides_size_t.size(), 4);

    // 测试int类型
    std::vector<int> shape_int = {2, 3, 4, 5};
    std::vector<int> strides_int = GetChannelsLastStrides2d<int>(shape_int);
    EXPECT_EQ(strides_int.size(), 4);
}

}// namespace