#ifndef AETHERMIND_TEST_UTILS_TENSOR_ASSERT_H
#define AETHERMIND_TEST_UTILS_TENSOR_ASSERT_H

#include "aethermind/base/tensor.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>

namespace aethermind::test_utils {

namespace detail {

template<typename T>
::testing::AssertionResult CompareAllCloseFloating(const T* actual,
                                                    const T* expected,
                                                    int64_t n,
                                                    double atol,
                                                    double rtol,
                                                    int64_t max_report) {
    int64_t mismatch_count = 0;
    std::ostringstream report;
    for (int64_t i = 0; i < n; ++i) {
        const double a = static_cast<double>(actual[i]);
        const double b = static_cast<double>(expected[i]);
        const double limit = atol + rtol * std::max(std::abs(a), std::abs(b));
        const double diff = std::abs(a - b);
        if (diff > limit) {
            ++mismatch_count;
            if (mismatch_count <= max_report) {
                report << " idx=" << i << " actual=" << a << " expected=" << b
                       << " diff=" << diff << " limit=" << limit;
            }
        }
    }

    if (mismatch_count == 0) {
        return ::testing::AssertionSuccess();
    }

    return ::testing::AssertionFailure()
           << "allclose mismatches=" << mismatch_count << ", showing up to " << max_report << ":"
           << report.str();
}

template<typename T>
::testing::AssertionResult CompareExact(const T* actual,
                                         const T* expected,
                                         int64_t n,
                                         int64_t max_report) {
    int64_t mismatch_count = 0;
    std::ostringstream report;
    for (int64_t i = 0; i < n; ++i) {
        if (actual[i] != expected[i]) {
            ++mismatch_count;
            if (mismatch_count <= max_report) {
                report << " idx=" << i << " actual=" << static_cast<long double>(actual[i])
                       << " expected=" << static_cast<long double>(expected[i]);
            }
        }
    }

    if (mismatch_count == 0) {
        return ::testing::AssertionSuccess();
    }

    return ::testing::AssertionFailure()
           << "exact mismatches=" << mismatch_count << ", showing up to " << max_report << ":"
           << report.str();
}

inline ::testing::AssertionResult CompareExactBool(const bool* actual,
                                                    const bool* expected,
                                                    int64_t n,
                                                    int64_t max_report) {
    int64_t mismatch_count = 0;
    std::ostringstream report;
    for (int64_t i = 0; i < n; ++i) {
        if (actual[i] != expected[i]) {
            ++mismatch_count;
            if (mismatch_count <= max_report) {
                report << " idx=" << i << " actual=" << static_cast<int>(actual[i])
                       << " expected=" << static_cast<int>(expected[i]);
            }
        }
    }

    if (mismatch_count == 0) {
        return ::testing::AssertionSuccess();
    }

    return ::testing::AssertionFailure()
           << "exact mismatches=" << mismatch_count << ", showing up to " << max_report << ":"
           << report.str();
}

template<typename T>
::testing::AssertionResult CompareExactReducedFloatBits(const T* actual,
                                                         const T* expected,
                                                         int64_t n,
                                                         int64_t max_report) {
    int64_t mismatch_count = 0;
    std::ostringstream report;
    for (int64_t i = 0; i < n; ++i) {
        if (actual[i].x != expected[i].x) {
            ++mismatch_count;
            if (mismatch_count <= max_report) {
                report << " idx=" << i << " actual_bits=" << actual[i].x
                       << " expected_bits=" << expected[i].x;
            }
        }
    }

    if (mismatch_count == 0) {
        return ::testing::AssertionSuccess();
    }

    return ::testing::AssertionFailure()
           << "exact mismatches=" << mismatch_count << ", showing up to " << max_report << ":"
           << report.str();
}

}// namespace detail

inline ::testing::AssertionResult ExpectTensorAllClose(const Tensor& actual,
                                                         const Tensor& expected,
                                                         double atol = 1e-6,
                                                         double rtol = 1e-6,
                                                         int64_t max_report = 5) {
    if (actual.dtype() != expected.dtype()) {
        return ::testing::AssertionFailure()
               << "dtype mismatch: actual=" << actual.dtype() << ", expected=" << expected.dtype();
    }

    if (actual.shape() != expected.shape()) {
        return ::testing::AssertionFailure()
               << "shape mismatch: actual=" << actual.shape() << ", expected=" << expected.shape();
    }

    if (actual.numel() != expected.numel()) {
        return ::testing::AssertionFailure()
               << "numel mismatch: actual=" << actual.numel() << ", expected=" << expected.numel();
    }

    const int64_t n = actual.numel();
    if (n == 0) {
        return ::testing::AssertionSuccess();
    }

    if (actual.dtype() == DataType::Float32()) {
        return detail::CompareAllCloseFloating<float>(
                static_cast<const float*>(actual.data()),
                static_cast<const float*>(expected.data()), n, atol, rtol, max_report);
    }

    if (actual.dtype() == DataType::Double()) {
        return detail::CompareAllCloseFloating<double>(
                static_cast<const double*>(actual.data()),
                static_cast<const double*>(expected.data()), n, atol, rtol, max_report);
    }

    if (actual.dtype() == DataType::Make<Half>()) {
        return detail::CompareAllCloseFloating<Half>(
                static_cast<const Half*>(actual.data()),
                static_cast<const Half*>(expected.data()), n, atol, rtol, max_report);
    }

    if (actual.dtype() == DataType::Make<BFloat16>()) {
        return detail::CompareAllCloseFloating<BFloat16>(
                static_cast<const BFloat16*>(actual.data()),
                static_cast<const BFloat16*>(expected.data()), n, atol, rtol, max_report);
    }

    return ::testing::AssertionFailure()
           << "ExpectTensorAllClose supports floating dtypes only, got " << actual.dtype();
}

inline ::testing::AssertionResult ExpectTensorEqual(const Tensor& actual,
                                                      const Tensor& expected,
                                                      int64_t max_report = 5) {
    if (actual.dtype() != expected.dtype()) {
        return ::testing::AssertionFailure()
               << "dtype mismatch: actual=" << actual.dtype() << ", expected=" << expected.dtype();
    }

    if (actual.shape() != expected.shape()) {
        return ::testing::AssertionFailure()
               << "shape mismatch: actual=" << actual.shape() << ", expected=" << expected.shape();
    }

    if (actual.numel() != expected.numel()) {
        return ::testing::AssertionFailure()
               << "numel mismatch: actual=" << actual.numel() << ", expected=" << expected.numel();
    }

    const int64_t n = actual.numel();
    if (n == 0) {
        return ::testing::AssertionSuccess();
    }

    if (actual.dtype() == DataType::Float32()) {
        return detail::CompareExact<float>(
                static_cast<const float*>(actual.data()),
                static_cast<const float*>(expected.data()), n, max_report);
    }

    if (actual.dtype() == DataType::Double()) {
        return detail::CompareExact<double>(
                static_cast<const double*>(actual.data()),
                static_cast<const double*>(expected.data()), n, max_report);
    }

    if (actual.dtype() == DataType::Make<Half>()) {
        return detail::CompareExactReducedFloatBits<Half>(
                static_cast<const Half*>(actual.data()),
                static_cast<const Half*>(expected.data()), n, max_report);
    }

    if (actual.dtype() == DataType::Make<BFloat16>()) {
        return detail::CompareExactReducedFloatBits<BFloat16>(
                static_cast<const BFloat16*>(actual.data()),
                static_cast<const BFloat16*>(expected.data()), n, max_report);
    }

    if (actual.dtype() == DataType::Int(8)) {
        return detail::CompareExact<int8_t>(
                static_cast<const int8_t*>(actual.data()),
                static_cast<const int8_t*>(expected.data()), n, max_report);
    }

    if (actual.dtype() == DataType::Int(16)) {
        return detail::CompareExact<int16_t>(
                static_cast<const int16_t*>(actual.data()),
                static_cast<const int16_t*>(expected.data()), n, max_report);
    }

    if (actual.dtype() == DataType::Int(32)) {
        return detail::CompareExact<int32_t>(
                static_cast<const int32_t*>(actual.data()),
                static_cast<const int32_t*>(expected.data()), n, max_report);
    }

    if (actual.dtype() == DataType::Int(64)) {
        return detail::CompareExact<int64_t>(
                static_cast<const int64_t*>(actual.data()),
                static_cast<const int64_t*>(expected.data()), n, max_report);
    }

    if (actual.dtype() == DataType::UInt(8)) {
        return detail::CompareExact<uint8_t>(
                static_cast<const uint8_t*>(actual.data()),
                static_cast<const uint8_t*>(expected.data()), n, max_report);
    }

    if (actual.dtype() == DataType::UInt(16)) {
        return detail::CompareExact<uint16_t>(
                static_cast<const uint16_t*>(actual.data()),
                static_cast<const uint16_t*>(expected.data()), n, max_report);
    }

    if (actual.dtype() == DataType::UInt(32)) {
        return detail::CompareExact<uint32_t>(
                static_cast<const uint32_t*>(actual.data()),
                static_cast<const uint32_t*>(expected.data()), n, max_report);
    }

    if (actual.dtype() == DataType::UInt(64)) {
        return detail::CompareExact<uint64_t>(
                static_cast<const uint64_t*>(actual.data()),
                static_cast<const uint64_t*>(expected.data()), n, max_report);
    }

    if (actual.dtype() == DataType::Bool()) {
        return detail::CompareExactBool(
                static_cast<const bool*>(actual.data()),
                static_cast<const bool*>(expected.data()), n, max_report);
    }

    return ::testing::AssertionFailure()
           << "ExpectTensorEqual unsupported dtype: " << actual.dtype();
}

}// namespace aethermind::test_utils

#endif