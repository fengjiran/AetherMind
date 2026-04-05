#ifndef AETHERMIND_TEST_UTILS_TENSOR_RANDOM_H
#define AETHERMIND_TEST_UTILS_TENSOR_RANDOM_H

#include "tensor_bk.h"

#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace aethermind::test_utils {

inline constexpr uint64_t kDefaultTensorRandomSeed = 123456789ULL;

template<typename T, typename Dist>
void FillTensor(Tensor_BK& tensor, Dist& dist, std::mt19937_64& rng) {
    auto* data = tensor.data_ptr<T>();
    for (int64_t i = 0; i < tensor.numel(); ++i) {
        data[i] = static_cast<T>(dist(rng));
    }
}

inline Tensor_BK RandomUniform(const std::vector<int64_t>& shape,
                            DataType dtype = DataType::Float32(),
                            double low = 0.0,
                            double high = 1.0,
                            uint64_t seed = kDefaultTensorRandomSeed) {
    AM_CHECK(high > low, "RandomUniform requires high > low.");

    Tensor_BK tensor(shape, 0, dtype, Device::CPU());
    std::mt19937_64 rng(seed);

    if (dtype == DataType::Float32()) {
        std::uniform_real_distribution<float> dist(static_cast<float>(low), static_cast<float>(high));
        FillTensor<float>(tensor, dist, rng);
        return tensor;
    }

    if (dtype == DataType::Double()) {
        std::uniform_real_distribution<double> dist(low, high);
        FillTensor<double>(tensor, dist, rng);
        return tensor;
    }

    if (dtype == DataType::Int(32)) {
        AM_CHECK(high <= static_cast<double>(std::numeric_limits<int32_t>::max()) + 1.0,
                 "RandomUniform int32 upper bound overflow.");
        AM_CHECK(low >= static_cast<double>(std::numeric_limits<int32_t>::min()),
                 "RandomUniform int32 lower bound overflow.");
        std::uniform_int_distribution<int32_t> dist(static_cast<int32_t>(low), static_cast<int32_t>(high) - 1);
        FillTensor<int32_t>(tensor, dist, rng);
        return tensor;
    }

    if (dtype == DataType::Int(64)) {
        AM_CHECK(high <= static_cast<double>(std::numeric_limits<int64_t>::max()) + 1.0,
                 "RandomUniform int64 upper bound overflow.");
        AM_CHECK(low >= static_cast<double>(std::numeric_limits<int64_t>::min()),
                 "RandomUniform int64 lower bound overflow.");
        std::uniform_int_distribution<int64_t> dist(static_cast<int64_t>(low), static_cast<int64_t>(high) - 1);
        FillTensor<int64_t>(tensor, dist, rng);
        return tensor;
    }

    AM_THROW(runtime_error) << "RandomUniform supports float32, float64, int32, int64 only.";
    return Tensor_BK();
}

inline Tensor_BK RandomNormal(const std::vector<int64_t>& shape,
                           DataType dtype = DataType::Float32(),
                           double mean = 0.0,
                           double stddev = 1.0,
                           uint64_t seed = kDefaultTensorRandomSeed) {
    AM_CHECK(stddev > 0.0, "RandomNormal requires stddev > 0.");

    Tensor_BK tensor(shape, 0, dtype, Device::CPU());
    std::mt19937_64 rng(seed);

    if (dtype == DataType::Float32()) {
        std::normal_distribution<float> dist(static_cast<float>(mean), static_cast<float>(stddev));
        FillTensor<float>(tensor, dist, rng);
        return tensor;
    }

    if (dtype == DataType::Double()) {
        std::normal_distribution<double> dist(mean, stddev);
        FillTensor<double>(tensor, dist, rng);
        return tensor;
    }

    AM_THROW(runtime_error) << "RandomNormal supports float32 and float64 only.";
    return Tensor_BK();
}

inline Tensor_BK RandomInt(const std::vector<int64_t>& shape,
                        int64_t low,
                        int64_t high,
                        uint64_t seed = kDefaultTensorRandomSeed,
                        DataType dtype = DataType::Int(64)) {
    AM_CHECK(high > low, "RandomInt requires high > low.");
    AM_CHECK(dtype == DataType::Int(32) || dtype == DataType::Int(64),
             "RandomInt supports int32 and int64 only.");

    Tensor_BK tensor(shape, 0, dtype, Device::CPU());
    std::mt19937_64 rng(seed);

    if (dtype == DataType::Int(32)) {
        AM_CHECK(high <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1,
                 "RandomInt int32 upper bound overflow.");
        AM_CHECK(low >= static_cast<int64_t>(std::numeric_limits<int32_t>::min()),
                 "RandomInt int32 lower bound overflow.");
        std::uniform_int_distribution<int32_t> dist(static_cast<int32_t>(low), static_cast<int32_t>(high) - 1);
        FillTensor<int32_t>(tensor, dist, rng);
        return tensor;
    }

    std::uniform_int_distribution<int64_t> dist(low, high - 1);
    FillTensor<int64_t>(tensor, dist, rng);
    return tensor;
}

}// namespace aethermind::test_utils

#endif
