//
// Created by 赵丹 on 25-6-17.
//
#include "dispatch_key_set.h"
#include "error.h"
#include "function_traits.h"
#include "tensor.h"

// #include <fmt/format.h>
#include <gtest/gtest.h>

#if BUILD_WITH_TORCH
#include <torch/torch.h>
#endif

using namespace aethermind;
using namespace aethermind::details;

namespace {

TEST(Tensor, format) {
    // GTEST_SKIP();
    std::string s1 = std::format("The answer is {}.", 42);
}

TEST(Tensor, envs) {
    EXPECT_TRUE(has_env("THP_MEM_ALLOC_ENABLE"));
    EXPECT_TRUE(check_env("THP_MEM_ALLOC_ENABLE"));

    EXPECT_TRUE(has_env("TRACEBACK_LIMIT"));
    auto trackback_limit = std::stoi(get_env("TRACEBACK_LIMIT").value());
    EXPECT_EQ(trackback_limit, 512);
    // std::cout << sysconf(_SC_PAGESIZE);
}

TEST(Tensor, base1) {
    Tensor t({3, 10});
    auto t1 = t;
    EXPECT_TRUE(t.shape() == std::vector<int64_t>({3, 10}));
    EXPECT_EQ(t.use_count(), 2);
    EXPECT_TRUE(t1.defined());
}

TEST(Tensor, init) {
#if BUILD_WITH_TORCH
    torch::Tensor t1;
    EXPECT_FALSE(t1.defined());
    EXPECT_EQ(t1.numel(), 0);
    EXPECT_EQ(t1.dim(), 1);
    EXPECT_TRUE(t1.is_contiguous());
    EXPECT_FALSE(t1.is_cpu());
    EXPECT_FALSE(t1.has_storage());
    std::cout << t1;

    torch::Tensor t11 = torch::ones({3, 10});
    torch::Tensor t12 = torch::ones({3, 10});
    auto t13 = torch::add(t11, t12);

    // UNUSED(t3.size(4));
    // auto t3 = torch::empty({10, 3, 32, 32});
    EXPECT_ANY_THROW(UNUSED(t1.itemsize()););
#endif

    Tensor t2;
    EXPECT_FALSE(t2.defined());
    EXPECT_TRUE(t2.is_contiguous());
    EXPECT_TRUE(t2.dtype() == DataType());
    EXPECT_EQ(t2.ndim(), 1);
    EXPECT_EQ(t2.numel(), 0);
    EXPECT_FALSE(t2.has_storage());
    EXPECT_EQ(t2.storage_offset(), 0);
    EXPECT_EQ(t2.shape(), std::vector<int64_t>{0});
    EXPECT_EQ(t2.strides(), std::vector<int64_t>{1});
    EXPECT_EQ(t2.use_count(), 0);
    EXPECT_FALSE(t2.unique());
    EXPECT_FALSE(t2.is_cpu());
    EXPECT_FALSE(t2.is_cuda());

    EXPECT_THROW(
            {
                try {
                    UNUSED(t2.data_ptr());
                } catch (const Error& e) {
                    // std::cout << e.what() << std::endl;
                    throw;
                    // AETHERMIND_THROW(RuntimeError) << "runtime error.";
                }
            },
            Error);

    // EXPECT_TRUE(t2.data_ptr() == nullptr);
    // EXPECT_TRUE(t2.const_data_ptr() == nullptr);
    EXPECT_ANY_THROW(UNUSED(t2.itemsize()));
}

TEST(Tensor, random) {
    std::vector<int64_t> shape({10, 3, 32, 32});
    int64_t numel = 1;
    for (int64_t x: shape) {
        numel *= x;
    }

#if BUILD_WITH_TORCH
    auto t1 = torch::rand(shape);
#endif

    auto t2 = Tensor::rand(shape);
    EXPECT_TRUE(t2.defined());
    EXPECT_TRUE(t2.is_cpu());
    EXPECT_FALSE(t2.is_cuda());
    EXPECT_EQ(t2.shape(), shape);
    for (int i = 0; i < t2.ndim(); ++i) {
        EXPECT_EQ(t2.shape(i), shape[i]);
        EXPECT_EQ(t2.shape(i - t2.ndim()), shape[i]);

        EXPECT_EQ(t2.strides(i), t2.strides()[i]);
        EXPECT_EQ(t2.strides(i - t2.ndim()), t2.strides()[i]);
    }
    EXPECT_ANY_THROW(UNUSED(t2.shape(t2.ndim())));

    EXPECT_EQ(t2.ndim(), 4);
    EXPECT_EQ(t2.numel(), numel);
    EXPECT_EQ(t2.nbytes(), numel * 4);
    EXPECT_EQ(t2.use_count(), 1);
    EXPECT_TRUE(t2.is_contiguous());
    EXPECT_TRUE(t2.unique());
    EXPECT_TRUE(t2.dtype() == DataType::Make<float>());
    EXPECT_TRUE(t2.device().type() == DeviceType::kCPU);
    EXPECT_TRUE(t2.is_cpu());
    EXPECT_FLOAT_EQ(t2.const_data_ptr<float>()[0], static_cast<const float*>(t2.const_data_ptr())[0]);
    {
        Tensor() = t2;
    }
    EXPECT_EQ(t2.use_count(), 1);
}

}
