//
// Created by 赵丹 on 25-6-17.
//
#include "error.h"
#include "function_traits.h"
#include "tensor.h"
#include "dispatch_key_set.h"

#include <fmt/format.h>
#include <gtest/gtest.h>

#if BUILD_WITH_TORCH
#include <torch/torch.h>
#endif

using namespace aethermind;

TEST(Tensor, format) {
    // GTEST_SKIP();
    fmt::print("hello world\n");
    std::string s1 = fmt::format("The answer is {}.", 42);
}

TEST(Tensor, envs) {
    EXPECT_TRUE(has_env("THP_MEM_ALLOC_ENABLE"));
    EXPECT_TRUE(check_env("THP_MEM_ALLOC_ENABLE"));

    EXPECT_TRUE(has_env("TRACEBACK_LIMIT"));
    auto trackback_limit = std::stoi(get_env("TRACEBACK_LIMIT").value());
    EXPECT_EQ(trackback_limit, 512);
    // std::cout << sysconf(_SC_PAGESIZE);
}

TEST(Tensor, unique_void_ptr) {
    struct Context {
        explicit Context(void* ptr) : data(ptr) {}
        void delete_ptr() const {
            // std::cout << "call free.\n";
            free(data);
        }
        void* data;
    };

    auto default_deleter = [](void* ptr) {
        static_cast<Context*>(ptr)->delete_ptr();
    };

    auto* ptr = malloc(10);
    Context ctx(ptr);
    UniqueVoidPtr p(ptr, static_cast<void*>(&ctx), default_deleter);
}

TEST(Tensor, scalar) {
    Scalar s1 = false;
    EXPECT_EQ(s1.toBool(), false);
    EXPECT_TRUE(s1.type() == DLDataTypeCode::kBool);
    EXPECT_TRUE(s1.isBool());
    s1 = true;
    std::cout << s1 << std::endl;
    std::cout << toString(s1) << std::endl;

    Scalar s2 = 10;
    EXPECT_EQ(s2.toInt(), 10);
    EXPECT_TRUE(s2.isIntegral());
    std::cout << s2 << std::endl;
    std::cout << toString(s2) << std::endl;

    Scalar s3 = 1.5;
    EXPECT_EQ(s3.toFloat(), 1.5);
    EXPECT_TRUE(s3.isFloatingPoint());
    EXPECT_TRUE(std::isfinite(s3.toFloat()));
    std::cout << s3 << std::endl;
    std::cout << toString(s3) << std::endl;
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

TEST(Tensor, function_traits) {
    static_assert(!is_tuple_v<int>);
    static_assert(is_tuple_v<std::tuple<>>);
    static_assert(is_tuple_v<std::tuple<int, float>>);

    auto f = [](int a, float b) {
        return a + b;
    };

    // using func_type = decltype(f);
    using func_type = float(int, float);
    // using func_type = std::function<float(int, float)>;
    // using func_type = std::function<float(*)(int, float)>;
    // using func_type = float(*)(int, float);

    using func_traits = infer_function_traits_t<func_type>;

    // static_assert(is_function_type_v<func_type>);
    static_assert(std::is_same_v<func_traits::return_type, float>);
    static_assert(std::is_same_v<func_traits::args_type_tuple, std::tuple<int, float>>);
    static_assert(std::is_same_v<func_traits::func_type, float(int, float)>);
    static_assert(func_traits::params_num == 2);

    static_assert(std::is_same_v<std::make_index_sequence<5>, std::index_sequence<0, 1, 2, 3, 4>>);
}
