//
// Created by richard on 9/25/25.
//
#include "container/string.h"
#include "function.h"
#include "testing_object.h"

#include <gtest/gtest.h>
#include <string>

using namespace aethermind;
using namespace aethermind::details;

namespace {

// 测试Function默认构造函数
TEST(FunctionTest, DefaultConstructor) {
    Function f;
    EXPECT_TRUE(IsNullTypePtr(f.get_impl_ptr_unsafe()));
    EXPECT_EQ(f.get_impl_ptr_unsafe(), NullTypeOf<FunctionImpl>::singleton());
}

// 测试Function从FunctionImpl创建
TEST(FunctionTest, FromFunctionImpl) {
    auto impl = make_object<FunctionImpl>();
    Function f(impl);
    EXPECT_NE(f.get_impl_ptr_unsafe(), nullptr);
    EXPECT_EQ(f.get_impl_ptr_unsafe(), impl.get());
}

// 测试Function从简单的可调用对象创建
TEST(FunctionTest, FromCallable) {
    // 使用PackedArgs版本的可调用对象
    auto packed_func = [](PackedArgs args, Any* res) -> void {
        int a = args[0].cast<int>();
        int b = args[1].cast<int>();
        *res = a + b;
    };

    Function f(packed_func);
    Any result = f(5, 3);
    EXPECT_EQ(result.cast<int>(), 8);
}

// 测试Function::FromPacked静态方法
TEST(FunctionTest, FromPackedMethod) {
    // 使用原始指针版本的可调用对象
    auto raw_func = [](const Any* args, int32_t num_args, Any* res) {
        EXPECT_EQ(num_args, 2);
        int a = args[0].cast<int>();
        int b = args[1].cast<int>();
        *res = a * b;
    };

    Function f = Function::FromPacked(raw_func);
    Any result = f(4, 7);
    EXPECT_EQ(result.cast<int>(), 28);
}

// 测试Function的operator()调用，带多个参数和不同类型
TEST(FunctionTest, OperatorCallWithMultipleArgs) {
    auto func = [](PackedArgs args, Any* res) {
        int a = args[0].cast<int>();
        double b = args[1].cast<double>();
        std::string c = args[2].cast<std::string>();
        *res = a + static_cast<int>(b) + c.size();
    };

    Function f(func);
    Any result = f(5, 3.14, std::string("hello"));
    EXPECT_EQ(result.cast<int>(), 5 + 3 + 5);// 5 + 3(整数部分) + 5(字符串长度)
}

// 测试Function的operator()调用，无参数
TEST(FunctionTest, OperatorCallWithNoArgs) {
    auto func = [](PackedArgs args, Any* res) {
        EXPECT_EQ(args.size(), 0);
        *res = 42;
    };

    Function f(func);
    Any result = f();
    EXPECT_EQ(result.cast<int>(), 42);
}

// 测试Function返回不同类型的结果
TEST(FunctionTest, ReturnDifferentTypes) {
    // 返回整数
    Function f1([](PackedArgs, Any* res) { *res = 100; });
    EXPECT_EQ(f1().cast<int>(), 100);

    // 返回浮点数
    Function f2([](PackedArgs, Any* res) { *res = 3.14159; });
    EXPECT_FLOAT_EQ(f2().cast<float>(), 3.14159);

    // 返回字符串
    Function f3([](PackedArgs, Any* res) { *res = std::string("test"); });
    EXPECT_EQ(f3().cast<std::string>(), "test");

    // 返回布尔值
    Function f4([](PackedArgs, Any* res) { *res = true; });
    EXPECT_TRUE(f4().cast<bool>());
}

// 测试get_impl_ptr_unsafe和release_impl_unsafe方法
TEST(FunctionTest, ImplPointerMethods) {
    auto func = [](PackedArgs, Any* res) { *res = 0; };
    Function f(func);

    // 测试get_impl_ptr_unsafe
    FunctionImpl* impl_ptr = f.get_impl_ptr_unsafe();
    EXPECT_NE(impl_ptr, nullptr);

    // 测试release_impl_unsafe
    FunctionImpl* released_impl = f.release_impl_unsafe();
    EXPECT_EQ(released_impl, impl_ptr);
    EXPECT_EQ(f.get_impl_ptr_unsafe(), NullTypeOf<FunctionImpl>::singleton());

    // 释放资源
    ObjectUnsafe::DecRef(released_impl);
}

// 测试FunctionImpl的CallPacked方法
TEST(FunctionImplTest, CallPacked) {
    auto func = [](const FunctionImpl*, const Any* args, int32_t num_args, Any* res) {
        EXPECT_EQ(num_args, 2);
        int a = args[0].cast<int>();
        int b = args[1].cast<int>();
        *res = a - b;
    };

    auto impl = make_object<FunctionImpl>(func);

    Any args[2] = {10, 4};
    Any result;
    impl->CallPacked(args, 2, &result);

    EXPECT_EQ(result.cast<int>(), 6);
}

// 测试FunctionImpl::create静态方法
TEST(FunctionImplTest, CreateMethod) {
    // 测试PackedArgs版本
    auto packed_func = [](PackedArgs args, Any* res) {
        int sum = 0;
        for (size_t i = 0; i < args.size(); ++i) {
            sum += args[i].cast<int>();
        }
        *res = sum;
    };

    auto impl = FunctionImpl::create(packed_func);

    Any args[3] = {1, 2, 3};
    Any result;
    impl->CallPacked(args, 3, &result);

    EXPECT_EQ(result.cast<int>(), 6);

    // 测试原始指针版本
    auto raw_func = [](const Any* args, int32_t num_args, Any* res) {
        int product = 1;
        for (int32_t i = 0; i < num_args; ++i) {
            product *= args[i].cast<int>();
        }
        *res = product;
    };

    auto impl2 = FunctionImpl::create(raw_func);

    Any args2[3] = {2, 3, 4};
    Any result2;
    impl2->CallPacked(args2, 3, &result2);

    EXPECT_EQ(result2.cast<int>(), 24);
}

// 测试PackedArgs类的基本功能
TEST(PackedArgsTest, BasicFunctionality) {
    Any args[4] = {10, 20.5, std::string("test"), true};
    PackedArgs packed(args, 4);

    // 测试size和data方法
    EXPECT_EQ(packed.size(), 4);
    EXPECT_EQ(packed.data(), args);

    // 测试operator[]
    EXPECT_EQ(packed[0].cast<int>(), 10);
    EXPECT_FLOAT_EQ(packed[1].cast<double>(), 20.5);
    EXPECT_EQ(packed[2].cast<std::string>(), "test");
    EXPECT_TRUE(packed[3].cast<bool>());

    // 测试slice方法
    PackedArgs slice = packed.slice(1, 3);
    EXPECT_EQ(slice.size(), 2);
    EXPECT_FLOAT_EQ(slice[0].cast<double>(), 20.5);
    EXPECT_EQ(slice[1].cast<std::string>(), "test");

    // 测试slice方法，只指定begin
    PackedArgs slice2 = packed.slice(2);
    EXPECT_EQ(slice2.size(), 2);
    EXPECT_EQ(slice2[0].cast<std::string>(), "test");
    EXPECT_TRUE(slice2[1].cast<bool>());
}


// 测试PackedArgs::Fill静态方法
TEST(PackedArgsTest, FillMethod) {
    Any data[4];
    PackedArgs::Fill(data, 100, 3.14, std::string("hello"), false);

    EXPECT_EQ(data[0].cast<int>(), 100);
    EXPECT_FLOAT_EQ(data[1].cast<double>(), 3.14);
    EXPECT_EQ(data[2].cast<std::string>(), "hello");
    EXPECT_FALSE(data[3].cast<bool>());
}

// 测试for_each函数
TEST(DetailsTest, ForEachFunction) {
    std::vector<int> results;
    auto func = [&results](size_t i, int value) {
        results.push_back(value * 2);
    };

    for_each(func, 1, 2, 3, 4, 5);

    EXPECT_EQ(results.size(), 5);
    EXPECT_EQ(results[0], 2);
    EXPECT_EQ(results[1], 4);
    EXPECT_EQ(results[2], 6);
    EXPECT_EQ(results[3], 8);
    EXPECT_EQ(results[4], 10);
}

// 测试TypeTraits<Function>的基本功能
TEST(FunctionTypeTraitsTest, BasicFunctionality) {
    // 准备Function对象
    auto func = [](PackedArgs, Any* res) { *res = 42; };
    Function f(func);

    // 测试TypeTraits的基本特性
    // EXPECT_TRUE(TypeTraits<Function>::is_specialized);
    // EXPECT_FALSE(TypeTraits<Function>::is_trivially_copyable);

    // 测试TypeStr方法
    EXPECT_EQ(TypeTraits<Function>::TypeStr(), "Function");
}

// 测试复杂嵌套的Function调用
TEST(FunctionTest, NestedFunctionCalls) {
    // 创建一个返回另一个Function的函数
    auto create_adder = [](PackedArgs args, Any* res) {
        int base = args[0].cast<int>();
        auto adder_func = [base](PackedArgs add_args, Any* add_res) {
            int value = add_args[0].cast<int>();
            *add_res = base + value;
        };
        *res = Function(adder_func);
    };

    Function create_func(create_adder);
    Any adder_any = create_func(100);

    // 从Any中提取Function并调用
    Function adder = adder_any.cast<Function>();
    Any result1 = adder(50);
    Any result2 = adder(-25);

    EXPECT_EQ(result1.cast<int>(), 150);
    EXPECT_EQ(result2.cast<int>(), 75);
}

// 测试边界情况：大量参数
TEST(FunctionTest, LargeNumberOfArgs) {
    const int NUM_ARGS = 15;
    auto sum_func = [](PackedArgs args, Any* res) {
        int sum = 0;
        for (size_t i = 0; i < args.size(); ++i) {
            sum += args[i].cast<int>();
        }
        *res = sum;
    };

    Function f(sum_func);

    // 使用可变参数模板展开来调用带有多个参数的函数
    auto call_with_args = [&f](auto&&... args) -> Any {
        return f(std::forward<decltype(args)>(args)...);
    };

    Any result = call_with_args(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

    // 计算预期的和：1+2+...+15 = 120
    EXPECT_EQ(result.cast<int>(), 120);
}

TEST(FunctionTest, function_traits) {
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
    static_assert(func_traits::num_args == 2);
    static_assert(std::is_same_v<std::make_index_sequence<5>, std::index_sequence<0, 1, 2, 3, 4>>);
    std::cout << func_traits::Schema() << std::endl;
}


}// namespace