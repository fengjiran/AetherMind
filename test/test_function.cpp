//
// Created by richard on 9/25/25.
//
#include "container/array.h"
#include "container/string.h"
#include "function.h"
#include "registry.h"

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

    // Function f = Function::FromPacked(packed_func);
    Function f = packed_func;
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

    // Function f = Function::FromPacked(raw_func);
    Function f = raw_func;
    Any result = f(4, 7);
    EXPECT_EQ(result.cast<int>(), 28);
    EXPECT_EQ(f.schema(), "(0: const Any*, 1: Int, 2: Any*) -> void");
}

// 测试Function的operator()调用，带多个参数和不同类型
TEST(FunctionTest, OperatorCallWithMultipleArgs) {
    auto func = [](PackedArgs args, Any* res) {
        int a = args[0].cast<int>();
        auto b = args[1].cast<double>();
        auto c = args[2].cast<std::string>();
        *res = a + static_cast<int>(b) + c.size();
    };

    // auto f = Function::FromPacked(func);
    Function f = func;
    Any result = f(5, 3.14, std::string("hello"));
    EXPECT_EQ(result.cast<int>(), 5 + 3 + 5);// 5 + 3(整数部分) + 5(字符串长度)
}

// 测试Function的operator()调用，无参数
TEST(FunctionTest, OperatorCallWithNoArgs) {
    auto func = [](PackedArgs args, Any* res) {
        EXPECT_EQ(args.size(), 0);
        *res = 42;
    };

    // Function f = Function::FromPacked(func);
    Function f = func;
    Any result = f();
    EXPECT_EQ(result.cast<int>(), 42);
}

// 测试Function返回不同类型的结果
TEST(FunctionTest, ReturnDifferentTypes) {
    // 返回整数
    // Function f1 = Function::FromPacked([](PackedArgs, Any* res) { *res = 100; });
    Function f1 = [](PackedArgs, Any* res) { *res = 100; };
    EXPECT_EQ(f1().cast<int>(), 100);

    // 返回浮点数
    // Function f2 = Function::FromPacked([](PackedArgs, Any* res) { *res = 3.14159; });
    Function f2 = [](PackedArgs, Any* res) { *res = 3.14159; };
    EXPECT_FLOAT_EQ(f2().cast<float>(), 3.14159);

    // 返回字符串
    // Function f3 = Function::FromPacked([](PackedArgs, Any* res) { *res = std::string("test"); });
    Function f3 = [](PackedArgs, Any* res) { *res = std::string("test"); };
    EXPECT_EQ(f3().cast<std::string>(), "test");

    // 返回布尔值
    // Function f4 = Function::FromPacked([](PackedArgs, Any* res) { *res = true; });
    Function f4 = [](PackedArgs, Any* res) { *res = true; };
    EXPECT_TRUE(f4().cast<bool>());
}

// 测试get_impl_ptr_unsafe和release_impl_unsafe方法
TEST(FunctionTest, ImplPointerMethods) {
    auto func = [](PackedArgs, Any* res) { *res = 0; };
    Function f = func;

    // 测试get_impl_ptr_unsafe
    FunctionImpl* impl_ptr = f.get_impl_ptr_unsafe();
    EXPECT_NE(impl_ptr, nullptr);

    // 测试release_impl_unsafe
    FunctionImpl* released_impl = f.release_impl_unsafe();
    EXPECT_EQ(released_impl, impl_ptr);
    EXPECT_EQ(f.get_impl_ptr_unsafe(), NullTypeOf<FunctionImpl>::singleton());

    // 释放资源
    ObjectUnsafe::DecRefObjectHandle(released_impl);
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
    Function f = func;

    // 测试TypeTraits的基本特性
    // EXPECT_TRUE(TypeTraits<Function>::is_specialized);
    // EXPECT_FALSE(TypeTraits<Function>::is_trivially_copyable);

    // 测试TypeStr方法
    EXPECT_EQ(Type2Str<Function>::value(), "Function");
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

    Function create_func = Function(create_adder);
    Any adder_any = create_func(100);

    // 从Any中提取Function并调用
    auto adder = adder_any.cast<Function>();
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

    Function f = sum_func;

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

    MAYBE_UNUSED auto f = [](int a, float b) {
        return a + b;
    };

    struct test_functor {
        float operator()(int a, float b) {
            return a + b;
        }
    };

    // using func_info = FunctionInfo<test_functor>;
    using func_info = FunctionInfo<decltype(f)>;

    static_assert(std::is_same_v<func_info::return_type, float>);
    static_assert(std::is_same_v<func_info::args_type_tuple, std::tuple<int, float>>);
    static_assert(std::is_same_v<func_info::func_type, float(int, float)>);
    static_assert(func_info::num_args == 2);
    static_assert(std::is_same_v<std::make_index_sequence<5>, std::index_sequence<0, 1, 2, 3, 4>>);
    std::cout << func_info::Schema() << std::endl;
}

// 测试辅助函数
int add(int a, int b) {
    return a + b;
}

std::string concat(const std::string& a, const std::string& b) {
    return a + b;
}

void void_function(int& counter) {
    counter++;
}

class TestClass {
public:
    static int multiply(int a, int b) {
        return a * b;
    }

    static std::string greet(const std::string& name) {
        return "Hello, " + name + "!";
    }
};

TEST(FunctionTest, class_method) {
    Function f = &TestClass::greet;
    EXPECT_EQ(f("World").cast<String>(), "Hello, World!");

    TypedFunction<std::string(std::string)> ft = &TestClass::greet;
    EXPECT_EQ(ft("World"), "Hello, World!");
    EXPECT_EQ(f.schema(), ft.schema());
}

TEST(TypeFunction, default_construction) {
    TypedFunction<int(int, int)> func;
    EXPECT_FALSE(func.packed().defined());
    EXPECT_THROW(func(0, 0), std::bad_function_call);
    Function f = func;
    EXPECT_FALSE(f.defined());
    EXPECT_FALSE(func.operator Function().defined());
}

TEST(TypeFunction, nullopt_construction) {
    TypedFunction<int(int, int)> func(std::nullopt);
    EXPECT_FALSE(func.packed().defined());
}

TEST(TypeFunction, lambda_construction) {
    // 从lambda构造
    auto f = [](int a, int b) { return a + b; };
    TypedFunction<int(int, int)> func(f);
    EXPECT_TRUE(func.packed().defined());

    // 测试调用
    EXPECT_EQ(func(2, 3), 5);
}

TEST(TypeFunction, lambda_with_name) {
    auto lambda = [](int a, int b) { return a * b; };
    TypedFunction<int(int, int)> func(lambda, "multiply_function");
    EXPECT_TRUE(func.packed().defined());
    EXPECT_EQ(func(3, 4), 12);
}

TEST(TypeFunction, std_function_construction) {
    std::function<int(int, int)> std_func = [](int a, int b) { return a - b; };
    TypedFunction<int(int, int)> func(std_func);
    EXPECT_TRUE(func.packed().defined());
    EXPECT_EQ(func(10, 3), 7);
}

TEST(TypeFunction, function_packed_construction) {
    auto lambda = [](int a, int b) { return a + b + 1; };
    Function packed_func = Function::FromTyped(lambda);
    TypedFunction<int(int, int)> func(packed_func);
    EXPECT_TRUE(func.packed().defined());
    EXPECT_EQ(func(2, 3), 6);
}

TEST(TypeFunction, assignment_operators) {
    TypedFunction<int(int, int)> func;

    // lambda赋值
    auto lambda = [](int a, int b) { return a * b; };
    func = lambda;
    EXPECT_EQ(func(3, 4), 12);

    // Function赋值
    auto lambda2 = [](int a, int b) { return a - b; };
    Function packed_func = Function::FromTyped(lambda2);
    func = packed_func;
    EXPECT_EQ(func(10, 3), 7);
}

TEST(TypeFunction, void_return_type) {
    int counter = 0;
    auto lambda = [&counter](int a, int b) {
        counter = a + b;
    };

    TypedFunction<void(int, int)> func(lambda);
    func(2, 3);
    EXPECT_EQ(counter, 5);
}

TEST(TypeFunction, string_arguments_and_return) {
    auto lambda = [](const std::string& a, const std::string& b) {
        return a + " " + b;
    };

    TypedFunction<std::string(const std::string&, const std::string&)> func(lambda);
    std::string result = func("Hello", "World");
    EXPECT_EQ(result, "Hello World");
}

TEST(TypeFunction, conversion_to_function) {
    auto lambda = [](int a, int b) { return a + b; };
    TypedFunction<int(int, int)> typed_func(lambda);

    // 转换为Function
    Function packed_func = typed_func;
    EXPECT_TRUE(packed_func.defined());

    // 通过Function调用
    Any result = packed_func(2, 3);
    EXPECT_EQ(result.cast<int>(), 5);
}

TEST(TypeFunction, member_function_wrapper) {
    // 包装成员函数
    auto lambda = [](int a, int b) { return TestClass::multiply(a, b); };
    TypedFunction<int(int, int)> func(lambda);

    EXPECT_EQ(func(3, 4), 12);
}

TEST(TypeFunction, any_return_type) {
    auto lambda = [](int a, int b) -> Any {
        return {a + b};
    };

    TypedFunction<Any(int, int)> func(lambda);
    Any result = func(2, 3);
    EXPECT_EQ(result.cast<int>(), 5);
}

// TEST(TypeFunction, reference_arguments) {
//     int value = 0;
//     auto lambda = [](int&& ref, int new_value) {
//         ref = new_value;
//     };
//
//     TypedFunction<void(int&&, int)> func(lambda);
//     func(std::move(value), 42);
//     EXPECT_EQ(value, 42);
// }

TEST(TypeFunction, move_semantics) {
    auto lambda = [](std::string&& str) {
        return std::move(str) + " processed";
    };

    TypedFunction<std::string(std::string&&)> func(lambda);
    std::string input = "test";
    std::string result = func(std::move(input));
    EXPECT_EQ(result, "test processed");
    EXPECT_TRUE(input.empty());// 输入被移动
}

TEST(TypeFunction, const_arguments) {
    auto lambda = [](const int& a, const int& b) {
        return a + b;
    };

    TypedFunction<int(const int&, const int&)> func(lambda);
    const int x = 5, y = 3;
    EXPECT_EQ(func(x, y), 8);
}

TEST(TypeFunction, variadic_arguments) {
    auto sum_lambda = [](int a, int b, int c) {
        return a + b + c;
    };

    TypedFunction<int(int, int, int)> func(sum_lambda);
    EXPECT_EQ(func(1, 2, 3), 6);
}

TEST(TypeFunction, exception_handling) {
    auto throwing_lambda = [](int a, int b) {
        if (b == 0) {
            throw std::runtime_error("Division by zero");
        }
        return a / b;
    };

    TypedFunction<int(int, int)> func(throwing_lambda);

    // 正常情况
    EXPECT_EQ(func(6, 2), 3);

    // 异常情况 - 测试是否会抛出异常
    EXPECT_THROW(func(5, 0), std::runtime_error);
}

TEST(TypeFunction, packed_methods) {
    auto lambda = [](int a, int b) { return a * b; };
    TypedFunction<int(int, int)> func(lambda);

    // 测试packed()方法
    const Function& packed = func.packed();
    EXPECT_TRUE(packed.defined());

    // 测试移动语义的packed()
    TypedFunction<int(int, int)> func2(lambda);
    Function moved_packed = std::move(func2).packed();
    EXPECT_TRUE(moved_packed.defined());
    EXPECT_FALSE(func2.packed().defined());// 原对象应该被移动
}

TEST(function_type_traits, function_copy_to_any) {
    // 测试Function的CopyToAny
    auto lambda = [](int a, int b) { return a + b; };
    Function func = Function::FromTyped(lambda);

    Any any_func = func;
    auto res = any_func.cast<Function>()(1, 2);
    EXPECT_EQ(any_func.use_count(), 2);
    EXPECT_EQ(res.cast<int>(), 3);
}

TEST(function_type_traits, function_move_to_any) {
    // 测试Function的MoveToAny
    auto lambda = [](int a, int b) { return a * b; };
    Function func = Function::FromTyped(lambda);

    Any any_data = std::move(func);
    EXPECT_EQ(func.use_count(), 0);
    EXPECT_EQ(any_data.use_count(), 1);
}

TEST(function_type_traits, function_copy_from_any) {
    // 测试Function的CopyFromAnyAfterCheck
    auto lambda = [](int a, int b) { return a - b; };
    Function original_func = Function::FromTyped(lambda);
    Any any_data = original_func;

    Function copied_func = any_data.cast<Function>();

    // 测试复制的函数是否正常工作
    Any result = copied_func(10, 3);
    EXPECT_EQ(result.cast<int>(), 7);

    // 验证引用计数增加
    EXPECT_EQ(copied_func.use_count(), original_func.use_count());
}

TEST(function_type_traits, function_move_from_any) {
    // 测试Function的MoveFromAnyAfterCheck
    auto lambda = [](int a, int b) { return a + b + 1; };
    Function original_func = Function::FromTyped(lambda);
    Any any_data = original_func;

    Function moved_func = any_data.MoveFromAny<Function>();
    EXPECT_EQ(moved_func.use_count(), 2);

    // 测试移动的函数是否正常工作
    Any result = moved_func(2, 3);
    EXPECT_EQ(result.cast<int>(), 6);
}

TEST(function_type_traits, function_try_cast_from_any) {
    // 测试Function的TryCastFromAny
    auto lambda = [](int a, int b) { return a * b; };
    Function func = Function::FromTyped(lambda);

    Any any_data = func;

    auto result = any_data.try_cast<Function>();
    EXPECT_TRUE(result.has_value());

    // 测试转换后的函数
    Any call_result = result.value()(3, 4);
    EXPECT_EQ(call_result.cast<int>(), 12);
}

TEST(function_type_traits, function_try_cast_from_wrong_type) {
    // 测试从错误类型的Any转换Function
    Any any_data = 42;

    auto result = any_data.try_cast<Function>();
    EXPECT_FALSE(result.has_value());
}

TEST(function_type_traits, function_type_str) {
    // 测试Function的TypeStr方法
    auto type_str = Type2Str<Function>::value();
    EXPECT_EQ(type_str, "Function");
}

TEST(typed_function_type_traits, typed_function_copy_to_any) {
    // 测试TypedFunction的CopyToAny
    auto lambda = [](int a, int b) { return a + b; };
    TypedFunction<int(int, int)> typed_func(lambda);

    Any any_data = typed_func;
}

TEST(typed_function_type_traits, typed_function_move_to_any) {
    // 测试TypedFunction的MoveToAny
    auto lambda = [](int a, int b) { return a * b; };
    TypedFunction<int(int, int)> typed_func(lambda);

    Any any_data = std::move(typed_func);
}

TEST(typed_function_type_traits, typed_function_copy_from_any) {
    // 测试TypedFunction的CopyFromAnyAfterCheck
    auto lambda = [](int a, int b) { return a - b; };
    TypedFunction<int(int, int)> original_func(lambda);

    Any any_data = original_func;

    TypedFunction<int(int, int)> copied_func = any_data.cast<TypedFunction<int(int, int)>>();

    // 测试复制的函数是否正常工作
    int result = copied_func(10, 3);
    EXPECT_EQ(result, 7);
}

TEST(typed_function_type_traits, typed_function_move_from_any) {
    // 测试TypedFunction的MoveFromAnyAfterCheck
    auto lambda = [](int a, int b) { return a + b + 1; };
    TypedFunction<int(int, int)> original_func(lambda);

    Any any_data = original_func;

    TypedFunction<int(int, int)> moved_func = any_data.MoveFromAny<TypedFunction<int(int, int)>>();
    // 测试移动的函数是否正常工作
    int result = moved_func(2, 3);
    EXPECT_EQ(result, 6);
}

TEST(typed_function_type_traits, typed_function_try_cast_from_any) {
    // 测试TypedFunction的TryCastFromAny
    auto lambda = [](int a, int b) { return a * b; };
    TypedFunction<int(int, int)> typed_func(lambda);

    Any any_data = typed_func;
    auto result = any_data.try_cast<TypedFunction<int(int, int)>>();
    EXPECT_TRUE(result.has_value());

    // 测试转换后的函数
    int call_result = result.value()(3, 4);
    EXPECT_EQ(call_result, 12);
}

TEST(typed_function_type_traits, typed_function_try_cast_from_wrong_type) {
    // 测试从错误类型的Any转换TypedFunction
    Any any_data = "hello world";

    auto result = any_data.try_cast<TypedFunction<int(int, int)>>();
    EXPECT_FALSE(result.has_value());
}

TEST(typed_function_type_traits, typed_function_type_str) {
    // 测试TypedFunction的TypeStr方法
    auto type_str = Type2Str<TypedFunction<int(int, int)>>::value();
    EXPECT_EQ(type_str, "Function");
}

TEST(typed_function_type_traits, void_return_type_handling) {
    // 测试void返回类型的TypedFunction
    int counter = 0;
    auto lambda = [&counter](int a, int b) {
        counter = a + b;
    };

    TypedFunction<void(int, int)> typed_func(lambda);

    Any any_data = typed_func;

    auto result = any_data.try_cast<TypedFunction<void(int, int)>>();
    EXPECT_TRUE(result.has_value());

    // 测试void函数调用
    result.value()(2, 3);
    EXPECT_EQ(counter, 5);
}

TEST(typed_function_type_traits, string_arguments_and_return) {
    // 测试字符串参数的TypedFunction
    auto lambda = [](const std::string& a, const std::string& b) {
        return a + " " + b;
    };

    TypedFunction<std::string(const std::string&, const std::string&)> typed_func(lambda);

    Any any_data = typed_func;

    auto result = any_data.try_cast<decltype(typed_func)>();
    EXPECT_TRUE(result.has_value());

    std::string call_result = result.value()("Hello", "World");
    EXPECT_EQ(call_result, "Hello World");
}

TEST(function_type_traits, integration_with_any_class) {
    // 测试与Any类的集成
    auto lambda = [](int a, int b) { return a + b; };
    Function func = Function::FromTyped(lambda);

    // 使用Any构造函数
    Any any_func = func;
    EXPECT_TRUE(any_func.IsObjectRef());

    // 使用as方法转换回Function
    auto result = any_func.as<Function>();
    EXPECT_TRUE(result.has_value());

    // 测试转换后的函数
    Any call_result = result.value()(2, 3);
    EXPECT_EQ(call_result.cast<int>(), 5);
}

TEST(typed_function_type_traits, integration_with_any_class) {
    // 测试TypedFunction与Any类的集成
    auto lambda = [](int a, int b) { return a * b; };
    TypedFunction<int(int, int)> typed_func(lambda);

    // 使用Any构造函数
    Any any_func = typed_func;
    EXPECT_TRUE(any_func.IsObjectRef());

    // 使用as方法转换回TypedFunction
    auto result = any_func.as<TypedFunction<int(int, int)>>();
    EXPECT_TRUE(result.has_value());

    // 测试转换后的函数
    int call_result = result.value()(3, 4);
    EXPECT_EQ(call_result, 12);
}

TEST(GlobalFunctionTable, init) {
    auto add_one = [](const int& i) {
        return i + 1;
    };

    Registry().def("test.add_one", add_one, __FILE__, __LINE__);
    std::cout << Registry::GetRegisteredLocation("test.add_one") << std::endl;
    std::cout << Registry::GetRegisteredLocation("ListGlobalFunctionNamesFunctor") << std::endl;
    auto fadd1 = Function::GetGlobalFunctionRequired("test.add_one");
    std::cout << fadd1.schema() << std::endl;
    int x = fadd1(1).cast<int>();
    EXPECT_EQ(x, 2);

    auto fnot_exist = Function::GetGlobalFunction("test.not_existing_func");
    EXPECT_TRUE(!fnot_exist);

    auto fname_functor = Function::GetGlobalFunctionRequired("ListGlobalFunctionNamesFunctor")().cast<Function>();
    std::cout << fname_functor.schema() << std::endl;
    int len = fname_functor(-1).cast<int>();
    Array<String> names(len);
    for (int i = 0; i < len; ++i) {
        names.Set(i, fname_functor(i).cast<String>());
    }
    EXPECT_TRUE(std::find(names.begin(), names.end(), "test.add_one") != names.end());
}

}// namespace