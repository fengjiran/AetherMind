//
// Created by richard on 9/3/25.
//
#include <gtest/gtest.h>

#include "error.h"

using namespace aethermind;

namespace {

TEST(AETHERMIND_THROW, BasicExceptionThrowing) {
    EXPECT_THROW({ AETHERMIND_THROW(index_error) << "Test index out of bounds"; }, Error);
}

TEST(AETHERMIND_THROW, ErrorKindAndMessage) {
    try {
        AETHERMIND_THROW(value_error) << "Invalid parameter value";
        FAIL() << "AETHERMIND_THROW should have thrown an exception";
    } catch (const Error& e) {
        const std::string what_str = e.what();
        EXPECT_NE(what_str.find("value_error"), std::string::npos);
        EXPECT_NE(what_str.find("Invalid parameter value"), std::string::npos);
        EXPECT_NE(what_str.find("Traceback (most recent call last):"), std::string::npos);
    }
}

TEST(AETHERMIND_THROW, TracebackInclusion) {
    try {
        AETHERMIND_THROW(runtime_error) << "Test traceback generation";
        FAIL() << "AETHERMIND_THROW should have thrown an exception";
    } catch (const Error& e) {
        const std::string what_str = e.what();
        // 验证回溯信息前缀
        EXPECT_NE(what_str.find("Traceback (most recent call last):"), std::string::npos);
        // 验证回溯中包含当前文件名信息
        EXPECT_NE(what_str.find("test_error.cpp"), std::string::npos);
        // 验证回溯中包含函数信息
        EXPECT_NE(what_str.find("TestBody"), std::string::npos);
    }
}

TEST(AETHERMIND_THROW, DifferentErrorKinds) {
    EXPECT_THROW({ AETHERMIND_THROW(type_error) << "Type mismatch"; }, Error);

    EXPECT_THROW({ AETHERMIND_THROW(range_error) << "Value out of range"; }, Error);

    EXPECT_THROW({ AETHERMIND_THROW(io_error) << "File not found"; }, Error);
}

TEST(AETHERMIND_THROW, MultiLineErrorMessage) {
    try {
        AETHERMIND_THROW(assertion_error)
                << "Assertion failed: " << 42
                << " is not equal to " << 24;
        FAIL() << "AETHERMIND_THROW should have thrown an exception";
    } catch (const Error& e) {
        const std::string what_str = e.what();
        EXPECT_NE(what_str.find("Assertion failed: 42 is not equal to 24"), std::string::npos);
    }
}

TEST(AETHERMIND_THROW, EmptyErrorMessage) {
    try {
        AETHERMIND_THROW(unknown_error);
        FAIL() << "AETHERMIND_THROW should have thrown an exception";
    } catch (const Error& e) {
        const std::string what_str = e.what();
        EXPECT_NE(what_str.find("unknown_error"), std::string::npos);
    }
}

}// namespace