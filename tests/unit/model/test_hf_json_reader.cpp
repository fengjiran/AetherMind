#include "aethermind/model/formats/hf/hf_json_reader.h"

#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace aethermind;

TEST(ModelLoader_HfJsonReaderTest, ParsesStringWithEscapeSequences) {
    hf::HfJsonReader reader(R"("a\nb\tc\"d\\e\/f\b\f\r")");

    const auto value = reader.ParseString();

    ASSERT_TRUE(value.ok()) << value.status().ToString();
    EXPECT_EQ(*value, "a\nb\tc\"d\\e/f\b\f\r");
}

TEST(ModelLoader_HfJsonReaderTest, ParsesInt64PositiveAndNegative) {
    hf::HfJsonReader positive("42");
    const auto pos = positive.ParseInt64();
    ASSERT_TRUE(pos.ok()) << pos.status().ToString();
    EXPECT_EQ(*pos, 42);

    hf::HfJsonReader negative("-7");
    const auto neg = negative.ParseInt64();
    ASSERT_TRUE(neg.ok()) << neg.status().ToString();
    EXPECT_EQ(*neg, -7);
}

TEST(ModelLoader_HfJsonReaderTest, ParsesUInt64) {
    hf::HfJsonReader reader("18446744073709551615");

    const auto value = reader.ParseUInt64();

    ASSERT_TRUE(value.ok()) << value.status().ToString();
    EXPECT_EQ(*value, UINT64_MAX);
}

TEST(ModelLoader_HfJsonReaderTest, ParsesDoubleWithExponent) {
    hf::HfJsonReader reader("1.5e3");

    const auto value = reader.ParseDouble();

    ASSERT_TRUE(value.ok()) << value.status().ToString();
    EXPECT_DOUBLE_EQ(*value, 1500.0);
}

TEST(ModelLoader_HfJsonReaderTest, ParsesBoolTrueAndFalse) {
    hf::HfJsonReader reader_true("true");
    const auto value_true = reader_true.ParseBool();
    ASSERT_TRUE(value_true.ok()) << value_true.status().ToString();
    EXPECT_TRUE(*value_true);

    hf::HfJsonReader reader_false("false");
    const auto value_false = reader_false.ParseBool();
    ASSERT_TRUE(value_false.ok()) << value_false.status().ToString();
    EXPECT_FALSE(*value_false);
}

TEST(ModelLoader_HfJsonReaderTest, ParsesStringArray) {
    hf::HfJsonReader reader(R"(["a", "b"])");

    const auto value = reader.ParseStringArray();

    ASSERT_TRUE(value.ok()) << value.status().ToString();
    EXPECT_EQ(*value, (std::vector<std::string>{"a", "b"}));
}

TEST(ModelLoader_HfJsonReaderTest, ParsesEmptyStringArray) {
    hf::HfJsonReader reader("[]");

    const auto value = reader.ParseStringArray();

    ASSERT_TRUE(value.ok()) << value.status().ToString();
    EXPECT_TRUE(value->empty());
}

TEST(ModelLoader_HfJsonReaderTest, ParsesInt64Array) {
    hf::HfJsonReader reader("[1, -2, 3]");

    const auto value = reader.ParseInt64Array();

    ASSERT_TRUE(value.ok()) << value.status().ToString();
    EXPECT_EQ(*value, (std::vector<int64_t>{1, -2, 3}));
}

TEST(ModelLoader_HfJsonReaderTest, TryConsumeSkipsWhitespace) {
    hf::HfJsonReader reader("  \t\n }");

    EXPECT_TRUE(reader.TryConsume('}'));
    EXPECT_TRUE(reader.AtEnd());
}

TEST(ModelLoader_HfJsonReaderTest, TryConsumeLiteralMatchesAndAdvances) {
    hf::HfJsonReader reader("nullish");

    EXPECT_TRUE(reader.TryConsumeLiteral("null"));
    EXPECT_FALSE(reader.TryConsumeLiteral("null"));
}

TEST(ModelLoader_HfJsonReaderTest, ExpectFailsWithContext) {
    hf::HfJsonReader reader("x");

    const auto status = reader.Expect('{', "at start of value");

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("Expected '{' at start of value"),
              std::string::npos);
}

TEST(ModelLoader_HfJsonReaderTest, RejectsNonStringValue) {
    hf::HfJsonReader reader("123");

    const auto value = reader.ParseString();

    ASSERT_FALSE(value.ok());
    EXPECT_EQ(value.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(value.status().message().find("Expected JSON string"),
              std::string::npos);
}

TEST(ModelLoader_HfJsonReaderTest, RejectsUnterminatedString) {
    hf::HfJsonReader reader("\"abc");

    const auto value = reader.ParseString();

    ASSERT_FALSE(value.ok());
    EXPECT_NE(value.status().message().find("Unterminated JSON string"),
              std::string::npos);
}

TEST(ModelLoader_HfJsonReaderTest, RejectsEscapeSequenceAtEnd) {
    hf::HfJsonReader reader("\"abc\\");

    const auto value = reader.ParseString();

    ASSERT_FALSE(value.ok());
    EXPECT_NE(value.status().message().find("Unexpected end of JSON escape"),
              std::string::npos);
}

TEST(ModelLoader_HfJsonReaderTest, RejectsUnsupportedEscapeSequence) {
    hf::HfJsonReader reader(R"("abc\x41")");

    const auto value = reader.ParseString();

    ASSERT_FALSE(value.ok());
    EXPECT_NE(value.status().message().find("Unsupported JSON escape sequence"),
              std::string::npos);
}

TEST(ModelLoader_HfJsonReaderTest, RejectsEmptyInt64) {
    hf::HfJsonReader reader("abc");

    const auto value = reader.ParseInt64();

    ASSERT_FALSE(value.ok());
    EXPECT_NE(value.status().message().find("Expected integer value"),
              std::string::npos);
}

TEST(ModelLoader_HfJsonReaderTest, RejectsLoneMinusInt64) {
    hf::HfJsonReader reader("-");

    const auto value = reader.ParseInt64();

    ASSERT_FALSE(value.ok());
    EXPECT_NE(value.status().message().find("Expected integer value"),
              std::string::npos);
}

TEST(ModelLoader_HfJsonReaderTest, RejectsInt64Overflow) {
    hf::HfJsonReader reader("99999999999999999999");

    const auto value = reader.ParseInt64();

    ASSERT_FALSE(value.ok());
    EXPECT_NE(value.status().message().find("Invalid integer value"),
              std::string::npos);
}

TEST(ModelLoader_HfJsonReaderTest, RejectsNegativeUInt64) {
    hf::HfJsonReader reader("-5");

    const auto value = reader.ParseUInt64();

    ASSERT_FALSE(value.ok());
    EXPECT_NE(value.status().message().find("Expected non-negative integer"),
              std::string::npos);
}

TEST(ModelLoader_HfJsonReaderTest, RejectsEmptyDouble) {
    hf::HfJsonReader reader("abc");

    const auto value = reader.ParseDouble();

    ASSERT_FALSE(value.ok());
    EXPECT_NE(value.status().message().find("Expected floating point value"),
              std::string::npos);
}

TEST(ModelLoader_HfJsonReaderTest, RejectsMalformedExponentDouble) {
    hf::HfJsonReader reader("1.5e");

    const auto value = reader.ParseDouble();

    ASSERT_FALSE(value.ok());
    EXPECT_NE(value.status().message().find("Invalid floating point value"),
              std::string::npos);
}

TEST(ModelLoader_HfJsonReaderTest, RejectsNonBoolValue) {
    hf::HfJsonReader reader("nope");

    const auto value = reader.ParseBool();

    ASSERT_FALSE(value.ok());
    EXPECT_NE(value.status().message().find("Expected boolean value"),
              std::string::npos);
}

TEST(ModelLoader_HfJsonReaderTest, RejectsMissingCommaInStringArray) {
    hf::HfJsonReader reader(R"(["a" "b"])");

    const auto value = reader.ParseStringArray();

    ASSERT_FALSE(value.ok());
    EXPECT_NE(value.status().message().find("between string array elements"),
              std::string::npos);
}

TEST(ModelLoader_HfJsonReaderTest, RejectsNonStringArrayElement) {
    hf::HfJsonReader reader("[123]");

    const auto value = reader.ParseStringArray();

    ASSERT_FALSE(value.ok());
    EXPECT_NE(value.status().message().find("Expected JSON string"),
              std::string::npos);
}

TEST(ModelLoader_HfJsonReaderTest, RejectsMissingCommaInInt64Array) {
    hf::HfJsonReader reader("[1 2]");

    const auto value = reader.ParseInt64Array();

    ASSERT_FALSE(value.ok());
    EXPECT_NE(value.status().message().find("between integer array elements"),
              std::string::npos);
}

TEST(ModelLoader_HfJsonReaderTest, SkipValueSkipsNestedStructures) {
    hf::HfJsonReader reader(R"({"a":{"b":[1,2.5,"x"],"c":null},"d":true})");

    const auto status = reader.SkipValue();

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_TRUE(reader.AtEnd());
}

TEST(ModelLoader_HfJsonReaderTest, SkipValueSkipsLiteralNull) {
    hf::HfJsonReader reader("null");

    const auto status = reader.SkipValue();

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_TRUE(reader.AtEnd());
}

TEST(ModelLoader_HfJsonReaderTest, RejectsTruncatedObjectWhileSkipping) {
    hf::HfJsonReader reader("");

    const auto status = reader.SkipValue();

    ASSERT_FALSE(status.ok());
    EXPECT_NE(status.message().find("Unexpected end of JSON while skipping"),
              std::string::npos);
}

TEST(ModelLoader_HfJsonReaderTest, RejectsUnsupportedValueWhileSkipping) {
    hf::HfJsonReader reader("undefined");

    const auto status = reader.SkipValue();

    ASSERT_FALSE(status.ok());
    EXPECT_NE(status.message().find("Unsupported JSON value while skipping"),
              std::string::npos);
}

TEST(ModelLoader_HfJsonReaderTest, RejectsExcessiveNestingWhileSkipping) {
    std::string deeply_nested;
    deeply_nested.append(40, '[').append(40, ']');
    hf::HfJsonReader reader(deeply_nested);

    const auto status = reader.SkipValue();

    ASSERT_FALSE(status.ok());
    EXPECT_NE(status.message().find("JSON nesting depth exceeds maximum"),
              std::string::npos);
}

}// namespace