#include "aethermind/model/formats/hf/hf_json_reader.h"
#include "inference/test_malloc_interposer.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace aethermind;

TEST(ModelLoader_HfJsonReaderTest, ParsesStringWithEscapeSequences) {
    hf::HfJsonReader reader(R"("a\nb\tc\"d\\e\/f\b\f\r")");

    const auto value = reader.ParseString();

    ASSERT_TRUE(value.ok()) << value.status().ToString();
    EXPECT_EQ(*value, "a\nb\tc\"d\\e/f\b\f\r");
}

TEST(ModelLoader_HfJsonReaderTest, DecodesUnicodeEscapesToUtf8) {
    hf::HfJsonReader reader(R"("\u0000\u007f\u0080\u07ff\u0800\u4e2d\u6587\uD7ff\ue000\uffff")");
    const auto value = reader.ParseString();
    ASSERT_TRUE(value.ok()) << value.status().ToString();
    EXPECT_EQ(*value, std::string(1, '\0') + "\x7f\xc2\x80\xdf\xbf\xe0\xa0\x80"
                                             "\xe4\xb8\xad\xe6\x96\x87\xed\x9f\xbf\xee\x80\x80\xef\xbf\xbf");
    EXPECT_TRUE(reader.AtEnd());
}

TEST(ModelLoader_HfJsonReaderTest, DecodesUtf16SurrogatePairs) {
    hf::HfJsonReader reader(R"("\ud800\udc00\uD83D\uDE00\udbff\udfff")");
    const auto value = reader.ParseString();
    ASSERT_TRUE(value.ok()) << value.status().ToString();
    EXPECT_EQ(*value, "\xf0\x90\x80\x80\xf0\x9f\x98\x80\xf4\x8f\xbf\xbf");
}

TEST(ModelLoader_HfJsonReaderTest, PreservesValidUtf8) {
    const std::string text = "\xc2\x80\xe4\xb8\xad\xf0\x9f\x98\x80";
    const std::string json = "\"" + text + "\"";
    hf::HfJsonReader reader(json);
    const auto value = reader.ParseString();
    ASSERT_TRUE(value.ok()) << value.status().ToString();
    EXPECT_EQ(*value, text);
}

TEST(ModelLoader_HfJsonReaderTest, RejectsMalformedUnicodeEscapes) {
    const std::string_view inputs[]{
            R"("\u")",
            R"("\u123")",
            R"("\u12xz")",
            R"("\ud800")",
            R"("\ud800x")",
            R"("\ud800\u0041")",
            R"("\ud800\ud800")",
            R"("\udc00")",
            R"("\udfff")",
            R"("\ud800\udc")",
    };
    for (const auto input: inputs) {
        SCOPED_TRACE(input);
        hf::HfJsonReader reader(input);
        EXPECT_EQ(reader.ParseString().status().code(), StatusCode::kInvalidArgument);
        hf::HfJsonReader skipped(input);
        EXPECT_EQ(skipped.SkipValue().code(), StatusCode::kInvalidArgument);
    }
}

TEST(ModelLoader_HfJsonReaderTest, RejectsUnescapedControlCharacters) {
    for (char control = 0; control < 0x20; ++control) {
        SCOPED_TRACE(static_cast<int>(control));
        const std::string json = "\"a" + std::string(1, control) + "b\"";
        hf::HfJsonReader reader(json);
        EXPECT_EQ(reader.ParseString().status().code(), StatusCode::kInvalidArgument);
        hf::HfJsonReader skipped(json);
        EXPECT_EQ(skipped.SkipValue().code(), StatusCode::kInvalidArgument);
    }
}

TEST(ModelLoader_HfJsonReaderTest, RejectsMalformedUtf8) {
    const std::string_view inputs[]{
            "\x80",
            "\xc0\x80",
            "\xc2",
            "\xc2\x20",
            "\xe0\x80\x80",
            "\xed\xa0\x80",
            "\xf0\x80\x80\x80",
            "\xf4\x90\x80\x80",
            "\xf5\x80\x80\x80",
            "\xff",
            "\xe4\xb8",
            "\xf0\x9f\x98",
    };
    for (const auto input: inputs) {
        SCOPED_TRACE(::testing::PrintToString(input));
        const std::string json = "\"" + std::string(input) + "\"";
        hf::HfJsonReader reader(json);
        EXPECT_EQ(reader.ParseString().status().code(), StatusCode::kInvalidArgument);
        hf::HfJsonReader skipped(json);
        EXPECT_EQ(skipped.SkipValue().code(), StatusCode::kInvalidArgument);
    }
}

TEST(ModelLoader_HfJsonReaderTest, SkipsNestedUnicodeStrings) {
    hf::HfJsonReader reader(R"({"\u4e2d":["\u6587","\ud83d\ude00"]})");
    EXPECT_TRUE(reader.SkipValue().ok());
    EXPECT_TRUE(reader.AtEnd());
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

TEST(ModelLoader_HfJsonReaderTest, RejectsNonJsonNumberSyntax) {
    const std::string_view inputs[]{
            "00",
            "01",
            "-00",
            "-01",
            ".5",
            "-.5",
            "1.",
            "1.e2",
            "1e",
            "1e+",
            "1e-",
            "+1",
            "--1",
            "NaN",
            "Infinity",
            "0x10",
            "1foo",
            "1true",
            "1:2",
            "1\f",
            "1\v",
    };
    for (const auto input: inputs) {
        SCOPED_TRACE(::testing::PrintToString(input));
        hf::HfJsonReader typed(input);
        EXPECT_EQ(typed.ParseDouble().status().code(), StatusCode::kInvalidArgument);
        hf::HfJsonReader skipped(input);
        EXPECT_EQ(skipped.SkipValue().code(), StatusCode::kInvalidArgument);
    }
}

TEST(ModelLoader_HfJsonReaderTest, AcceptsJsonNumberForms) {
    const std::pair<std::string_view, double> cases[]{
            {"0", 0},
            {"-0", -0.0},
            {"10", 10},
            {"-10", -10},
            {"0.5", 0.5},
            {"-0.5", -0.5},
            {"1e2", 100},
            {"1E+2", 100},
            {"1e-2", 0.01},
            {"1.5e-2", 0.015},
    };
    for (const auto& [input, expected]: cases) {
        SCOPED_TRACE(input);
        hf::HfJsonReader typed(input);
        const auto value = typed.ParseDouble();
        ASSERT_TRUE(value.ok()) << value.status().ToString();
        EXPECT_DOUBLE_EQ(*value, expected);
        if (input == "-0") EXPECT_TRUE(std::signbit(*value));
        EXPECT_TRUE(typed.AtEnd());
        hf::HfJsonReader skipped(input);
        EXPECT_TRUE(skipped.SkipValue().ok());
        EXPECT_TRUE(skipped.AtEnd());
    }
}

TEST(ModelLoader_HfJsonReaderTest, IntegerReadersRejectNonIntegerTokensAndLeadingZeros) {
    for (const std::string_view input: {"01", "00", "1.0", "1e0", "1x"}) {
        SCOPED_TRACE(input);
        hf::HfJsonReader signed_reader(input);
        EXPECT_EQ(signed_reader.ParseInt64().status().code(), StatusCode::kInvalidArgument);
        hf::HfJsonReader unsigned_reader(input);
        EXPECT_EQ(unsigned_reader.ParseUInt64().status().code(), StatusCode::kInvalidArgument);
    }
    hf::HfJsonReader negative_zero("-0");
    EXPECT_EQ(negative_zero.ParseUInt64().status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfJsonReaderTest, IntegerReadersPreserveRangeChecks) {
    hf::HfJsonReader minimum("-9223372036854775808");
    const auto min_value = minimum.ParseInt64();
    ASSERT_TRUE(min_value.ok());
    EXPECT_EQ(*min_value, std::numeric_limits<int64_t>::min());
    hf::HfJsonReader maximum("9223372036854775807");
    const auto max_value = maximum.ParseInt64();
    ASSERT_TRUE(max_value.ok());
    EXPECT_EQ(*max_value, std::numeric_limits<int64_t>::max());
    for (const std::string_view input: {"9223372036854775808", "-9223372036854775809"}) {
        hf::HfJsonReader reader(input);
        EXPECT_EQ(reader.ParseInt64().status().code(), StatusCode::kInvalidArgument);
    }
    hf::HfJsonReader unsigned_overflow("18446744073709551616");
    EXPECT_EQ(unsigned_overflow.ParseUInt64().status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfJsonReaderTest, SkipsNumbersWithoutRequiringMachineRepresentation) {
    const std::string_view inputs[]{"1e400", "1e-4000", "-1E99999999999999999999"};
    for (const auto input: inputs) {
        SCOPED_TRACE(input);
        hf::HfJsonReader typed(input);
        EXPECT_EQ(typed.ParseDouble().status().code(), StatusCode::kInvalidArgument);
        hf::HfJsonReader skipped(input);
        EXPECT_TRUE(skipped.SkipValue().ok());
        EXPECT_TRUE(skipped.AtEnd());
    }
    const std::string large_integer(1024, '9');
    hf::HfJsonReader skipped(large_integer);
    EXPECT_TRUE(skipped.SkipValue().ok());
    EXPECT_TRUE(skipped.AtEnd());
}

TEST(ModelLoader_HfJsonReaderTest, LeavesNumberDelimiterForEnclosingParser) {
    hf::HfJsonReader reader(" \t-1.5e2,0]");
    const auto first = reader.ParseDouble();
    ASSERT_TRUE(first.ok());
    EXPECT_DOUBLE_EQ(*first, -150);
    EXPECT_TRUE(reader.TryConsume(','));
    const auto second = reader.ParseInt64();
    ASSERT_TRUE(second.ok());
    EXPECT_EQ(*second, 0);
    EXPECT_TRUE(reader.TryConsume(']'));
    EXPECT_TRUE(reader.AtEnd());
}

TEST(ModelLoader_HfJsonReaderTest, AcceptsOnlyJsonWhitespace) {
    hf::HfJsonReader valid(" \t\r\n[1, true, null]");
    EXPECT_TRUE(valid.SkipValue().ok());
    EXPECT_TRUE(valid.AtEnd());
    for (const char c: {'\f', '\v'}) {
        for (const auto& input: {std::string(1, c) + "[1]", "[1," + std::string(1, c) + "2]"}) {
            hf::HfJsonReader reader(input);
            EXPECT_EQ(reader.SkipValue().code(), StatusCode::kInvalidArgument);
        }
    }
}

TEST(ModelLoader_HfJsonReaderTest, RejectsInvalidLiteralSuffixes) {
    for (const std::string_view input: {"truex", "false0", "truefalse"}) {
        hf::HfJsonReader typed(input);
        EXPECT_EQ(typed.ParseBool().status().code(), StatusCode::kInvalidArgument);
    }
    for (const std::string_view input: {"truex", "false0", "nullx", "nulltrue"}) {
        hf::HfJsonReader skipped(input);
        EXPECT_EQ(skipped.SkipValue().code(), StatusCode::kInvalidArgument);
    }
}

TEST(ModelLoader_HfJsonReaderTest, ReportsByteOffsetForNumberSyntaxErrors) {
    const std::pair<std::string_view, size_t> cases[]{{" 01", 2}, {" \n1e+", 5}, {"1x", 1}, {"1.", 2}};
    for (const auto& [input, offset]: cases) {
        hf::HfJsonReader reader(input);
        const auto value = reader.ParseDouble();
        EXPECT_EQ(value.status().code(), StatusCode::kInvalidArgument);
        EXPECT_NE(value.status().message().find("at byte offset " + std::to_string(offset)), std::string::npos);
    }
}

TEST(ModelLoader_HfJsonReaderTest, ReportsUtf8ByteOffsetForDelimiterErrors) {
    hf::HfJsonReader reader("\"\xe4\xb8\xad\" ?");
    ASSERT_TRUE(reader.ParseString().ok());
    const auto status = reader.Expect('[', "after string");
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("Expected '[' after string at byte offset 6"), std::string::npos);
}

TEST(ModelLoader_HfJsonReaderTest, RejectsEveryTruncationOfNestedDocument) {
    constexpr std::string_view json = "{\"a\":[\"\\u4e2d\\u6587\",\"\\uD83D\\uDE00\",\"\xe4\xb8\xad\",1.5e+2,true,null]}";
    for (size_t length = 0; length < json.size(); ++length) {
        SCOPED_TRACE(length);
        hf::HfJsonReader reader(json.substr(0, length));
        const auto status = reader.SkipValue();
        EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
        EXPECT_NE(status.message().find("at byte offset "), std::string::npos);
    }
    hf::HfJsonReader complete(json);
    EXPECT_TRUE(complete.SkipValue().ok());
    EXPECT_TRUE(complete.AtEnd());
}

TEST(ModelLoader_HfJsonReaderTest, EnforcesExactScalarNestingBoundary) {
    const size_t depth = hf::HfJsonReader::kMaxSkipDepth;
    const std::string within_limit = std::string(depth, '[') + "0" + std::string(depth, ']');
    hf::HfJsonReader accepted(within_limit);
    EXPECT_TRUE(accepted.SkipValue().ok());
    EXPECT_TRUE(accepted.AtEnd());
    const std::string beyond_limit = "[" + within_limit + "]";
    hf::HfJsonReader rejected(beyond_limit);
    EXPECT_EQ(rejected.SkipValue().code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfJsonReaderTest, SkipsLongStringsAndNumbersWithoutAllocating) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    const std::string long_key(1024, 'k');
    const std::string long_string(4096, 'x');
    std::string escapes;
    for (size_t i = 0; i < 256; ++i) escapes += R"(\u4e2d\ud83d\ude00)";
    const std::string json = "{\"" + long_key + "\":[\"" + long_string + "\",\"" + escapes + "\",1e400]}";
    hf::HfJsonReader reader(json);
    test::BeginMallocCallCounting();
    const Status status = reader.SkipValue();
    const auto counts = test::EndMallocCallCounting();
    EXPECT_TRUE(status.ok()) << status.ToString();
    EXPECT_TRUE(reader.AtEnd());
    EXPECT_EQ(counts.malloc_calls + counts.calloc_calls + counts.realloc_calls +
                      counts.aligned_alloc_calls + counts.posix_memalign_calls + counts.memalign_calls,
              0U);
}

} // namespace
