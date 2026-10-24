#include "utils/parse_number.h"
#include "parse_number_internal.h"

#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>

#if defined(__APPLE__) && defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && \
        __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 260000
#define AETHERMIND_FP_FROM_CHARS_FALLBACK 1
#else
#include <charconv>
#include <system_error>
#endif

#if defined(__APPLE__)
#include <xlocale.h>
#elif defined(__linux__)
#include <locale.h>
#endif

namespace aethermind::utils {

#if defined(__APPLE__) || defined(__linux__)
namespace {

bool IsDigit(char value) noexcept {
    return value >= '0' && value <= '9';
}

char LowerAscii(char value) noexcept {
    return value >= 'A' && value <= 'Z' ? static_cast<char>(value - 'A' + 'a') : value;
}

bool EqualIgnoringAsciiCase(std::string_view value, std::string_view expected) noexcept {
    if (value.size() != expected.size()) {
        return false;
    }
    for (size_t i = 0; i < value.size(); ++i) {
        if (LowerAscii(value[i]) != expected[i]) {
            return false;
        }
    }
    return true;
}

bool IsGeneralLiteral(std::string_view text, bool& nonzero_significand) noexcept {
    if (text.empty()) {
        return false;
    }
    if (text.front() == '-') {
        text.remove_prefix(1);
    }
    if (EqualIgnoringAsciiCase(text, "inf") ||
        EqualIgnoringAsciiCase(text, "infinity") ||
        EqualIgnoringAsciiCase(text, "nan")) {
        return true;
    }
    if (text.size() >= 5 && EqualIgnoringAsciiCase(text.substr(0, 4), "nan(") &&
        text.back() == ')') {
        for (const char value: text.substr(4, text.size() - 5)) {
            const char lower = LowerAscii(value);
            if (!IsDigit(value) && !(lower >= 'a' && lower <= 'z') && value != '_') {
                return false;
            }
        }
        return true;
    }

    size_t position = 0;
    bool has_digits = false;
    const auto consume_digits = [&] {
        while (position < text.size() && IsDigit(text[position])) {
            has_digits = true;
            nonzero_significand |= text[position] != '0';
            ++position;
        }
    };
    consume_digits();
    if (position < text.size() && text[position] == '.') {
        ++position;
        consume_digits();
    }
    if (!has_digits) {
        return false;
    }
    if (position < text.size() && (text[position] == 'e' || text[position] == 'E')) {
        ++position;
        if (position < text.size() && (text[position] == '+' || text[position] == '-')) {
            ++position;
        }
        const size_t exponent_start = position;
        while (position < text.size() && IsDigit(text[position])) {
            ++position;
        }
        if (position == exponent_start) {
            return false;
        }
    }
    return position == text.size();
}

class NumericCLocale {
public:
    NumericCLocale() noexcept : locale_(newlocale(LC_NUMERIC_MASK, "C", nullptr)) {}
    ~NumericCLocale() {
        if (locale_ != nullptr) {
            freelocale(locale_);
        }
    }

    NumericCLocale(const NumericCLocale&) = delete;
    NumericCLocale& operator=(const NumericCLocale&) = delete;

    locale_t get() const noexcept { return locale_; }

private:
    locale_t locale_;
};

class NearestRoundingMode {
public:
    NearestRoundingMode() noexcept : original_(std::fegetround()) {
        if (original_ == -1) {
            return;
        }
        if (original_ == FE_TONEAREST) {
            ready_ = true;
            return;
        }
        changed_ = true;
        ready_ = std::fesetround(FE_TONEAREST) == 0;
    }

    ~NearestRoundingMode() { Restore(); }

    NearestRoundingMode(const NearestRoundingMode&) = delete;
    NearestRoundingMode& operator=(const NearestRoundingMode&) = delete;

    bool ready() const noexcept { return ready_; }

    bool Restore() noexcept {
        if (!changed_) {
            return true;
        }
        if (std::fesetround(original_) != 0) {
            return false;
        }
        changed_ = false;
        return true;
    }

private:
    int original_;
    bool changed_ = false;
    bool ready_ = false;
};

template<typename T>
bool ParseFallback(std::string_view text, T& out) noexcept try {
    bool nonzero_significand = false;
    if (!IsGeneralLiteral(text, nonzero_significand)) {
        return false;
    }

    // strto*_l follows the thread's rounding mode, whereas from_chars requires
    // nearest rounding. Restore it on every exit, without clearing host flags.
    NearestRoundingMode rounding;
    if (!rounding.ready()) {
        return false;
    }
    const std::string buffer{text};
    // A failed locale creation is recoverable on the next call. No global
    // locale changes or process-lifetime locale allocations are needed.
    const NumericCLocale locale;
    if (locale.get() == nullptr) {
        return false;
    }

    errno = 0;
    char* end = nullptr;
    const T value = [&] {
        if constexpr (std::is_same_v<T, float>) {
            return strtof_l(buffer.c_str(), &end, locale.get());
        } else {
            return strtod_l(buffer.c_str(), &end, locale.get());
        }
    }();
    if (end != buffer.c_str() + buffer.size() ||
        (errno != 0 && errno != ERANGE)) {
        return false;
    }
    // C conversion may report ERANGE for a representable, nonzero subnormal.
    // Reject only overflow and a nonzero significand rounded down to zero.
    if (errno == ERANGE &&
        (!std::isfinite(value) || (value == 0 && nonzero_significand))) {
        return false;
    }
    if (!rounding.Restore()) {
        return false;
    }
    out = value;
    return true;
} catch (const std::bad_alloc&) {
    return false;
} catch (const std::length_error&) {
    return false;
}

} // namespace

namespace detail {

bool ParseFloatFallback(std::string_view text, float& out) noexcept {
    return ParseFallback(text, out);
}

bool ParseDoubleFallback(std::string_view text, double& out) noexcept {
    return ParseFallback(text, out);
}

} // namespace detail
#endif

namespace {

template<typename T>
bool Parse(std::string_view text, T& out) noexcept {
#ifdef AETHERMIND_FP_FROM_CHARS_FALLBACK
    return ParseFallback(text, out);
#else
    if (text.empty()) {
        return false;
    }
    T value;
    const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || ptr != text.data() + text.size()) {
        return false;
    }
    out = value;
    return true;
#endif
}

} // namespace

bool ParseFloat(std::string_view text, float& out) noexcept {
    return Parse(text, out);
}

bool ParseDouble(std::string_view text, double& out) noexcept {
    return Parse(text, out);
}

} // namespace aethermind::utils

#undef AETHERMIND_FP_FROM_CHARS_FALLBACK
