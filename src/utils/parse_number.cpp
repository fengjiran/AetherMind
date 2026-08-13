#include "utils/parse_number.h"

#include <cerrno>
#include <cstdlib>
#include <string>

#if defined(__APPLE__) && defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && \
        __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 260000
#include <xlocale.h>
#define AETHERMIND_FP_FROM_CHARS_FALLBACK 1
#else
#include <charconv>
#include <system_error>
#endif

namespace aethermind::utils {

#ifdef AETHERMIND_FP_FROM_CHARS_FALLBACK

namespace {

// Cached "C" locale for strtod_l/strtof_l. Parsing with the "C" locale keeps
// the decimal point fixed and matches std::from_chars semantics; the locale
// lives for the process lifetime and is never freed (a single tiny object).
locale_t CCachedLocale() noexcept {
    static const locale_t kCLocale = newlocale(LC_ALL_MASK, "C", nullptr);
    return kCLocale;
}

bool ParseFloatFallback(std::string_view text, float& out) noexcept {
    const std::string buffer{text};
    errno = 0;
    char* end = nullptr;
    const float value = strtof_l(buffer.c_str(), &end, CCachedLocale());
    if (errno != 0 || end != buffer.c_str() + buffer.size()) {
        return false;
    }
    out = value;
    return true;
}

bool ParseDoubleFallback(std::string_view text, double& out) noexcept {
    const std::string buffer{text};
    errno = 0;
    char* end = nullptr;
    const double value = strtod_l(buffer.c_str(), &end, CCachedLocale());
    if (errno != 0 || end != buffer.c_str() + buffer.size()) {
        return false;
    }
    out = value;
    return true;
}

}// namespace

bool ParseFloat(std::string_view text, float& out) noexcept {
    return ParseFloatFallback(text, out);
}

bool ParseDouble(std::string_view text, double& out) noexcept {
    return ParseDoubleFallback(text, out);
}

#else

bool ParseFloat(std::string_view text, float& out) noexcept {
    const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), out);
    return ec == std::errc{} && ptr == text.data() + text.size();
}

bool ParseDouble(std::string_view text, double& out) noexcept {
    const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), out);
    return ec == std::errc{} && ptr == text.data() + text.size();
}

#endif

}// namespace aethermind::utils

#undef AETHERMIND_FP_FROM_CHARS_FALLBACK