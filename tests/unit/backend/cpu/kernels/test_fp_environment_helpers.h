#ifndef AETHERMIND_TEST_CPU_KERNELS_FP_ENVIRONMENT_HELPERS_H
#define AETHERMIND_TEST_CPU_KERNELS_FP_ENVIRONMENT_HELPERS_H

#include <cfenv>

#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

namespace aethermind::test {

// Numerical kernel contracts use round-to-nearest and gradual underflow. Keep
// test-local changes from leaking into other suites, including exception flags.
class ScopedNearestGradualUnderflow {
public:
    ScopedNearestGradualUnderflow() noexcept {
        saved_ = std::fegetenv(&environment_) == 0;
        if (!saved_) {
            return;
        }
#if defined(__x86_64__) || defined(_M_X64)
        control_ = _mm_getcsr();
#endif
        nearest_ = std::fesetround(FE_TONEAREST) == 0;
#if defined(__x86_64__) || defined(_M_X64)
        _mm_setcsr(_mm_getcsr() & ~(0x8000U | 0x0040U));
#endif
    }

    ~ScopedNearestGradualUnderflow() {
        if (saved_) {
            std::fesetenv(&environment_);
#if defined(__x86_64__) || defined(_M_X64)
            _mm_setcsr(control_);
#endif
        }
    }

    ScopedNearestGradualUnderflow(const ScopedNearestGradualUnderflow&) = delete;
    ScopedNearestGradualUnderflow& operator=(const ScopedNearestGradualUnderflow&) = delete;

    bool ok() const noexcept { return saved_ && nearest_; }

private:
    std::fenv_t environment_{};
    bool saved_ = false;
    bool nearest_ = false;
#if defined(__x86_64__) || defined(_M_X64)
    unsigned control_ = 0;
#endif
};

} // namespace aethermind::test

#endif // AETHERMIND_TEST_CPU_KERNELS_FP_ENVIRONMENT_HELPERS_H
