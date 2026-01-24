//
// Created by 赵丹 on 25-6-25.
//

#include "memory/cpu_allocator.h"
#include "alignment.h"
#include "env.h"

#include <cstring>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace aethermind {

namespace {

// Fill the data memory region of num bytes with a particular garbage pattern.
// The garbage value is chosen to be NaN if interpreted as floating point value
// or a very large integer.
void memset_junk(void* data, size_t num) {
    // This garbage pattern is NaN when interpreted as floating point values
    // or as very large integer values.
    static constexpr int32_t kJunkPattern = 0x7fedbeef;
    static constexpr int64_t kJunkPattern64 = static_cast<int64_t>(kJunkPattern) << 32 | kJunkPattern;
    auto int64_count = num / sizeof(kJunkPattern64);
    auto remaining_bytes = num % sizeof(kJunkPattern64);
    auto* data_i64 = static_cast<int64_t*>(data);
    for (size_t i = 0; i < int64_count; ++i) {
        data_i64[i] = kJunkPattern64;
    }

    if (remaining_bytes > 0) {
        memcpy(data_i64 + int64_count, &kJunkPattern64, remaining_bytes);
    }
}

bool is_thp_alloc_enabled() {
    static bool value = [&] {
        auto env = check_env("THP_MEM_ALLOC_ENABLE");
        return env.has_value() ? env.value() : 0;
    }();
    return value;
}

bool is_thp_alloc(size_t nbytes) {
    return is_thp_alloc_enabled() && nbytes >= gAlloc_threshold_thp;
}

}// namespace

size_t get_alignment(size_t nbytes) {
    static const auto pagesize = sysconf(_SC_PAGESIZE);
    const size_t thp_alignment = pagesize < 0 ? gPagesize : pagesize;
    return is_thp_alloc(nbytes) ? thp_alignment : gAlignment;
}

void* alloc_cpu(size_t nbytes) {
    if (nbytes == 0) {
        return nullptr;
    }

    AM_CHECK(static_cast<ptrdiff_t>(nbytes) >= 0, "the nbytes is seems a negative number.");

    void* data = nullptr;
    int err = posix_memalign(&data, get_alignment(nbytes), nbytes);
    AM_CHECK(err == 0, "Try allocate {} bytes memory failed.", nbytes);

    if (is_thp_alloc(nbytes)) {
#ifdef __linux__
        int ret = madvise(data, nbytes, MADV_HUGEPAGE);
        if (ret != 0) {
            spdlog::warn("thp madvise for HUGEPAGE failed");
        }
#endif
    }

    // TODO: set numa node

    memset_junk(data, nbytes);

    return data;
}

void free_cpu(void* data) {
    free(data);
}

REGISTER_ALLOCATOR(DeviceType::kCPU, CPUAllocator);

}// namespace aethermind
