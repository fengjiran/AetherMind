#define _GNU_SOURCE

#include <errno.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

// Test-only preload: force the existing allocator warning without changing allocation.
int madvise(void* address, size_t length, int advice) {
    if (advice == MADV_HUGEPAGE) {
        errno = EINVAL;
        return -1;
    }
    return (int) syscall(SYS_madvise, address, length, advice);
}
