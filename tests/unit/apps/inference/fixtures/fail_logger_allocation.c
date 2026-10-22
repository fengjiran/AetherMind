#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>

#if !defined(__linux__) || !defined(__GLIBC__)
#error "Logger allocation injection requires Linux/glibc"
#endif

typedef int (*main_function)(int, char**, char**);
typedef int (*start_main_function)(main_function, int, char**,
                                   void (*)(void), void (*)(void), void (*)(void), void*);

extern void* __libc_malloc(size_t size);

static main_function original_main;
static _Thread_local int fail_next_allocation;

void* malloc(size_t size) {
    if (fail_next_allocation) {
        // The exception and fixed diagnostic remain free to allocate afterwards.
        fail_next_allocation = 0;
        errno = ENOMEM;
        return NULL;
    }
    return __libc_malloc(size);
}

static int wrapped_main(int argc, char** argv, char** environment) {
    // Arm after library/static initialization, directly before the tool's main.
    fail_next_allocation = 1;
    return original_main(argc, argv, environment);
}

int __libc_start_main(main_function main, int argc, char** argv,
                      void (*init)(void), void (*fini)(void),
                      void (*rtld_fini)(void), void* stack_end) {
    start_main_function start = (start_main_function) dlsym(RTLD_NEXT, "__libc_start_main");
    if (start == NULL) {
        _exit(127);
    }
    original_main = main;
    return start(&wrapped_main, argc, argv, init, fini, rtld_fini, stack_end);
}
