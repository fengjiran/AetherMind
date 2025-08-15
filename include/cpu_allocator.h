//
// Created by 赵丹 on 25-6-25.
//

#ifndef AETHERMIND_CPU_ALLOCATOR_H
#define AETHERMIND_CPU_ALLOCATOR_H

#include "allocator.h"

namespace aethermind {

void* alloc_cpu(size_t nbytes);

void free_cpu(void* data);

class CPUAllocator final : public Allocator {
public:
    CPUAllocator() = default;
    //
    // NODISCARD void* allocate(size_t nbytes) const override {
    //     // return malloc(n);
    //     return alloc_cpu(nbytes);
    // }

    NODISCARD DataPtr allocate(size_t nbytes) const override {
        void* data = alloc_cpu(nbytes);
        return {data, data, deleter, Device(kCPU)};
    }

    static void deleter(void* ptr) {
        if (ptr == nullptr) {
            return;
        }
        free_cpu(ptr);
    }

    void deallocate(void* p) const override {
        // free(p);
        free_cpu(p);
    }
};

}// namespace aethermind

#endif//AETHERMIND_CPU_ALLOCATOR_H
