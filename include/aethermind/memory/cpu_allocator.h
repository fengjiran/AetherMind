//
// Created by 赵丹 on 25-6-25.
//

#ifndef AETHERMIND_CPU_ALLOCATOR_H
#define AETHERMIND_CPU_ALLOCATOR_H

#include "aethermind/memory/buffer.h"
#include "allocator.h"
#include <cstddef>

namespace aethermind {

void* alloc_cpu(size_t nbytes);

void free_cpu(void* data);

class CPUAllocator final : public Allocator {
public:
    explicit CPUAllocator(Device device) : device_(device) {}

    Buffer Allocate(size_t nbytes) override;

    AM_NODISCARD Device device() const noexcept override;

private:
    Device device_;
};

class CPUAllocatorProvider final : public AllocatorProvider {
public:
    std::unique_ptr<Allocator> CreateAllocator(Device device) override;
};

}// namespace aethermind

#endif// AETHERMIND_CPU_ALLOCATOR_H
