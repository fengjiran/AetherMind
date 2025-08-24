//
// Created by richard on 6/24/25.
//

#ifndef AETHERMIND_STORAGE_IMPL_H
#define AETHERMIND_STORAGE_IMPL_H

#include "allocator.h"
#include "object.h"

namespace aethermind {

class StorageImpl : public Object {
public:
    StorageImpl(size_t nbytes, DataPtr data_ptr, const std::unique_ptr<Allocator>& alloc)
        : nbytes_(nbytes), data_ptr_(std::move(data_ptr)), alloc_(alloc) {}

    StorageImpl(size_t nbytes, const std::unique_ptr<Allocator>& alloc)
        : nbytes_(nbytes), data_ptr_(alloc->allocate(nbytes)), alloc_(alloc) {}

    NODISCARD size_t nbytes() const {
        return nbytes_;
    }

    NODISCARD DataPtr& data_ptr() {
        return data_ptr_;
    }

    NODISCARD const DataPtr& const_data_ptr() const {
        return data_ptr_;
    }

    NODISCARD void* get() const {
        return data_ptr_.get();
    }

    NODISCARD const void* const_get() const {
        return data_ptr_.get();
    }

    NODISCARD Device device() const {
        return data_ptr_.device();
    }

    NODISCARD DeviceType device_type() const {
        return data_ptr_.device().type();
    }

    StorageImpl() = delete;
    StorageImpl(const StorageImpl&) = delete;
    StorageImpl(StorageImpl&&) noexcept = delete;
    StorageImpl& operator=(const StorageImpl&) = delete;
    StorageImpl& operator=(StorageImpl&&) noexcept = delete;

private:
    size_t nbytes_;
    DataPtr data_ptr_;
    const std::unique_ptr<Allocator>& alloc_;
};

class StorageImplNullType final : public StorageImpl {
    static StorageImplNullType singleton_;
    StorageImplNullType() : StorageImpl(0, AllocatorTable::Global().get_allocator(kUndefined)) {}

public:
    static constexpr StorageImpl* singleton() noexcept {
        return &singleton_;
    }
};

template<>
struct GetNullType<StorageImpl> {
    using type = StorageImplNullType;
};

static_assert(std::is_same_v<null_type<StorageImpl>, StorageImplNullType>);

}// namespace aethermind

#endif//AETHERMIND_STORAGE_IMPL_H
