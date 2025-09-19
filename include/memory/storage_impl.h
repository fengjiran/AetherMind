//
// Created by richard on 6/24/25.
//

#ifndef AETHERMIND_STORAGE_IMPL_H
#define AETHERMIND_STORAGE_IMPL_H

#include "memory/allocator.h"
#include "object.h"

namespace aethermind {

/*!
 * \brief A storage represents the underlying backing data buffer for a tensor.
 *
 * \note
 */
class StorageImpl : public Object {
public:
    StorageImpl(size_t nbytes, DataPtr data_ptr, const std::unique_ptr<Allocator>& alloc)
        : nbytes_(nbytes), data_ptr_(std::move(data_ptr)), alloc_(alloc) {}

    StorageImpl(size_t nbytes, const std::unique_ptr<Allocator>& alloc)
        : StorageImpl(nbytes, alloc->allocate(nbytes), alloc) {}

    StorageImpl() : StorageImpl(0, AllocatorTable::Global().get_allocator(kUndefined)) {}

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

    StorageImpl(const StorageImpl&) = delete;
    StorageImpl(StorageImpl&&) noexcept = delete;
    StorageImpl& operator=(const StorageImpl&) = delete;
    StorageImpl& operator=(StorageImpl&&) noexcept = delete;

private:
    size_t nbytes_;
    DataPtr data_ptr_;
    const std::unique_ptr<Allocator>& alloc_;
};

DEFINE_OBJECT_NULLTYPE(StorageImpl);

}// namespace aethermind

#endif//AETHERMIND_STORAGE_IMPL_H
