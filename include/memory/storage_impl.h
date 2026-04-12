//
// Created by richard on 6/24/25.
//

#ifndef AETHERMIND_STORAGE_IMPL_H
#define AETHERMIND_STORAGE_IMPL_H

#include "../aethermind/memory/allocator.h"
#include "object.h"

namespace aethermind {

// Legacy storage implementation retained only for staged Storage-Tensor ->
// Buffer-Tensor migration. New code should use aethermind::Buffer.

/*! 
 * \brief A storage represents the underlying backing data buffer for a tensor.
 *
 * \note
 */
class StorageImpl : public Object {
public:
    StorageImpl(size_t nbytes, DataPtr data_ptr, const std::unique_ptr<AllocatorBK>& alloc)
        : nbytes_(nbytes), data_ptr_(std::move(data_ptr)), alloc_(alloc) {}

    StorageImpl(size_t nbytes, const std::unique_ptr<AllocatorBK>& alloc)
        : StorageImpl(nbytes, alloc->allocate(nbytes), alloc) {}

    StorageImpl() : StorageImpl(0, AllocatorTable::Global().get_allocator(kUndefined)) {}

    AM_NODISCARD size_t nbytes() const {
        return nbytes_;
    }

    AM_NODISCARD DataPtr& data_ptr() {
        return data_ptr_;
    }

    AM_NODISCARD const DataPtr& const_data_ptr() const {
        return data_ptr_;
    }

    AM_NODISCARD void* get() const {
        return data_ptr_.get();
    }

    AM_NODISCARD const void* const_get() const {
        return data_ptr_.get();
    }

    AM_NODISCARD Device device() const {
        return data_ptr_.device();
    }

    AM_NODISCARD DeviceType device_type() const {
        return data_ptr_.device().type();
    }

    StorageImpl(const StorageImpl&) = delete;
    StorageImpl(StorageImpl&&) noexcept = delete;
    StorageImpl& operator=(const StorageImpl&) = delete;
    StorageImpl& operator=(StorageImpl&&) noexcept = delete;

private:
    size_t nbytes_;
    DataPtr data_ptr_;
    const std::unique_ptr<AllocatorBK>& alloc_;
};

class Storage : public ObjectRef {
public:
    Storage() = default;

    // Creates storage with pre-allocated memory buffer.
    Storage(size_t nbytes, DataPtr data_ptr, const std::unique_ptr<AllocatorBK>& alloc)
        : impl_(make_object<StorageImpl>(nbytes, std::move(data_ptr), alloc)) {}

    // Allocates memory buffer using the given allocator and creates a storage with it
    Storage(size_t nbytes, const std::unique_ptr<AllocatorBK>& alloc)
        : Storage(nbytes, alloc->allocate(nbytes), alloc) {}

    explicit Storage(ObjectPtr<StorageImpl> ptr) : impl_(std::move(ptr)) {}

    // Read-only bridge surface used by migration compat code.
    AM_NODISCARD bool defined() const {
        return impl_;
    }

    AM_NODISCARD size_t nbytes() const {
        return impl_->nbytes();
    }

    AM_NODISCARD void* data() const {
        return impl_->get();
    }

    AM_NODISCARD const void* const_data() const {
        return impl_->const_get();
    }

    AM_NODISCARD DataPtr& data_ptr() const {
        // CHECK(impl_ != nullptr) << "Storage is not initialized";
        return impl_->data_ptr();
    }

    AM_NODISCARD const DataPtr& const_data_ptr() const {
        // CHECK(impl_ != nullptr) << "Storage is not initialized";
        return impl_->data_ptr();
    }

    AM_NODISCARD Device device() const {
        // CHECK(impl_ != nullptr) << "Storage is not initialized";
        return impl_->device();
    }

    operator bool() const {
        return impl_;
    }

    AM_NODISCARD uint32_t use_count() const {
        return impl_.use_count();
    }

    AM_NODISCARD bool unique() const {
        return use_count() == 1;
    }

private:
    ObjectPtr<StorageImpl> impl_;
};

}// namespace aethermind

#endif// AETHERMIND_STORAGE_IMPL_H
