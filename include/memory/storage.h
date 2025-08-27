//
// Created by 赵丹 on 25-6-27.
//

#ifndef AETHERMIND_STORAGE_H
#define AETHERMIND_STORAGE_H

#include "container/array_view.h"
#include "memory/storage_impl.h"

namespace aethermind {

class Storage {
public:
    Storage() = default;

    // Creates storage with pre-allocated memory buffer.
    Storage(size_t nbytes, DataPtr data_ptr, const std::unique_ptr<Allocator>& alloc)
        : impl_(make_object<StorageImpl>(nbytes, std::move(data_ptr), alloc)) {}

    // Allocates memory buffer using the given allocator and creates a storage with it
    Storage(size_t nbytes, const std::unique_ptr<Allocator>& alloc)
        : Storage(nbytes, alloc->allocate(nbytes), alloc) {}

    explicit Storage(ObjectPtr<StorageImpl> ptr) : impl_(std::move(ptr)) {}

    NODISCARD bool defined() const {
        return impl_;
    }

    NODISCARD size_t nbytes() const {
        return impl_->nbytes();
    }

    NODISCARD void* data() const {
        return impl_->get();
    }

    NODISCARD const void* const_data() const {
        return impl_->const_get();
    }

    NODISCARD DataPtr& data_ptr() const {
        // CHECK(impl_ != nullptr) << "Storage is not initialized";
        return impl_->data_ptr();
    }

    NODISCARD const DataPtr& const_data_ptr() const {
        // CHECK(impl_ != nullptr) << "Storage is not initialized";
        return impl_->data_ptr();
    }

    NODISCARD Device device() const {
        // CHECK(impl_ != nullptr) << "Storage is not initialized";
        return impl_->device();
    }

    operator bool() const {
        return impl_;
    }

    NODISCARD uint32_t use_count() const {
        return impl_.use_count();
    }

    NODISCARD bool unique() const {
        return use_count() == 1;
    }

private:
    ObjectPtr<StorageImpl> impl_;
};

}// namespace aethermind

#endif//AETHERMIND_STORAGE_H
