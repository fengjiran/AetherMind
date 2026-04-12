//
// Created by richard on 4/12/26.
//

#include "aethermind/migration/tensor_compat.h"

#include "object_allocator.h"
#include "utils/logging.h"

#include <cstddef>
#include <cstring>
#include <limits>

namespace aethermind {

namespace {

struct LegacyStorageAliasContext {
    Storage storage_;
};

void DeleteLegacyStorageAliasContext(void* deleter_ctx, void* data_ptr) noexcept { // NOLINT(bugprone-easily-swappable-parameters)
    (void) data_ptr;
    delete static_cast<LegacyStorageAliasContext*>(deleter_ctx);
}

char& ZeroByteSentinel() {
    static char sentinel = 0;
    return sentinel;
}

Buffer MakeZeroSizedBuffer(Device device) {
    return {0,
            MemoryHandle(&ZeroByteSentinel(),
                         nullptr,
                         &NoOpMemoryDeleter,
                         device,
                         alignof(std::max_align_t))};
}

Storage CopyStorageFromBuffer(const Buffer& buffer) {
    AM_CHECK(buffer.is_initialized(), "Cannot convert uninitialized Buffer to legacy Storage.");

    const auto device = buffer.device();
    const auto nbytes = buffer.nbytes();
    auto& allocator = AllocatorTable::Global().get_allocator(device.type());
    Storage storage(nbytes, allocator);

    if (nbytes == 0) {
        return storage;
    }

    AM_CHECK(device.is_cpu(), "Legacy Tensor compatibility currently supports CPU Buffer copies only.");
    AM_CHECK(buffer.data() != nullptr, "Initialized Buffer must expose a data pointer for non-zero size copy.");
    AM_CHECK(storage.data() != nullptr, "Legacy Storage allocation failed for Tensor compatibility conversion.");
    std::memcpy(storage.data(), buffer.data(), nbytes);
    return storage;
}

size_t LegacyStorageOffsetBytes(const Tensor_BK& legacy) {
    const auto itemsize = legacy.itemsize();
    const auto storage_offset = legacy.storage_offset();
    AM_CHECK(storage_offset >= 0, "Legacy tensor storage_offset must be non-negative.");

    const auto storage_offset_u64 = static_cast<uint64_t>(storage_offset);
    AM_CHECK(storage_offset_u64 <= static_cast<uint64_t>(std::numeric_limits<size_t>::max()),
             "Legacy tensor storage_offset exceeds size_t range.");

    const auto storage_offset_size = static_cast<size_t>(storage_offset_u64);
    AM_CHECK(itemsize == 0 || storage_offset_size <= std::numeric_limits<size_t>::max() / itemsize,
             "Legacy tensor byte_offset conversion overflow.");
    return storage_offset_size * itemsize;
}

}// namespace

Buffer BufferFromLegacyStorage(const Storage& storage) {
    if (!storage.defined()) {
        return {};
    }

    if (storage.nbytes() == 0) {
        return MakeZeroSizedBuffer(storage.device());
    }

    auto* ctx = new LegacyStorageAliasContext{storage};
    return {storage.nbytes(), MemoryHandle(storage.data(), ctx, &DeleteLegacyStorageAliasContext, storage.device(), 0)};
}

Tensor TensorFromLegacy(const Tensor_BK& legacy) {
    if (!legacy.defined()) {
        return {};
    }

    AM_CHECK(legacy.has_storage(), "Cannot convert legacy Tensor without Storage to the new Tensor abstraction.");

    const auto shape = legacy.shape();
    const auto strides = legacy.strides();
    Buffer buffer = BufferFromLegacyStorage(legacy.get_impl_ptr_unsafe()->storage());
    const size_t byte_offset = LegacyStorageOffsetBytes(legacy);
    return {std::move(buffer), byte_offset, legacy.dtype(), shape, strides};
}

Tensor_BK LegacyTensorFromTensor(const Tensor& tensor) {
    if (!tensor.is_initialized()) {
        return {};
    }

    const auto itemsize = tensor.itemsize();
    AM_CHECK(itemsize > 0, "Cannot convert Tensor with non-positive itemsize to legacy Tensor.");
    AM_CHECK(tensor.byte_offset() % itemsize == 0,
             "Tensor byte_offset {} is not representable as legacy storage_offset for itemsize {}.",
             tensor.byte_offset(), itemsize);

    Storage storage = CopyStorageFromBuffer(tensor.buffer());
    auto impl = make_object<TensorImpl>(std::move(storage), tensor.dtype(), tensor.device());
    impl->set_shape_and_strides(tensor.shape(), tensor.strides(),
                                static_cast<int64_t>(tensor.byte_offset() / itemsize));
    return Tensor_BK(std::move(impl));
}

}// namespace aethermind
