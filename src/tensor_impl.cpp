//
// Created by richard on 6/22/25.
//

#include "tensor_impl.h"
#include "error.h"

namespace aethermind {

TensorImplNullType TensorImplNullType::singleton_;

TensorImpl::TensorImpl(const std::vector<int64_t>& shape, int64_t storage_offset, DataType dtype, Device device)
    : storage_offset_(storage_offset), numel_(0), dtype_(dtype), device_opt_(device) {
    CHECK(dtype_initialized()) << "dtype should be initialized.";
    CHECK(device.type() != DeviceType::kUndefined) << "device should be initialized.";

    init_bitfield();
    auto ndim = shape.size();
    std::vector<int64_t> strides(ndim, -1);
    set_shape_and_strides(shape, strides);

    int64_t nbytes = numel() * this->dtype().nbytes();
    storage_ = Storage(nbytes, AllocatorTable::Global().get_allocator(device.type()));
}

TensorImpl::TensorImpl(Storage&& storage, DataType dtype, std::optional<Device> device_opt)
    : storage_(std::move(storage)), numel_(0), dtype_(dtype), device_opt_(device_opt) {
    init_bitfield();
}

TensorImpl::TensorImpl(DataType dtype, std::optional<Device> device_opt)
    : TensorImpl({}, dtype, device_opt) {}

TensorImpl::TensorImpl(Storage&& storage, DataType dtype)
    : TensorImpl(std::move(storage), dtype, storage.device()) {}


int64_t TensorImpl::numel() const {
    return numel_;
}

bool TensorImpl::empty() const {
    return numel_ == 0;
}

int64_t TensorImpl::ndim() const {
    return shape_and_stride_.size();
}

IntArrayView TensorImpl::shape() const {
    return shape_and_stride_.get_shape();
}

int64_t TensorImpl::shape(int64_t dim) const {
    dim = get_real_dim(dim);
    return shape_and_stride_.shape_at_uncheck(dim);
}

IntArrayView TensorImpl::strides() const {
    return shape_and_stride_.get_strides();
}

int64_t TensorImpl::strides(int64_t dim) const {
    dim = get_real_dim(dim);
    return shape_and_stride_.stride_at_uncheck(dim);
}

size_t TensorImpl::itemsize() const {
    // CHECK(dtype_initialized()) << "Can't get item sizer of Tensor that doesn't have initialized dtype.";
    if (!dtype_initialized()) {
        throw std::runtime_error("Can't get item size of Tensor that doesn't have initialized dtype.");
    }
    return dtype_.nbytes();
}

bool TensorImpl::has_storage() const {
    return storage_;
}

const Storage& TensorImpl::storage() const {
    return storage_;
}

bool TensorImpl::storage_initialized() const {
    // CHECK(has_storage()) << "Can't call storage_initialized() on a tensor that doesn't have storage.";
    return has_storage() && (storage_.const_data() != nullptr || numel_ == 0);
}

bool TensorImpl::dtype_initialized() const {
    return dtype_ != DataType();
}

int64_t TensorImpl::storage_offset() const {
    return storage_offset_;
}

Device TensorImpl::device() const {
    CHECK(device_opt_.has_value()) << "tensor does not have a device.";
    return *device_opt_;
}

bool TensorImpl::is_cpu() const {
    return device_opt_.has_value() && device_opt_->is_cpu();
}

bool TensorImpl::is_cuda() const {
    return device_opt_.has_value() && device_opt_->is_cuda();
}

int64_t TensorImpl::get_real_dim(int64_t dim) const {
    if (dim >= -1 * ndim() && dim < ndim()) {
        if (dim < 0) {
            return dim + ndim();
        }
        return dim;
    }
    throw std::out_of_range("dim out of range.");
}

DataType TensorImpl::dtype() const {
    return dtype_;
}

bool TensorImpl::is_contiguous() const {
    return is_contiguous_;
}

void TensorImpl::set_shape_and_strides(IntArrayView shape, IntArrayView strides, std::optional<int64_t> storage_offset) {
    CHECK(shape.size() == strides.size()) << "dimensionality of shape must match dimensionality of strides.";
    auto ndim = shape.size();
    shape_and_stride_.set_shape(shape);

    if (ndim > 0) {
        bool overflowed = false;
        for (size_t i = ndim - 1;; --i) {
            if (strides[i] >= 0) {
                shape_and_stride_.stride_at_uncheck(i) = strides[i];
            } else {
                if (i == ndim - 1) {
                    shape_and_stride_.stride_at_uncheck(i) = 1;
                } else {
                    overflowed |= mul_overflow(
                            shape_and_stride_.stride_at_uncheck(i + 1),
                            std::max<int64_t>(shape_and_stride_.shape_at_uncheck(i + 1), 1),
                            &shape_and_stride_.stride_at_uncheck(i));
                }
            }

            if (i == 0) {
                break;
            }
        }
        CHECK(!overflowed) << "stride calculation overflowed.";
    }

    refresh_numel();
    refresh_contiguous();
    if (storage_offset.has_value()) {
        storage_offset_ = storage_offset.value();
    }
}

void TensorImpl::set_shape_contiguous(IntArrayView shape) {
    shape_and_stride_.set_shape(shape);
    refresh_numel();
    refresh_contiguous();
}


int64_t TensorImpl::safe_compute_numel() const {
    uint64_t numel = 1;
    bool overflow = safe_multiply_u64(shape_and_stride_.get_shape(), &numel);
    constexpr auto numel_max = std::min<uint64_t>(
            std::numeric_limits<int64_t>::max(),
            std::numeric_limits<size_t>::max());
    overflow |= numel > numel_max;
    CHECK(!overflow) << "interger multiplication overflow when compute numel.";
    return static_cast<int64_t>(numel);
}

void TensorImpl::refresh_numel() {
    numel_ = safe_compute_numel();
}

bool TensorImpl::compute_contiguous() const {
    return _compute_contiguous(shape(), strides());
}

void TensorImpl::set_contiguous(bool b) {
    is_contiguous_ = b;
}

void TensorImpl::refresh_contiguous() {
    set_contiguous(compute_contiguous());
}

void TensorImpl::set_storage_keep_dtype(Storage storage) {
    storage_ = std::move(storage);
    device_opt_ = storage_.device();
}

void TensorImpl::set_storage_and_dtype(Storage storage, DataType dtype) {
    set_storage_keep_dtype(std::move(storage));
    dtype_ = dtype;
}


void TensorImpl::set_storage_offset(int64_t storage_offset) {
    CHECK(storage_offset >= 0) << "storage_offset must be non-negative.";
    storage_offset_ = storage_offset;
}

void* TensorImpl::data() const {
    return data_impl<void>(
            [this] {
                return static_cast<char*>(storage_.data());
            });
}

const void* TensorImpl::const_data() const {
    return data_impl<const void>(
            [this] {
                return static_cast<const char*>(storage_.const_data());
            });
}

void TensorImpl::init_bitfield() {
    is_contiguous_ = true;
}

}// namespace aethermind
