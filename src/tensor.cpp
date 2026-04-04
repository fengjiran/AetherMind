//
// Created by 赵丹 on 25-6-12.
//

#include "tensor.h"
#include <random>

namespace aethermind {

namespace {
void check_type(const Tensor_BK& t, DLDataTypeCode type_code, int8_t type_bits, int16_t type_lanes) {
    AM_CHECK(t.dtype() == DataType(type_code, type_bits, type_lanes));
    // CHECK(t.dtype().code() == type_code && t.dtype().bits() == type_bits && t.dtype().lanes() == type_lanes)
    //         << "data type mismatch.";
}
}// namespace

#define DEFINE_DATA_PTR(type_code, type_bits, type_lanes, T, name) \
    template<>                                                     \
    T* Tensor_BK::data_ptr<T>() const {                            \
        check_type(*this, type_code, type_bits, type_lanes);       \
        return impl_->data_ptr_impl<T>();                          \
    }                                                              \
                                                                   \
    template<>                                                     \
    const T* Tensor_BK::const_data_ptr<T>() const {                \
        check_type(*this, type_code, type_bits, type_lanes);       \
        return impl_->const_data_ptr_impl<T>();                    \
    }                                                              \
                                                                   \
    template<>                                                     \
    const T* Tensor_BK::const_data_ptr<const T>() const {          \
        check_type(*this, type_code, type_bits, type_lanes);       \
        return impl_->const_data_ptr_impl<T>();                    \
    }

SCALAR_TYPE_TO_CPP_TYPE_AND_NAME(DEFINE_DATA_PTR);
#undef DEFINE_DATA_PTR

Tensor_BK::Tensor_BK(const std::vector<int64_t>& shape, int64_t storage_offset, DataType dtype, Device device)
    : impl_(make_object<TensorImpl>(shape, storage_offset, dtype, device)) {}

Tensor_BK::Tensor_BK(ObjectPtr<TensorImpl> impl) : impl_(std::move(impl)) {
    if (impl_ == nullptr) {
        AM_THROW(runtime_error) << "TensorImpl with nullptr is not supported";
    }
}

bool Tensor_BK::defined() const {
    // return impl_->storage_initialized();
    return impl_;
}

uint32_t Tensor_BK::use_count() const {
    return impl_.use_count();
}

bool Tensor_BK::unique() const {
    return impl_.unique();
}

TensorImpl* Tensor_BK::get_impl_ptr_unsafe() const noexcept {
    return impl_.get();
}

TensorImpl* Tensor_BK::release_impl_unsafe() {
    return impl_.release();
}

void* Tensor_BK::data_ptr() const {
    return impl_->data();
}

const void* Tensor_BK::const_data_ptr() const {
    return impl_->const_data();
}

IntArrayView Tensor_BK::shape() const {
    return impl_->shape();
}

IntArrayView Tensor_BK::strides() const {
    return impl_->strides();
}

int64_t Tensor_BK::shape(int64_t dim) const {
    return impl_->shape(dim);
}

int64_t Tensor_BK::strides(int64_t dim) const {
    return impl_->strides(dim);
}

DataType Tensor_BK::dtype() const {
    return impl_->dtype();
}

Device Tensor_BK::device() const {
    return impl_->device();
}

int32_t Tensor_BK::ndim() const {
    return impl_->ndim();
}

int64_t Tensor_BK::numel() const {
    return impl_->numel();
}

size_t Tensor_BK::itemsize() const {
    return impl_->itemsize();
}

size_t Tensor_BK::nbytes() const {
    return numel() * itemsize();
}

bool Tensor_BK::has_storage() const {
    return impl_->has_storage();
}

int64_t Tensor_BK::storage_offset() const {
    return impl_->storage_offset();
}

Layout Tensor_BK::layout() const {
    return impl_->layout();
}

bool Tensor_BK::is_nested() const {
    return impl_->is_nested();
}

bool Tensor_BK::requires_grad() const {
    return impl_->requires_grad();
}

bool Tensor_BK::is_contiguous() const {
    return impl_->is_contiguous();
}

bool Tensor_BK::is_cpu() const {
    return impl_->is_cpu();
}

bool Tensor_BK::is_cuda() const {
    return impl_->is_cuda();
}

Tensor_BK Tensor_BK::rand(const std::vector<int64_t>& shape) {
    Tensor_BK t(shape);
    AM_CHECK(t.numel() > 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1);
    for (int64_t i = 0; i < t.numel(); ++i) {
        t.data_ptr<float>()[i] = dist(gen);
        // static_cast<float*>(t.data())[i] = dist(gen);
    }

    return t;
}

Tensor_BK Tensor_BK::randn(const std::vector<int64_t>& shape) {
    Tensor_BK t(shape);
    AM_CHECK(t.numel() > 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0, 1);
    for (int64_t i = 0; i < t.numel(); ++i) {
        t.data_ptr<float>()[i] = dist(gen);
        // static_cast<float*>(t.data())[i] = dist(gen);
    }
    return t;
}

Tensor_BK Tensor_BK::randint(int64_t low, int64_t high, const std::vector<int64_t>& shape) {
    Tensor_BK t(shape, 0, {DLDataTypeCode::kInt, 64, 1}, Device::CPU());
    AM_CHECK(t.numel() > 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> dist(low, high);
    for (int64_t i = 0; i < t.numel(); ++i) {
        t.data_ptr<int64_t>()[i] = dist(gen);
        // static_cast<int64_t*>(t.data())[i] = dist(gen);
    }
    return t;
}

}// namespace aethermind
