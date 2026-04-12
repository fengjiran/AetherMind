//
// Created by 赵丹 on 25-6-12.
//

#ifndef AETHERMIND_TENSOR_H
#define AETHERMIND_TENSOR_H

#include "tensor_impl.h"

namespace aethermind {

// Legacy Tensor wrapper retained only for staged Storage-Tensor -> Buffer-Tensor
// migration. New features should target aethermind::Tensor instead.
class Tensor_BK {
public:
    Tensor_BK() = default;

    explicit Tensor_BK(const std::vector<int64_t>& shape,
                    int64_t storage_offset = 0,
                    DataType dtype = DataType::Float32(),
                    Device device = Device::CPU());

    explicit Tensor_BK(ObjectPtr<TensorImpl> impl);

    Tensor_BK(const Tensor_BK&) = default;
    Tensor_BK(Tensor_BK&&) = default;

    Tensor_BK& operator=(const Tensor_BK&) & = default;
    Tensor_BK& operator=(Tensor_BK&&) & noexcept = default;

    Tensor_BK& operator=(const Tensor_BK&) && = default;
    Tensor_BK& operator=(Tensor_BK&&) && noexcept = default;

    AM_NODISCARD bool defined() const;

    AM_NODISCARD uint32_t use_count() const;

    AM_NODISCARD bool unique() const;

    AM_NODISCARD IntArrayView shape() const;

    AM_NODISCARD IntArrayView strides() const;

    AM_NODISCARD int64_t shape(int64_t dim) const;

    AM_NODISCARD int64_t strides(int64_t dim) const;

    AM_NODISCARD DataType dtype() const;

    AM_NODISCARD Device device() const;

    AM_NODISCARD int32_t ndim() const;

    AM_NODISCARD int64_t numel() const;

    AM_NODISCARD size_t itemsize() const;

    AM_NODISCARD size_t nbytes() const;

    AM_NODISCARD bool has_storage() const;

    AM_NODISCARD int64_t storage_offset() const;

    AM_NODISCARD Layout layout() const;

    AM_NODISCARD bool is_nested() const;

    AM_NODISCARD bool requires_grad() const;

    AM_NODISCARD bool is_contiguous() const;

    AM_NODISCARD bool is_cpu() const;

    AM_NODISCARD bool is_cuda() const;

    // Returns the underlying legacy impl without transferring ownership.
    // Migration code may inspect it transiently but must not retain the raw
    // pointer beyond the lifetime of this Tensor_BK.
    AM_NODISCARD TensorImpl* get_impl_ptr_unsafe() const noexcept;

    // Releases ownership of the underlying legacy impl. This is migration-only
    // escape hatch code and callers become responsible for rebuilding an
    // owning ObjectPtr<TensorImpl> immediately.
    AM_NODISCARD TensorImpl* release_impl_unsafe();

    // NODISCARD Scalar item() const;

    AM_NODISCARD void* data_ptr() const;

    AM_NODISCARD const void* const_data_ptr() const;

    template<typename T>
    T* data_ptr() const;

    template<typename T,
             std::enable_if_t<!std::is_const_v<T>>* = nullptr>
    const T* const_data_ptr() const;

    template<typename T,
             std::enable_if_t<std::is_const_v<T>>* = nullptr>
    const std::remove_const_t<T>* const_data_ptr() const;

    // returns a tensor filled with random numbers from
    // a uniform distribution on the interval [0, 1)
    static Tensor_BK rand(const std::vector<int64_t>& shape);

    // Returns a tensor filled with random numbers from
    // a normal distribution with mean 0 and variance 1
    static Tensor_BK randn(const std::vector<int64_t>& shape);

    static Tensor_BK randint(int64_t low, int64_t high, const std::vector<int64_t>& shape);

private:
    ObjectPtr<TensorImpl> impl_;
};

// std::ostream& operator<<(std::ostream& os, const Tensor& t);

}// namespace aethermind


#endif// AETHERMIND_TENSOR_H
