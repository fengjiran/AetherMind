//
// Created by 赵丹 on 25-6-12.
//

#ifndef AETHERMIND_TENSOR_H
#define AETHERMIND_TENSOR_H

#include "tensor_impl.h"

namespace aethermind {

class Tensor {
public:
    Tensor() = default;

    explicit Tensor(const std::vector<int64_t>& shape,
                    int64_t storage_offset = 0,
                    DataType dtype = DataType::Float32(),
                    Device device = Device(kCPU));

    explicit Tensor(ObjectPtr<TensorImpl> impl);

    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;

    Tensor& operator=(const Tensor&) & = default;
    Tensor& operator=(Tensor&&) & noexcept = default;

    Tensor& operator=(const Tensor&) && = default;
    Tensor& operator=(Tensor&&) && noexcept = default;

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

    AM_NODISCARD TensorImpl* get_impl_ptr_unsafe() const noexcept;

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
    static Tensor rand(const std::vector<int64_t>& shape);

    // Returns a tensor filled with random numbers from
    // a normal distribution with mean 0 and variance 1
    static Tensor randn(const std::vector<int64_t>& shape);

    static Tensor randint(int64_t low, int64_t high, const std::vector<int64_t>& shape);

private:
    ObjectPtr<TensorImpl> impl_;
};

// std::ostream& operator<<(std::ostream& os, const Tensor& t);

}// namespace aethermind


#endif//AETHERMIND_TENSOR_H
