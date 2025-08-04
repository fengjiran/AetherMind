//
// Created by 赵丹 on 25-6-12.
//

#ifndef AETHERMIND_TENSOR_H
#define AETHERMIND_TENSOR_H

#include "tensor_impl.h"

namespace aethermind {

class Tensor {
public:
    Tensor();

    explicit Tensor(const std::vector<int64_t>& shape,
                    int64_t storage_offset = 0,
                    DataType dtype = DataType::Float32(),
                    DeviceType device = DeviceType::kCPU);

    explicit Tensor(std::shared_ptr<TensorImpl> impl);

    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;

    Tensor& operator=(const Tensor&) & = default;
    Tensor& operator=(Tensor&&) & noexcept = default;

    Tensor& operator=(const Tensor&) && = default;
    Tensor& operator=(Tensor&&) && noexcept = default;

    NODISCARD bool defined() const;

    NODISCARD int32_t use_count() const;

    NODISCARD bool unique() const;

    NODISCARD std::vector<int64_t> shape() const;

    NODISCARD std::vector<int64_t> strides() const;

    NODISCARD int64_t shape(int64_t dim) const;

    NODISCARD int64_t strides(int64_t dim) const;

    NODISCARD DataType dtype() const;

    NODISCARD DeviceType device() const;

    NODISCARD int32_t ndim() const;

    NODISCARD int64_t numel() const;

    NODISCARD size_t itemsize() const;

    NODISCARD size_t nbytes() const;

    NODISCARD bool has_storage() const;

    NODISCARD int64_t storage_offset() const;

    NODISCARD bool is_contiguous() const;

    NODISCARD bool is_cpu() const;

    NODISCARD bool is_cuda() const;

    // NODISCARD Scalar item() const;

    NODISCARD void* data_ptr() const;

    NODISCARD const void* const_data_ptr() const;

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
    std::shared_ptr<TensorImpl> impl_;
};

std::ostream& operator<<(std::ostream& os, const Tensor& t);

}// namespace aethermind


#endif//AETHERMIND_TENSOR_H
