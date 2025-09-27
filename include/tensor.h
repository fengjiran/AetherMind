//
// Created by 赵丹 on 25-6-12.
//

#ifndef AETHERMIND_TENSOR_H
#define AETHERMIND_TENSOR_H

#include "tensor_impl.h"
#include "type_traits.h"

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

    NODISCARD bool defined() const;

    NODISCARD uint32_t use_count() const;

    NODISCARD bool unique() const;

    NODISCARD IntArrayView shape() const;

    NODISCARD IntArrayView strides() const;

    NODISCARD int64_t shape(int64_t dim) const;

    NODISCARD int64_t strides(int64_t dim) const;

    NODISCARD DataType dtype() const;

    NODISCARD Device device() const;

    NODISCARD int32_t ndim() const;

    NODISCARD int64_t numel() const;

    NODISCARD size_t itemsize() const;

    NODISCARD size_t nbytes() const;

    NODISCARD bool has_storage() const;

    NODISCARD int64_t storage_offset() const;

    NODISCARD bool is_contiguous() const;

    NODISCARD bool is_cpu() const;

    NODISCARD bool is_cuda() const;

    NODISCARD TensorImpl* get_impl_ptr_unsafe() const noexcept;

    NODISCARD TensorImpl* release_impl_unsafe();

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
    ObjectPtr<TensorImpl> impl_;
};

// Tensor type
template<>
struct TypeTraits<Tensor> : TypeTraitsBase {
    static void CopyToAny(const Tensor& src, AetherMindAny* dst) {
        dst->tag_ = AnyTag::Tensor;
        Object* obj = src.get_impl_ptr_unsafe();
        dst->payload_ = obj;
        if (!IsNullTypePtr(obj)) {
            details::ObjectUnsafe::IncRef(obj);
        }
    }

    static void MoveToAny(Tensor src, AetherMindAny* dst) {
        dst->tag_ = AnyTag::Tensor;
        dst->payload_ = static_cast<Object*>(src.release_impl_unsafe());
    }

    static Tensor CopyFromAnyAfterCheck(const AetherMindAny* src) {
        auto* obj = std::get<Object*>(src->payload_);
        if (!IsNullTypePtr(obj)) {
            details::ObjectUnsafe::IncRef(obj);
        }

        return Tensor(ObjectPtr<TensorImpl>::reclaim(static_cast<TensorImpl*>(obj)));
    }

    static Tensor MoveFromAnyAfterCheck(AetherMindAny* src) {
        auto* obj = std::get<Object*>(src->payload_);
        src->payload_ = static_cast<Object*>(nullptr);
        src->tag_ = AnyTag::None;
        return Tensor(ObjectPtr<TensorImpl>::reclaim(static_cast<TensorImpl*>(obj)));
    }

    static std::optional<Tensor> TryCastFromAny(const AetherMindAny* src) {
        if (check(src)) {
            return CopyFromAnyAfterCheck(src);
        }
        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == AnyTag::Tensor;
    }

    static std::string TypeStr() {
        return AnyTagToString(AnyTag::Tensor);
    }
};


std::ostream& operator<<(std::ostream& os, const Tensor& t);

}// namespace aethermind


#endif//AETHERMIND_TENSOR_H
