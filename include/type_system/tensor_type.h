//
// Created by richard on 10/18/25.
//

#ifndef AETHERMIND_TYPE_SYSTEM_TENSOR_TYPE_H
#define AETHERMIND_TYPE_SYSTEM_TENSOR_TYPE_H

#include "memory_format.h"
#include "type_system/type.h"

namespace aethermind {

class Tensor;

// shape placeholder
class ShapeSymbol {
public:
    ShapeSymbol() : value_(-1) {}

    NODISCARD int64_t value() const {
        return value_;
    }

    // is this symbol a fixed/static dimension
    NODISCARD bool IsStatic() const {
        return value_ >= 0;
    }

    NODISCARD int64_t GetStaticValue() const {
        CHECK(IsStatic());
        return value_;
    }

    bool operator==(const ShapeSymbol& other) const {
        return value_ == other.value_;
    }

    bool operator<(const ShapeSymbol& other) const {
        return value_ < other.value_;
    }

    static ShapeSymbol CreateFromValue(int64_t val) {
        return ShapeSymbol(val);
    }

    static ShapeSymbol Create() {
        return CreateFromValue(-static_cast<int64_t>(++num_symbols_));
    }

private:
    explicit ShapeSymbol(int64_t val) : value_(val) {}

    int64_t value_;
    static std::atomic<size_t> num_symbols_;
};

// Shape of a Tensor represented with ShapeSymbol's. Unranked, ranked unknown
// dims, partially known and fully known shapes are all supported.
class SymbolicShape {
public:
    SymbolicShape() = default;

    // Known rank but unknown dimensions
    explicit SymbolicShape(std::optional<size_t> rank);

    // Mix of known and unknown ranks
    SymbolicShape(const std::vector<std::optional<int64_t>>& dims);//NOLINT

    SymbolicShape(std::vector<ShapeSymbol> dims) : dims_(std::move(dims)) {}//NOLINT

    SymbolicShape(IntArrayView dims);//NOLINT

    ShapeSymbol operator[](size_t i) const;

    NODISCARD ShapeSymbol at(size_t i) const;

    // Returns rank or nullopt in case of unranked shape.
    NODISCARD std::optional<size_t> rank() const;

    NODISCARD const std::optional<std::vector<ShapeSymbol>>& Shape() const;

    NODISCARD std::optional<std::vector<bool>> GetSymbolicDims() const;

    // Checks whether the shape is fully static, i.e. rank and shape
    // of every dimension are known.
    NODISCARD bool IsComplete() const;

    void Dump() const;

    // Create new SymbolicShape that is result of merging self and another
    // SymbolicShape. Only dimensions that are static and equal will be
    // preserved.
    // If either of two shapes are of unknown rank or they have unmatching rank,
    // result will be unranked.
    NODISCARD SymbolicShape Merge(const SymbolicShape& other) const;

    friend bool operator==(const SymbolicShape& lhs, const SymbolicShape& rhs) {
        return lhs.dims_ == rhs.dims_;
    }

    friend bool operator!=(const SymbolicShape& lhs, const SymbolicShape& rhs) {
        return !(lhs == rhs);
    }

private:
    std::optional<std::vector<ShapeSymbol>> dims_{std::nullopt};
};

class Stride {
public:
    Stride() = default;

    Stride(const std::optional<size_t>& stride_idx,
           std::optional<bool> contiguous,
           const std::optional<size_t>& stride)
        : stride_idx_(stride_idx), contiguous_(contiguous), stride_(stride) {}

    NODISCARD bool IsComplete() const {
        return stride_idx_ && contiguous_ && stride_;
    }

    bool operator==(const Stride& other) const {
        return stride_idx_ == other.stride_idx_ &&
               contiguous_ == other.contiguous_ &&
               stride_ == other.stride_;
    }

    // private:
    std::optional<size_t> stride_idx_;
    std::optional<bool> contiguous_;
    std::optional<size_t> stride_;
};

template<typename T>
class VaryingShape {
public:
    using ListOfOptionalElements = std::vector<std::optional<T>>;

    explicit VaryingShape(ListOfOptionalElements dims) : dims_(std::move(dims)) {}

    explicit VaryingShape(const std::vector<T>& vec)
        : VaryingShape(ListOfOptionalElements(vec.begin(), vec.end())) {}

    explicit VaryingShape(ArrayView<T> vec)
        : VaryingShape(ListOfOptionalElements(vec.begin(), vec.end())) {}

    explicit VaryingShape(std::optional<size_t> size = std::nullopt)
        : dims_(std::nullopt) {
        if (size.has_value()) {
            dims_ = ListOfOptionalElements(size.value());
        }
    }

    explicit VaryingShape(size_t size) : VaryingShape(std::optional<size_t>(size)) {}

    const std::optional<T>& operator[](size_t i) const {
        if (!dims_.has_value()) {
            AETHERMIND_THROW(RuntimeError) << "Rank isn't fixed";
        }
        return dims_.value()[i];
    }

    NODISCARD std::optional<size_t> size() const {
        if (!dims_.has_value()) {
            return std::nullopt;
        }

        return dims_.value().size();
    }

    NODISCARD const std::optional<ListOfOptionalElements>& shape() const {
        return dims_;
    }

    bool operator==(const VaryingShape& other) const {
        return dims_ == other.dims_;
    }

    NODISCARD std::optional<std::vector<T>> get_concrete_value() const;

    NODISCARD bool IsComplete() const;

    NODISCARD VaryingShape merge(const VaryingShape& other) const;

private:
    std::optional<ListOfOptionalElements> dims_;
};

std::ostream& operator<<(std::ostream& os, const ShapeSymbol& s);
std::ostream& operator<<(std::ostream& os, const SymbolicShape& s);
std::ostream& operator<<(std::ostream& os, const Stride& s);
template<typename T>
std::ostream& operator<<(std::ostream& os, const VaryingShape<T>& t);

template<>
inline std::optional<Stride> merge_primitive(const std::optional<Stride>& a,
                                             const std::optional<Stride>& b) {
    auto lhs = a;
    auto rhs = b;
    if (!lhs.has_value()) {
        lhs = Stride();
    }

    if (!rhs.has_value()) {
        rhs = Stride();
    }

    auto merged_idx = merge_primitive(lhs->stride_idx_, rhs->stride_idx_);
    auto merged_contiguous = merge_primitive(lhs->contiguous_, rhs->contiguous_);
    auto merged_stride = merge_primitive(lhs->stride_, rhs->stride_);

    if (!(merged_idx.has_value() || merged_contiguous.has_value() || merged_stride.has_value())) {
        return std::optional<Stride>{};
    }

    return Stride(merged_idx, merged_contiguous, merged_stride);
}

inline ShapeSymbol merge_primitive(const ShapeSymbol& a, const ShapeSymbol& b) {
    if (a.IsStatic() && b.IsStatic() && a == b) {
        return a;
    }
    return ShapeSymbol::Create();
}

namespace details {

inline bool is_complete(const Stride& s) {
    return s.IsComplete();
}

}// namespace details

class TensorType;
using TensorTypePtr = std::shared_ptr<TensorType>;
class TensorType : public SharedType {
public:
    String str() const override {
        return "Tensor";
    }

    String repr_str() const override {
        if (is_inferred_type()) {
            return str() + " (inferred)";
        }
        return str();
    }

    bool equals(const Type& rhs) const override;

    const std::optional<DataType>& data_type() const {
        return dtype_;
    }

    const std::optional<Device>& device() const {
        return device_;
    }

    VaryingShape<int64_t> shape() const;

    VaryingShape<int64_t> strides() const;

    std::optional<size_t> dim() const {
        return shape().size();
    }

    std::optional<size_t> numel() const;

    const VaryingShape<Stride>& stride_properties() const {
        return strides_;
    }

    const std::optional<bool>& requiresGrad() const {
        return requires_grad_;
    }

    bool requires_grad() const override {
        return requires_grad_ ? *requires_grad_ : true;
    }

    const std::optional<bool>& undefined() const {
        return undefined_;
    }

    bool is_inferred_type() const {
        return is_inferred_;
    }

    // is all information about the type specified except for autograd?
    // This replaces the notion of a 'CompleteTensorType' that used to exist
    // in the type-hierarchy. Excluding require_grad and undefined allows
    // this to match the old behavior.
    bool is_complete() const {
        return data_type() && device() && shape_.IsComplete() && strides_.IsComplete();
    }

    TensorTypePtr contiguous() const;

    static std::vector<int64_t> contiguous_stride_of(IntArrayView shape,
                                                     MemoryFormat memory_format = MemoryFormat::Contiguous);

    static TensorTypePtr create(const Tensor& t);

    static TensorTypePtr create(
            std::optional<DataType> dtype,
            std::optional<Device> device,
            const VaryingShape<int64_t>& shape,
            const VaryingShape<int64_t>& strides,
            std::optional<bool> requires_grad,
            std::optional<bool> undefined = false,
            bool tensor_contiguity = false);

    static TensorTypePtr create(
            std::optional<DataType> dtype,
            std::optional<Device> device,
            std::optional<size_t> dim,
            std::optional<bool> requires_grad);

    static TensorTypePtr create(
            std::optional<DataType> dtype,
            std::optional<Device> device,
            SymbolicShape shape,
            VaryingShape<Stride> strides,
            std::optional<bool> requires_grad,
            std::optional<bool> undefined = false);

    static TensorTypePtr create_contiguous(DataType dtype, Device device, IntArrayView shape);

    TensorTypePtr with_requires_grad(std::optional<bool> s) const;

    TensorTypePtr with_data_type(const std::optional<DataType>&) const;

    TensorTypePtr with_dim(std::optional<size_t>) const;

    TensorTypePtr with_shape(IntArrayView shape) const;

    TensorTypePtr with_strides(VaryingShape<Stride>) const;

    TensorTypePtr with_device(std::optional<Device> device) const;

    TensorTypePtr with_symbolic_shape(SymbolicShape symbolic_shape) const;

    TensorTypePtr with_shape_and_strides(IntArrayView shape, IntArrayView strides) const;

    TensorTypePtr with_undefined() const;

    TensorTypePtr with_possibly_undefined() const;

    static const TensorTypePtr& get();

    static constexpr auto Kind = TypeKind::TensorType;

private:
    // constructor
    TensorType(std::optional<DataType> dtype,
               std::optional<Device> device,
               SymbolicShape shape,
               VaryingShape<Stride> strides,
               std::optional<bool> requires_grad,
               std::optional<bool> undefined = false);

    static VaryingShape<Stride> compute_stride_props(
            IntArrayView shape,
            IntArrayView strides,
            bool tensor_contiguity = false);

    TensorTypePtr clone() const {
        auto ptr = new TensorType(dtype_, device_, shape_, strides_, requires_grad_, undefined_);
        return TensorTypePtr(ptr);
    }

    std::optional<DataType> dtype_;
    std::optional<Device> device_;
    SymbolicShape shape_;
    VaryingShape<Stride> strides_;
    std::optional<bool> requires_grad_;
    std::optional<bool> undefined_;
    // Whether this type was inferred.
    bool is_inferred_ = false;
};


}// namespace aethermind

#endif//AETHERMIND_TYPE_SYSTEM_TENSOR_TYPE_H
