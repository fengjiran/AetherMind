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
    explicit SymbolicShape(const std::vector<std::optional<int64_t>>& shape);

    explicit SymbolicShape(std::vector<ShapeSymbol> shape) : symbolic_shape_(std::move(shape)) {}

    explicit SymbolicShape(IntArrayView shape);

    // Returns rank or nullopt in case of unranked shape.
    NODISCARD std::optional<size_t> rank() const;

    NODISCARD const std::optional<std::vector<ShapeSymbol>>& shape() const;

    ShapeSymbol operator[](size_t i) const;

    NODISCARD ShapeSymbol at(size_t i) const;

    NODISCARD std::optional<std::vector<bool>> GetSymbolicDims() const;

    // Checks whether the shape is fully static, i.e. rank and shape
    // of every dimension are known.
    NODISCARD bool IsComplete() const;

    void Dump() const;

    // Generate a new SymbolicShape through merging itself with another SymbolicShape.
    // Only dimensions that are both static and identical will be retained.
    // If either shape has an unknown rank, or if their ranks differ,
    // the resulting shape will be unranked.
    NODISCARD SymbolicShape Merge(const SymbolicShape& other) const;

    friend bool operator==(const SymbolicShape& lhs, const SymbolicShape& rhs) {
        return lhs.symbolic_shape_ == rhs.symbolic_shape_;
    }

    friend bool operator!=(const SymbolicShape& lhs, const SymbolicShape& rhs) {
        return !(lhs == rhs);
    }

private:
    std::optional<std::vector<ShapeSymbol>> symbolic_shape_{std::nullopt};
};

class Stride {
public:
    Stride() = default;

    Stride(const std::optional<size_t>& stride_idx,
           std::optional<bool> is_contiguous,
           const std::optional<size_t>& stride)
        : stride_idx_(stride_idx), stride_(stride), is_contiguous_(is_contiguous) {}

    NODISCARD std::optional<size_t> stride_idx() const {
        return stride_idx_;
    }

    NODISCARD std::optional<size_t> stride() const {
        return stride_;
    }

    NODISCARD std::optional<bool> is_contiguous() const {
        return is_contiguous_;
    }

    NODISCARD bool IsComplete() const {
        return stride_idx_ && stride_ && is_contiguous_;
    }

    bool operator==(const Stride& other) const {
        return stride_idx_ == other.stride_idx_ &&
               stride_ == other.stride_ &&
               is_contiguous_ == other.is_contiguous_;
    }

private:
    std::optional<size_t> stride_idx_;
    std::optional<size_t> stride_;
    std::optional<bool> is_contiguous_;
};

template<typename T>
class VaryingShape {
public:
    using ListOfOptionalElements = std::vector<std::optional<T>>;

    explicit VaryingShape(ListOfOptionalElements shape) : shape_(std::move(shape)) {}

    explicit VaryingShape(const std::vector<T>& vec)
        : VaryingShape(ListOfOptionalElements(vec.begin(), vec.end())) {}

    explicit VaryingShape(ArrayView<T> vec)
        : VaryingShape(ListOfOptionalElements(vec.begin(), vec.end())) {}

    explicit VaryingShape(std::optional<size_t> size = std::nullopt)
        : shape_(std::nullopt) {
        if (size.has_value()) {
            shape_ = ListOfOptionalElements(size.value());
        }
    }

    explicit VaryingShape(size_t size) : VaryingShape(std::optional(size)) {}

    const std::optional<T>& operator[](size_t i) const {
        if (!shape_.has_value()) {
            AETHERMIND_THROW(RuntimeError) << "Rank isn't fixed";
        }
        return shape_.value()[i];
    }

    NODISCARD std::optional<size_t> size() const {
        if (!shape_.has_value()) {
            return std::nullopt;
        }

        return shape_.value().size();
    }

    NODISCARD const std::optional<ListOfOptionalElements>& shape() const {
        return shape_;
    }

    bool operator==(const VaryingShape& other) const {
        return shape_ == other.shape_;
    }

    NODISCARD std::optional<std::vector<T>> GetConcreteValue() const;

    NODISCARD bool IsComplete() const;

    NODISCARD VaryingShape Merge(const VaryingShape& other) const;

private:
    std::optional<ListOfOptionalElements> shape_;
};

std::ostream& operator<<(std::ostream& os, const ShapeSymbol& s);
std::ostream& operator<<(std::ostream& os, const SymbolicShape& s);
std::ostream& operator<<(std::ostream& os, const Stride& s);
template<typename T>
std::ostream& operator<<(std::ostream& os, const VaryingShape<T>& t);

template<>
inline std::optional<Stride> MergePrimitiveValue(const std::optional<Stride>& a,
                                                 const std::optional<Stride>& b) {
    auto lhs = a;
    auto rhs = b;
    if (!lhs.has_value()) {
        lhs = Stride();
    }

    if (!rhs.has_value()) {
        rhs = Stride();
    }

    auto merged_idx = MergePrimitiveValue(lhs->stride_idx(), rhs->stride_idx());
    auto merged_contiguous = MergePrimitiveValue(lhs->is_contiguous(), rhs->is_contiguous());
    auto merged_stride = MergePrimitiveValue(lhs->stride(), rhs->stride());

    if (!(merged_idx.has_value() || merged_contiguous.has_value() || merged_stride.has_value())) {
        return std::optional<Stride>{};
    }

    return Stride(merged_idx, merged_contiguous, merged_stride);
}

inline ShapeSymbol MergePrimitiveValue(const ShapeSymbol& a, const ShapeSymbol& b) {
    if (a.IsStatic() && b.IsStatic() && a == b) {
        return a;
    }
    return ShapeSymbol::Create();
}

namespace details {

inline bool IsComplete(const Stride& s) {
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

    const std::optional<DataType>& dtype() const {
        return dtype_;
    }

    const std::optional<Device>& device() const {
        return device_;
    }

    VaryingShape<int64_t> shape() const;

    VaryingShape<int64_t> strides() const;

    std::optional<size_t> ndim() const {
        return shape().size();
    }

    std::optional<size_t> numel() const;

    const std::optional<bool>& undefined() const {
        return undefined_;
    }

    bool is_inferred() const {
        return is_inferred_;
    }

    bool requires_grad() const override {
        return requires_grad_ ? requires_grad_.value() : true;
    }

    static const TensorTypePtr& get();

    const SymbolicShape& GetSymbolicShape() const {
        return shape_;
    }

    const VaryingShape<Stride>& GetStrideProperties() const {
        return strides_;
    }

    const std::optional<bool>& RequiresGrad() const {
        return requires_grad_;
    }

    String ReprStr() const override {
        return is_inferred() ? str() + " (inferred)" : str();
    }

    bool Equals(const Type& rhs) const override;

    bool IsSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;

    // is all information about the type specified except for autograd?
    // This replaces the notion of a 'CompleteTensorType' that used to exist
    // in the type-hierarchy. Excluding require_grad and undefined allows
    // this to match the old behavior.
    bool IsComplete() const {
        return dtype() && device() && shape_.IsComplete() && strides_.IsComplete();
    }

    bool MatchTensor(const Tensor& t) const;

    TensorTypePtr Merge(const TensorType& other, bool merge_shape = true) const;

    TensorTypePtr Contiguity() const;

    static std::vector<int64_t> GetContiguousStrideOf(
            IntArrayView shape,
            MemoryFormat memory_format = MemoryFormat::kContiguous);

    static TensorTypePtr Create(std::optional<DataType> dtype,
                                std::optional<Device> device,
                                SymbolicShape shape,
                                VaryingShape<Stride> strides,
                                std::optional<bool> requires_grad,
                                std::optional<bool> undefined = false);

    static TensorTypePtr Create(std::optional<DataType> dtype,
                                std::optional<Device> device,
                                const VaryingShape<int64_t>& shape,
                                const VaryingShape<int64_t>& strides,
                                std::optional<bool> requires_grad,
                                std::optional<bool> undefined = false,
                                bool tensor_contiguity = false);

    static TensorTypePtr Create(std::optional<DataType> dtype,
                                std::optional<Device> device,
                                std::optional<size_t> dim,
                                std::optional<bool> requires_grad);

    static TensorTypePtr Create(const Tensor& t);

    static TensorTypePtr CreateContiguous(DataType dtype, Device device, IntArrayView shape);

    static TypePtr CreateFromBoolType();

    static TypePtr CreateFromNumberType(const Type& t);

    TensorTypePtr WithRequiresGrad(std::optional<bool> s) const;

    TensorTypePtr WithDataType(const std::optional<DataType>&) const;

    TensorTypePtr WithDim(std::optional<size_t>) const;

    TensorTypePtr WithShape(IntArrayView shape) const;

    TensorTypePtr WithStrides(VaryingShape<Stride>) const;

    TensorTypePtr WithDevice(std::optional<Device> device) const;

    TensorTypePtr WithSymbolicShape(SymbolicShape symbolic_shape) const;

    TensorTypePtr WithShapeAndStrides(IntArrayView shape, IntArrayView strides) const;

    TensorTypePtr WithUndefined() const;

    TensorTypePtr WithPossibleUndefined() const;

    TensorTypePtr WithDimensionOnly() const;

    static constexpr auto Kind = TypeKind::TensorType;

private:
    // constructor
    TensorType(std::optional<DataType> dtype,
               std::optional<Device> device,
               SymbolicShape shape,
               VaryingShape<Stride> strides,
               std::optional<bool> requires_grad,
               std::optional<bool> undefined = false);

    static VaryingShape<Stride> ComputeStrideProps(
            IntArrayView shape,
            IntArrayView strides,
            bool tensor_contiguity = false);

    TensorTypePtr Clone() const {
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
