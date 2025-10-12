//
// Created by richard on 10/2/25.
//
#include "tensor.h"
#include "type.h"

#include <utility>

namespace aethermind {

std::atomic<size_t> ShapeSymbol::num_symbols_ = 1;

std::ostream& operator<<(std::ostream& os, const Stride& s) {
    os << "{";
    if (s.stride_idx_.has_value()) {
        os << *s.stride_idx_;
    } else {
        os << "*";
    }

    os << ":";
    if (s.stride_.has_value()) {
        os << *s.stride_;
    } else {
        os << "*";
    }
    os << '}';
    return os;
}

std::ostream& operator<<(std::ostream& os, const ShapeSymbol& s) {
    if (s.is_static()) {
        os << s.value();
    } else {
        os << "SS(" << s.value() << ')';
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const SymbolicShape& s) {
    auto rank_opt = s.rank();
    if (!rank_opt.has_value()) {
        os << "(*)";
        return os;
    }
    auto size_opt = s.sizes();
    os << "(";
    for (size_t i = 0; i < rank_opt.value(); ++i) {
        if (i > 0) {
            os << ", ";
        }

        if (size_opt.has_value() && size_opt.value()[i].is_static()) {
            os << size_opt.value()[i];
        } else {
            os << "*";
        }
    }
    os << ")";
    return os;
}

SymbolicShape::SymbolicShape(std::optional<size_t> rank) {
    if (rank.has_value()) {
        std::vector<ShapeSymbol> shape_symbols(rank.value());
        for (size_t i = 0; i < rank.value(); ++i) {
            shape_symbols[i] = ShapeSymbol::Create();
        }
        dims_ = shape_symbols;
    }
}

SymbolicShape::SymbolicShape(const std::vector<std::optional<int64_t>>& dims) {
    std::vector<ShapeSymbol> shape_symbols(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i].has_value()) {
            shape_symbols[i] = ShapeSymbol::CreateFromStaticSize(dims[i].value());
        } else {
            shape_symbols[i] = ShapeSymbol::Create();
        }
    }
    dims_ = shape_symbols;
}

SymbolicShape::SymbolicShape(IntArrayView dims) {
    std::vector<ShapeSymbol> shape_symbols(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape_symbols[i] = ShapeSymbol::CreateFromStaticSize(dims[i]);
    }
    dims_ = shape_symbols;
}

ShapeSymbol SymbolicShape::operator[](size_t i) const {
    if (!dims_.has_value()) {
        AETHERMIND_THROW(RuntimeError) << "Rank isn't fixed";
    }
    return dims_.value()[i];
}

ShapeSymbol SymbolicShape::at(size_t i) const {
    if (!dims_.has_value()) {
        AETHERMIND_THROW(RuntimeError) << "Rank isn't fixed";
    }

    if (i >= dims_->size()) {
        AETHERMIND_THROW(OutOfRangeError) << "Out of range";
    }
    return dims_.value()[i];
}

std::optional<size_t> SymbolicShape::rank() const {
    if (dims_.has_value()) {
        return dims_->size();
    }
    return std::nullopt;
}

const std::optional<std::vector<ShapeSymbol>>& SymbolicShape::sizes() const {
    return dims_;
}

std::optional<std::vector<bool>> SymbolicShape::symbolic_dims() const {
    if (!dims_.has_value()) {
        return std::nullopt;
    }

    std::vector<bool> res(rank().value());
    for (size_t i = 0; i < rank().value(); ++i) {
        res[i] = !dims_.value()[i].is_static();
    }
    return res;
}

bool SymbolicShape::is_complete() const {
    if (!dims_.has_value()) {
        return false;
    }

    for (auto d: dims_.value()) {
        if (!d.is_static()) {
            return false;
        }
    }
    return true;
}

void SymbolicShape::dump() const {
    std::cout << *this << std::endl;
}

SymbolicShape SymbolicShape::merge(const SymbolicShape& other) const {
    if (!dims_.has_value() || !other.dims_.has_value() || dims_->size() != other.dims_->size()) {
        return {};
    }

    auto n = dims_->size();
    std::vector<ShapeSymbol> dims(n);
    for (size_t i = 0; i < n; ++i) {
        dims[i] = merge_primitive(dims_.value()[i], other.dims_.value()[i]);
    }
    return {std::move(dims)};
}

template<typename T>
std::optional<std::vector<T>> VaryingShape<T>::concrete_sizes() const {
    if (!dims_.has_value()) {
        return std::nullopt;
    }

    auto n = dims_->size();
    auto shape = dims_.value();
    std::vector<T> res(n);
    for (size_t i = 0; i < n; i++) {
        if (!shape[i].has_value()) {
            return std::nullopt;
        }
        res[i] = shape[i].value();
    }
    return res;
}

template<typename T>
bool VaryingShape<T>::is_complete() const {
    if (!dims_.has_value()) {
        return false;
    }

    for (auto d: dims_.value()) {
        if (!d.has_value() || details::is_complete(d.value())) {
            return false;
        }
    }
    return true;
}

template<typename T>
VaryingShape<T> VaryingShape<T>::merge(const VaryingShape& other) const {
    if (!dims_.has_value() || !other.dims_.has_value() || dims_->size() != other.dims_->size()) {
        return {};
    }

    auto n = dims_->size();
    ListOfOptionalElements dims(n);
    for (size_t i = 0; i < n; ++i) {
        dims[i] = merge_primitive(dims_.value()[i], other.dims_.value()[i]);
    }

    return {std::move(dims)};
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const VaryingShape<T>& t) {
    const auto& sizes_opt = t.sizes();
    if (!sizes_opt.has_value()) {
        os << "(*)";
        return os;
    }

    auto n = sizes_opt->size();
    os << "(";
    for (size_t i = 0; i < n; ++i) {
        if (i > 0) {
            os << ", ";
        }
        const auto& v = t[i];
        if (v.has_value()) {
            os << v.value();
        } else {
            os << "*";
        }
    }
    os << ")";
    return os;
}

template std::ostream& operator<<(std::ostream& os, const VaryingShape<int64_t>&);
template std::ostream& operator<<(std::ostream& os, const VaryingShape<Stride>&);

template struct VaryingShape<bool>;
template struct VaryingShape<size_t>;
template struct VaryingShape<int64_t>;
template struct VaryingShape<ShapeSymbol>;
template struct VaryingShape<Stride>;

TensorType::TensorType(std::optional<DataType> dtype,
                       std::optional<Device> device,
                       SymbolicShape shape,
                       VaryingShape<Stride> strides,
                       std::optional<bool> requires_grad,
                       std::optional<bool> undefined)
    : SharedType(Kind), dtype_(std::move(dtype)), device_(device),
      shape_(std::move(shape)), strides_(std::move(strides)),
      requires_grad_(requires_grad), undefined_(undefined) {}

VaryingShape<int64_t> TensorType::shape() const {
    if (!shape_.rank().has_value()) {
        return {};
    }

    auto n = shape_.rank().value();
    std::vector<std::optional<int64_t>> dims;
    dims.reserve(n);
    for (const auto& ss: shape_.sizes().value()) {
        dims.push_back(ss.is_static() ? std::optional<int64_t>(ss.static_size())
                                      : std::nullopt);
    }
    return {std::move(dims)};
}

VaryingShape<int64_t> TensorType::strides() const {
    const auto& sizes = strides_.sizes();
    if (!sizes.has_value()) {
        return {};
    }

    auto n = sizes->size();
    std::vector<std::optional<int64_t>> dims(n);
    for (const auto& stride: sizes.value()) {
        if (!stride.has_value()) {
            continue;
        }
        const auto& s = stride.value();
        if (s.stride_idx_.has_value() && s.stride_.has_value()) {
            dims[*s.stride_idx_] = *s.stride_;
        }
    }
    return {std::move(dims)};
}


bool TensorType::equals(const Type& rhs) const {
    if (rhs.kind() != kind()) {
        return false;
    }

    auto t = rhs.expect<TensorType>();
    return data_type() == t->data_type() &&
           shape() == t->shape() &&
           stride_properties() == t->stride_properties() &&
           device() == t->device() &&
           requiresGrad() == t->requiresGrad() &&
           undefined() == t->undefined();
}

VaryingShape<Stride> TensorType::compute_stride_props(
        IntArrayView shape, IntArrayView strides, bool tensor_contiguity) {
    int n_dim = static_cast<int>(shape.size());

}


TensorTypePtr TensorType::create(std::optional<DataType> dtype,
                                 std::optional<Device> device,
                                 SymbolicShape shape,
                                 VaryingShape<Stride> strides,
                                 std::optional<bool> requires_grad,
                                 std::optional<bool> undefined) {
    //NOLINTBEGIN
    auto ptr = new TensorType(dtype, device, std::move(shape),
                              std::move(strides), requires_grad,
                              undefined);
    //NOLINTEND
    return TensorTypePtr(ptr);
}

TensorTypePtr TensorType::create(std::optional<DataType> dtype,
                                 std::optional<Device> device,
                                 const VaryingShape<int64_t>& shape,
                                 const VaryingShape<int64_t>& strides,
                                 std::optional<bool> requires_grad,
                                 std::optional<bool> undefined,
                                 bool tensor_contiguity) {
    auto concrete_stride_size = strides.concrete_sizes();
    if (concrete_stride_size.has_value()) {
        //
    }
}


TensorTypePtr TensorType::create(const Tensor& t) {
    VaryingShape<bool> contiguity;
    VaryingShape<size_t> stride_indices;
    VaryingShape<int64_t> strides;
    VaryingShape<int64_t> shape;

    // return create(t.dtype(),t.device(),SymbolicShape(), VaryingShape<Stride>{},);
}


}// namespace aethermind