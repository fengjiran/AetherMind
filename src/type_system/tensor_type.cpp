//
// Created by richard on 10/2/25.
//
#include "type_system/tensor_type.h"
#include "tensor.h"

#include <numeric>
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
    if (s.IsStatic()) {
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
    auto size_opt = s.Shape();
    os << "(";
    for (size_t i = 0; i < rank_opt.value(); ++i) {
        if (i > 0) {
            os << ", ";
        }

        if (size_opt.has_value() && size_opt.value()[i].IsStatic()) {
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
            shape_symbols[i] = ShapeSymbol::CreateFromValue(dims[i].value());
        } else {
            shape_symbols[i] = ShapeSymbol::Create();
        }
    }
    dims_ = shape_symbols;
}

SymbolicShape::SymbolicShape(IntArrayView dims) {
    std::vector<ShapeSymbol> shape_symbols(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape_symbols[i] = ShapeSymbol::CreateFromValue(dims[i]);
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

const std::optional<std::vector<ShapeSymbol>>& SymbolicShape::Shape() const {
    return dims_;
}

std::optional<std::vector<bool>> SymbolicShape::GetSymbolicDims() const {
    const auto rank_opt = rank();
    if (!rank_opt.has_value()) {
        return std::nullopt;
    }

    const auto n = rank_opt.value();
    std::vector<bool> symbolic_dims;
    symbolic_dims.reserve(n);
    for (const auto& s: dims_.value()) {
        symbolic_dims.push_back(!s.IsStatic());
    }
    return symbolic_dims;
}

bool SymbolicShape::IsComplete() const {
    if (!dims_.has_value()) {
        return false;
    }

    for (auto d: dims_.value()) {
        if (!d.IsStatic()) {
            return false;
        }
    }
    return true;
}

void SymbolicShape::Dump() const {
    std::cout << *this << std::endl;
}

SymbolicShape SymbolicShape::Merge(const SymbolicShape& other) const {
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
std::optional<std::vector<T>> VaryingShape<T>::get_concrete_value() const {
    if (!dims_.has_value()) {
        return std::nullopt;
    }

    std::vector<T> shape;
    shape.reserve(dims_->size());
    for (auto d: dims_.value()) {
        if (!d.has_value()) {
            return std::nullopt;
        }
        shape.push_back(d.value());
    }

    return shape;
}

template<typename T>
bool VaryingShape<T>::IsComplete() const {
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
        return VaryingShape<T>{};
    }

    auto n = dims_->size();
    ListOfOptionalElements dims(n);
    for (size_t i = 0; i < n; ++i) {
        dims[i] = merge_primitive(dims_.value()[i], other.dims_.value()[i]);
    }

    return VaryingShape<T>{std::move(dims)};
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const VaryingShape<T>& t) {
    const auto& sizes_opt = t.shape();
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
    auto rank = shape_.rank();
    if (!rank.has_value()) {
        return VaryingShape<int64_t>{};
    }

    auto n = rank.value();
    std::vector<std::optional<int64_t>> dims;
    dims.reserve(n);
    for (const auto& ss: shape_.Shape().value()) {
        dims.push_back(ss.IsStatic() ? std::optional<int64_t>(ss.GetStaticValue())
                                     : std::nullopt);
    }
    return VaryingShape<int64_t>{std::move(dims)};
}

const SymbolicShape& TensorType::symbolic_shape() const {
    return shape_;
}

VaryingShape<int64_t> TensorType::strides() const {
    const auto& shape = strides_.shape();
    if (!shape.has_value()) {
        return VaryingShape<int64_t>{};
    }

    auto n = shape->size();
    std::vector<std::optional<int64_t>> dims(n);
    for (const auto& stride: shape.value()) {
        if (!stride.has_value()) {
            continue;
        }
        const auto& s = stride.value();
        if (s.stride_idx_.has_value() && s.stride_.has_value()) {
            dims[s.stride_idx_.value()] = s.stride_.value();
        }
    }
    return VaryingShape<int64_t>{std::move(dims)};
}

std::optional<size_t> TensorType::numel() const {
    size_t prod = 1;
    for (size_t i = 0; i < shape().size(); ++i) {
        auto s = shape()[i];
        if (!s.has_value()) {
            return std::optional<size_t>{};
        }
        prod *= s.value();
    }
    return prod;
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

bool TensorType::isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const {
    if (auto ptr = rhs.cast<TensorType>()) {
        if (this == ptr.get()) {
            return true;
        }
        return *merge(*ptr) == *ptr;
    }
    return Type::isSubtypeOfExt(rhs, why_not);
}

// The idea is to only mark possible overlap across dimensions. We want to
// return false for expanded tensors and permuted tensors, for which dimensional
// collapsing is safe.
bool possible_cross_dimension_overlap(IntArrayView shape, IntArrayView strides) {
    int ndim = static_cast<int>(shape.size());
    std::vector<size_t> stride_indices(ndim);// stride ascend order index
    std::iota(stride_indices.rbegin(), stride_indices.rend(), 0);

    // sort indices going with ascending strides
    for (int i = 1; i < ndim; ++i) {
        int c = i;
        for (int j = i - 1; j >= 0; --j) {
            if (strides[stride_indices[j]] > strides[stride_indices[c]]) {
                std::swap(stride_indices[j], stride_indices[c]);
                c = j;
            }
        }
    }

    for (int i = 1; i < ndim; ++i) {
        if (i != 0) {
            // we are being conservative on checking for memory overlap
            if (shape[stride_indices[i]] != 1 &&
                strides[stride_indices[i]] < shape[stride_indices[i - 1]] * strides[stride_indices[i - 1]]) {
                return true;
            }
        }
    }
    return false;
}

VaryingShape<Stride> TensorType::compute_stride_props(IntArrayView shape,
                                                      IntArrayView strides,
                                                      bool tensor_contiguity) {
    int ndim = static_cast<int>(shape.size());
    std::vector<size_t> stride_indices(ndim);

    // default has_overlap to false as we only compute overlap when:
    // 1. input shape/strides fails format check;
    // 2. tensor_contiguity are not set.
    bool has_overlap = false;

    // Sorting strides in ascending order
    // Example:
    //  Prior to sorting
    //  Idx:     [0,   1,  2,  3]
    //  shape:   [8,   1, 10, 16]
    //  Strides: [160, 1, 16,  1]
    //
    //  After sorting
    //  Idx:     [1,  3,  2,   0]
    //  shape:   [1, 16, 10,   8]
    //  Strides: [1,  1, 16, 160]
    //
    if (is_channels_last_strides_2d(shape, strides) || is_channels_last_strides_3d(shape, strides)) {
        std::iota(stride_indices.rbegin() + 1, stride_indices.rend() - 1, 2);
        stride_indices[0] = 1;
        stride_indices[ndim - 1] = 0;
    } else if (is_contiguous_stride(shape, strides)) {
        std::iota(stride_indices.rbegin(), stride_indices.rend(), 0);
    } else {
        // For broadcasted dimension where stride is 0, we have to stick to
        // TensorIterator behavior in eager, where they introduce an ambiguous
        // comparison result to preserve permutation by best effort.
        // For more details, see NOTE: [Computing output strides]
        std::iota(stride_indices.rbegin(), stride_indices.rend(), 0);
        auto should_swap = [&](size_t i, size_t j) {
            if (strides[i] == 0 || strides[j] == 0) {
                return 0;
            }

            if (strides[i] < strides[j]) {
                return -1;
            }

            if (strides[i] > strides[j]) {
                return 1;
            }

            // strides[i] == strides[j]
            if (shape[i] > shape[j]) {
                return 1;
            }

            return 0;
        };

        for (int i = 1; i < ndim; ++i) {
            int dim1 = i;
            for (int dim0 = i - 1; dim0 >= 0; --dim0) {
                int comp = should_swap(stride_indices[dim0], stride_indices[dim1]);
                if (comp > 0) {
                    std::swap(stride_indices[dim0], stride_indices[dim1]);
                    dim1 = dim0;
                } else if (comp < 0) {
                    break;
                }
            }
        }

        // conveniently is_contiguous_strides/is_contiguous_strides only returns
        // true when there's no memory overlap, so we only re-compute has_overlap
        // in the last branch when both returns false
        if (!tensor_contiguity) {
            // trust tensor_contiguity and only computes overlap when it is not set
            has_overlap = possible_cross_dimension_overlap(shape, strides);
        }
    }

    std::vector<Stride> stride_properties;
    stride_properties.reserve(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        auto cur_idx = stride_indices[i];
        bool contiguous = tensor_contiguity;
        if (!contiguous) {// the contiguity is not set, compute contiguity by shape and stride
            if (!has_overlap) {
                if (i == 0) {
                    contiguous = strides[cur_idx] == 1;
                } else {
                    auto pre_idx = stride_indices[i - 1];
                    contiguous = strides[cur_idx] == 1 ||
                                 (strides[cur_idx] != 0 && strides[cur_idx] == strides[pre_idx] * shape[pre_idx]);
                }
            } else {
                contiguous = false;
            }
        }
        stride_properties.emplace_back(cur_idx, contiguous, strides[cur_idx]);
    }
    return VaryingShape<Stride>{stride_properties};
}

template<typename T>
static bool is_null_or_equal(std::optional<T> a, IntArrayView b) {
    return !a.has_value() || a.value() == b;
}

bool TensorType::matchTensor(const Tensor& t) const {
    bool undef = undefined().value_or(!t.defined());

    // When the followings are true, we consider it's not a match:
    // - undefined().has_value() == true
    // - undefined().value() != !t.defined()
    if (undef != !t.defined()) {
        return false;
    }

    // When the followings are true, we consider it's a match:
    // - t is not defined
    // - undefined() == null or undefined().value() == true
    if (!t.defined()) {
        return true;
    }

    // TODO
    bool rg = t.requires_grad();
    bool matched_strides = !stride_properties().size() ||
                           (!t.has_storage() && !stride_properties().IsComplete()) ||
                           stride_properties() == compute_stride_props(t.shape(), t.strides(), t.is_contiguous());
    return data_type().value_or(t.dtype()) == t.dtype() &&
           device().value_or(t.device()) == t.device() &&
           requiresGrad().value_or(rg) == rg &&
           matched_strides &&
           is_null_or_equal(shape().get_concrete_value(), t.shape());
}


TensorTypePtr TensorType::merge(const TensorType& other, bool merge_shape) const {
    auto dtype = merge_primitive(data_type(), other.data_type());
    auto shape = merge_shape ? symbolic_shape().Merge(other.symbolic_shape()) : symbolic_shape();
    auto sprops = stride_properties().merge(other.stride_properties());
    auto dev = merge_primitive(device(), other.device());
    auto gr = merge_primitive(requiresGrad(), other.requiresGrad());
    auto undef = merge_primitive(undefined(), other.undefined());
    return create(dtype, dev, shape, sprops, gr, undef);
}

TensorTypePtr TensorType::contiguous() const {
    auto cloned = clone();
    auto concrete_shape = shape().get_concrete_value();
    CHECK(concrete_shape.has_value());
    auto strides = compute_stride_props(concrete_shape.value(),
                                        contiguous_stride_of(concrete_shape.value()));
    cloned->strides_ = strides;
    return cloned;
}

std::vector<int64_t> TensorType::contiguous_stride_of(IntArrayView shape, MemoryFormat memory_format) {
    if (shape.empty()) {
        return {};
    }

    auto ndim = shape.size();
    std::vector<int64_t> stride_ascend_order(ndim);
    if (memory_format == MemoryFormat::ChannelsLast) {
        stride_ascend_order = {1, 3, 2, 0};
    } else if (memory_format == MemoryFormat::ChannelsLast3d) {
        stride_ascend_order = {1, 4, 3, 2, 0};
    } else {
        for (size_t i = 0; i < ndim; ++i) {
            stride_ascend_order[i] = static_cast<int64_t>(ndim - i - 1);
        }
    }

    std::vector<int64_t> strides(ndim);
    strides[stride_ascend_order[0]] = 1;
    for (int i = 1; i < ndim; ++i) {
        auto pre_idx = stride_ascend_order[i - 1];
        auto cur_idx = stride_ascend_order[i];
        strides[cur_idx] = strides[pre_idx] * shape[pre_idx];
    }
    return strides;
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
    auto concrete_stride = strides.get_concrete_value();
    if (concrete_stride.has_value()) {
        auto concrete_shape = shape.get_concrete_value();
        CHECK(concrete_shape.has_value() && concrete_shape->size() == concrete_stride->size());
        auto sprops = compute_stride_props(concrete_shape.value(),
                                           concrete_stride.value(),
                                           tensor_contiguity);
        auto symbol_shape = SymbolicShape(concrete_shape.value());
        return create(dtype, device, std::move(symbol_shape), std::move(sprops), requires_grad, undefined);
    }

    // strides are all null, but still have number of strides equal to number of ranks
    const auto& shape_opt = shape.shape();
    CHECK(shape_opt.has_value() && shape.size().has_value());
    auto symbol_shape = SymbolicShape(shape_opt.value());
    return create(dtype, device, std::move(symbol_shape),
                  VaryingShape<Stride>(shape_opt->size()), requires_grad, undefined);
}

TensorTypePtr TensorType::create(std::optional<DataType> dtype,
                                 std::optional<Device> device,
                                 std::optional<size_t> dim,
                                 std::optional<bool> requires_grad) {
    return create(dtype, device, SymbolicShape(dim), VaryingShape<Stride>(dim), requires_grad);
}


TensorTypePtr TensorType::create(const Tensor& t) {
    VaryingShape<bool> contiguity;
    VaryingShape<size_t> stride_indices;

    if (t.layout() == kStrided && !t.is_nested()) {
        auto shape = VaryingShape<int64_t>{t.shape().vec()};
        auto strides = VaryingShape<int64_t>{t.strides().vec()};
        return create(t.dtype(), t.device(), shape, strides,
                      t.requires_grad(), false, t.is_contiguous());
    }

    return create(t.dtype(), t.device(), SymbolicShape(), VaryingShape<Stride>{},
                  t.requires_grad(), false);
}

bool is_contiguous_stride(IntArrayView shape, IntArrayView strides) {
    if (shape.empty()) {
        return true;
    }

    auto ndim = static_cast<int>(shape.size());
    if (strides[ndim - 1] != 1) {
        return false;
    }

    for (int i = ndim - 2; i >= 0; --i) {
        if (strides[i] != strides[i + 1] * shape[i + 1]) {
            return false;
        }
    }
    return true;
}

TensorTypePtr TensorType::create_contiguous(DataType dtype, Device device, IntArrayView shape) {
    auto strides = contiguous_stride_of(shape);
    CHECK(shape.size() == strides.size());
    return create(dtype, device, VaryingShape<int64_t>(shape),
                  VaryingShape<int64_t>(strides), std::nullopt);
}

TypePtr TensorType::create_from_bool_type() {
    return create_contiguous(DataType::Bool(), Device::CPU(), {});
}

TypePtr TensorType::create_from_number_type(const Type& t) {
    if (t.is_subtype_of(*IntType::Global())) {
        return create_contiguous(DataType::Int(64), Device::CPU(), {});
    }

    if (t.is_subtype_of(*FloatType::Global())) {
        return create_contiguous(DataType::Double(), Device::CPU(), {});
    }

    if (t.is_subtype_of(*BoolType::Global())) {
        return create_contiguous(DataType::Bool(), Device::CPU(), {});
    }

    if (t.kind() == NumberType::Kind) {
        return create(std::nullopt, Device::CPU(), {}, std::nullopt);
    }

    CHECK(false) << "Unknown number type: " << t.str();
    AETHERMIND_UNREACHABLE();
}


TensorTypePtr TensorType::with_requires_grad(std::optional<bool> s) const {
    auto cloned = clone();
    cloned->requires_grad_ = s;
    return cloned;
}

TensorTypePtr TensorType::with_data_type(const std::optional<DataType>& dtype) const {
    auto cloned = clone();
    cloned->dtype_ = dtype;
    return cloned;
}

TensorTypePtr TensorType::with_dim(std::optional<size_t> d) const {
    auto cloned = clone();
    cloned->shape_ = SymbolicShape(d);
    cloned->strides_ = VaryingShape<Stride>(d);
    return cloned;
}

TensorTypePtr TensorType::with_strides(VaryingShape<Stride> strides) const {
    auto cloned = clone();
    cloned->strides_ = std::move(strides);
    return cloned;
}

TensorTypePtr TensorType::with_shape(IntArrayView shape) const {
    return with_shape_and_strides(shape, contiguous_stride_of(shape));
}

TensorTypePtr TensorType::with_device(std::optional<Device> device) const {
    auto cloned = clone();
    cloned->device_ = device;
    return cloned;
}

TensorTypePtr TensorType::with_shape_and_strides(IntArrayView shape, IntArrayView strides) const {
    auto cloned = clone();
    cloned->shape_ = SymbolicShape(shape);
    cloned->strides_ = compute_stride_props(shape, strides);
    return cloned;
}

TensorTypePtr TensorType::with_symbolic_shape(SymbolicShape symbolic_shape) const {
    auto cloned = clone();
    cloned->shape_ = std::move(symbolic_shape);
    return cloned;
}

TensorTypePtr TensorType::with_undefined() const {
    auto cloned = clone();
    cloned->undefined_ = true;
    return cloned;
}

TensorTypePtr TensorType::with_possibly_undefined() const {
    auto cloned = clone();
    cloned->undefined_ = std::nullopt;
    return cloned;
}

TensorTypePtr TensorType::dimensioned_only() const {
    auto cloned = clone();
    cloned->shape_ = SymbolicShape(shape().size());
    cloned->strides_ = VaryingShape<Stride>(shape().size());
    return cloned;
}

const TensorTypePtr& TensorType::get() {
    static auto ptr = create({}, {}, SymbolicShape(), VaryingShape<Stride>{}, {});
    return ptr;
}

}// namespace aethermind