//
// Created by richard on 10/2/25.
//
#include "type_system/tensor_type.h"
#include "tensor.h"

#include <numeric>
#include <utility>

namespace aethermind {

std::atomic<size_t> ShapeSymbol::num_symbols_ = 1;

std::ostream& operator<<(std::ostream& os, const ShapeSymbol& s) {
    if (s.IsStatic()) {
        os << s.value();
    } else {
        os << "SS(" << s.value() << ')';
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const SymbolicShape& s) {
    const auto rank_opt = s.rank();
    if (!rank_opt.has_value()) {
        os << "(*)";
        return os;
    }

    const auto& shape = s.shape().value();
    os << "(";
    for (size_t i = 0; i < rank_opt.value(); ++i) {
        if (i > 0) {
            os << ", ";
        }

        if (shape[i].IsStatic()) {
            os << shape[i];
        } else {
            os << "*";
        }
    }
    os << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const Stride& s) {
    os << "{";
    if (s.stride_idx().has_value()) {
        os << s.stride_idx().value();
    } else {
        os << "*";
    }

    os << ":";
    if (s.stride().has_value()) {
        os << s.stride().value();
    } else {
        os << "*";
    }
    os << '}';
    return os;
}

SymbolicShape::SymbolicShape(std::optional<size_t> rank) {
    if (rank.has_value()) {
        std::vector<ShapeSymbol> symbolic_shape(rank.value());
        for (size_t i = 0; i < rank.value(); ++i) {
            symbolic_shape[i] = ShapeSymbol::Create();
        }
        symbolic_shape_ = symbolic_shape;
    }
}

SymbolicShape::SymbolicShape(const std::vector<std::optional<int64_t>>& shape) {
    std::vector<ShapeSymbol> symbolic_shape(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i].has_value()) {
            symbolic_shape[i] = ShapeSymbol::CreateFromValue(shape[i].value());
        } else {
            symbolic_shape[i] = ShapeSymbol::Create();
        }
    }
    symbolic_shape_ = symbolic_shape;
}

SymbolicShape::SymbolicShape(IntArrayView shape) {
    std::vector<ShapeSymbol> symbolic_shape(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        symbolic_shape[i] = ShapeSymbol::CreateFromValue(shape[i]);
    }
    symbolic_shape_ = symbolic_shape;
}

ShapeSymbol SymbolicShape::operator[](size_t i) const {
    if (!symbolic_shape_.has_value()) {
        AETHERMIND_THROW(RuntimeError) << "Rank isn't fixed";
    }
    return symbolic_shape_.value()[i];
}

ShapeSymbol SymbolicShape::at(size_t i) const {
    if (!symbolic_shape_.has_value()) {
        AETHERMIND_THROW(RuntimeError) << "Rank isn't fixed";
    }

    if (i >= symbolic_shape_->size()) {
        AETHERMIND_THROW(OutOfRangeError) << "Out of range";
    }
    return symbolic_shape_.value()[i];
}

std::optional<size_t> SymbolicShape::rank() const {
    if (symbolic_shape_.has_value()) {
        return symbolic_shape_->size();
    }
    return std::nullopt;
}

const std::optional<std::vector<ShapeSymbol>>& SymbolicShape::shape() const {
    return symbolic_shape_;
}

std::optional<std::vector<bool>> SymbolicShape::GetSymbolicDims() const {
    const auto rank_opt = rank();
    if (!rank_opt.has_value()) {
        return std::nullopt;
    }

    const auto n = rank_opt.value();
    std::vector<bool> is_symbolic_dims;
    is_symbolic_dims.reserve(n);
    for (const auto& s: symbolic_shape_.value()) {
        is_symbolic_dims.push_back(!s.IsStatic());
    }
    return is_symbolic_dims;
}

bool SymbolicShape::IsComplete() const {
    if (!symbolic_shape_.has_value()) {
        return false;
    }

    auto shape = symbolic_shape_.value();
    return std::all_of(shape.begin(), shape.end(),
                       [](const ShapeSymbol& s) { return s.IsStatic(); });
}

void SymbolicShape::Dump() const {
    std::cout << *this << std::endl;
}

SymbolicShape SymbolicShape::Merge(const SymbolicShape& other) const {
    if (!symbolic_shape_.has_value() ||
        !other.symbolic_shape_.has_value() ||
        symbolic_shape_->size() != other.symbolic_shape_->size()) {
        return {};
    }

    const auto n = symbolic_shape_->size();
    std::vector<ShapeSymbol> dims(n);
    for (size_t i = 0; i < n; ++i) {
        dims[i] = MergePrimitiveValue(symbolic_shape_.value()[i], other.symbolic_shape_.value()[i]);
    }
    return SymbolicShape(std::move(dims));
}

template<typename T>
std::optional<std::vector<T>> VaryingShape<T>::GetConcreteValue() const {
    if (!shape_.has_value()) {
        return std::nullopt;
    }

    std::vector<T> shape;
    shape.reserve(shape_->size());
    for (auto d: shape_.value()) {
        if (!d.has_value()) {
            return std::nullopt;
        }
        shape.push_back(d.value());
    }

    return shape;
}

template<typename T>
bool VaryingShape<T>::IsComplete() const {
    if (!shape_.has_value()) {
        return false;
    }

    for (auto d: shape_.value()) {
        if (!d.has_value() || !details::IsComplete(d.value())) {
            return false;
        }
    }
    return true;
}

template<typename T>
VaryingShape<T> VaryingShape<T>::Merge(const VaryingShape& other) const {
    if (!shape_.has_value() || !other.shape_.has_value() || shape_->size() != other.shape_->size()) {
        return VaryingShape();
    }

    auto n = shape_->size();
    ListOfOptionalElements shape(n);
    for (size_t i = 0; i < n; ++i) {
        shape[i] = MergePrimitiveValue(shape_.value()[i], other.shape_.value()[i]);
    }

    return VaryingShape(std::move(shape));
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const VaryingShape<T>& t) {
    const auto& shape_opt = t.shape();
    if (!shape_opt.has_value()) {
        os << "(*)";
        return os;
    }

    auto n = shape_opt->size();
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
    const auto rank = shape_.rank();
    if (!rank.has_value()) {
        return VaryingShape<int64_t>{};
    }

    const auto n = rank.value();
    std::vector<std::optional<int64_t>> shape;
    shape.reserve(n);
    for (const auto& ss: shape_.shape().value()) {
        shape.push_back(ss.IsStatic() ? std::optional(ss.GetStaticValue())
                                      : std::nullopt);
    }
    return VaryingShape{std::move(shape)};
}

VaryingShape<int64_t> TensorType::strides() const {
    const auto& strides_opt = strides_.shape();
    if (!strides_opt.has_value()) {
        return VaryingShape<int64_t>{};
    }

    std::vector<std::optional<int64_t>> strides(strides_opt->size());
    for (const auto& stride: strides_opt.value()) {
        if (!stride.has_value()) {
            continue;
        }

        if (const auto& s = stride.value(); s.stride_idx().has_value() && s.stride().has_value()) {
            strides[s.stride_idx().value()] = s.stride().value();
        }
    }
    return VaryingShape{std::move(strides)};
}

std::optional<size_t> TensorType::numel() const {
    const auto& vary_shape = shape();
    const auto& ndim = vary_shape.size();
    if (!ndim.has_value()) {
        return std::optional<size_t>{};
    }

    size_t prod = 1;
    for (size_t i = 0; i < ndim.value(); ++i) {
        auto s = vary_shape[i];
        if (!s.has_value()) {
            return std::optional<size_t>{};
        }
        prod *= s.value();
    }
    return prod;
}

bool TensorType::Equals(const Type& rhs) const {
    if (rhs.kind() != kind()) {
        return false;
    }

    const auto t = rhs.Expect<TensorType>();
    return dtype() == t->dtype() &&
           device() == t->device() &&
           shape() == t->shape() &&
           undefined() == t->undefined() &&
           GetStrideProperties() == t->GetStrideProperties() &&
           RequiresGrad() == t->RequiresGrad();
}

bool TensorType::IsSubtypeOfExtTypeImpl(const Type& rhs, std::ostream* why_not) const {
    if (auto ptr = rhs.Cast<TensorType>()) {
        if (this == ptr.get()) {
            return true;
        }
        return *Merge(*ptr) == *ptr;
    }
    return Type::IsSubtypeOfExtTypeImpl(rhs, why_not);
}

// The idea is to only mark possible overlap across dimensions. We want to
// return false for expanded tensors and permuted tensors, for which dimensional
// collapsing is safe.
bool IsCrossDimensionOverlap(IntArrayView shape, IntArrayView strides) {
    const int ndim = static_cast<int>(shape.size());
    std::vector<size_t> stride_ascend_idx(ndim);// stride ascend order index
    std::iota(stride_ascend_idx.rbegin(), stride_ascend_idx.rend(), 0);

    // sort indices going with ascending strides
    // insertion sort
    // for (int i = 1; i < ndim; ++i) {
    //     int c = i;
    //     for (int j = i - 1; j >= 0; --j) {
    //         if (strides[stride_ascend_idx[j]] > strides[stride_ascend_idx[c]]) {
    //             std::swap(stride_ascend_idx[j], stride_ascend_idx[c]);
    //             c = j;
    //         }
    //     }
    // }

    int exch_num = 0;
    for (size_t i = ndim - 1; i > 0; --i) {
        if (strides[stride_ascend_idx[i]] < strides[stride_ascend_idx[i - 1]]) {
            std::swap(stride_ascend_idx[i], stride_ascend_idx[i - 1]);
            ++exch_num;
        }
    }

    if (exch_num != 0) {
        for (int i = 2; i < ndim; ++i) {
            auto x = stride_ascend_idx[i];
            auto j = i;
            while (strides[x] < strides[stride_ascend_idx[j - 1]]) {
                stride_ascend_idx[j] = stride_ascend_idx[j - 1];
                --j;
            }
            stride_ascend_idx[j] = x;
        }
    }

    for (int i = 1; i < ndim; ++i) {
        if (i != 0) {
            // we are being conservative on checking for memory overlap
            if (shape[stride_ascend_idx[i]] != 1 &&
                strides[stride_ascend_idx[i]] < shape[stride_ascend_idx[i - 1]] * strides[stride_ascend_idx[i - 1]]) {
                return true;
            }
        }
    }
    return false;
}

VaryingShape<Stride> TensorType::ComputeStrideProps(IntArrayView shape,
                                                    IntArrayView strides,
                                                    bool tensor_contiguity) {
    int ndim = static_cast<int>(shape.size());
    std::vector<size_t> stride_idx(ndim);

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
    if (IsChannelsLastStrides2d(shape, strides) || IsChannelsLastStrides3d(shape, strides)) {
        std::iota(stride_idx.rbegin() + 1, stride_idx.rend() - 1, 2);
        stride_idx[0] = 1;
        stride_idx[ndim - 1] = 0;
    } else if (IsContiguousStride(shape, strides)) {
        std::iota(stride_idx.rbegin(), stride_idx.rend(), 0);
    } else {
        // For broadcasted dimension where stride is 0, we have to stick to
        // TensorIterator behavior in eager, where they introduce an ambiguous
        // comparison result to preserve permutation by best effort.
        // For more details, see NOTE: [Computing output strides]
        std::iota(stride_idx.rbegin(), stride_idx.rend(), 0);
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
                int comp = should_swap(stride_idx[dim0], stride_idx[dim1]);
                if (comp > 0) {
                    std::swap(stride_idx[dim0], stride_idx[dim1]);
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
            has_overlap = IsCrossDimensionOverlap(shape, strides);
        }
    }

    std::vector<Stride> stride_properties;
    stride_properties.reserve(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        auto cur_idx = stride_idx[i];
        bool is_contiguous = tensor_contiguity;
        if (!is_contiguous) {// the contiguity is not set, compute contiguity by shape and stride
            if (!has_overlap) {
                if (i == 0) {
                    is_contiguous = strides[cur_idx] == 1;
                } else {
                    auto pre_idx = stride_idx[i - 1];
                    is_contiguous = strides[cur_idx] == 1 ||
                                    (strides[cur_idx] != 0 && strides[cur_idx] == strides[pre_idx] * shape[pre_idx]);
                }
            } else {
                is_contiguous = false;
            }
        }
        stride_properties.emplace_back(cur_idx, is_contiguous, strides[cur_idx]);
    }
    return VaryingShape{stride_properties};
}

template<typename T>
static bool IsNullOrEqual(std::optional<T> a, IntArrayView b) {
    return !a.has_value() || a.value() == b;
}

bool TensorType::MatchTensor(const Tensor& t) const {
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
    bool matched_strides = !GetStrideProperties().size() ||
                           (!t.has_storage() && !GetStrideProperties().IsComplete()) ||
                           GetStrideProperties() == ComputeStrideProps(t.shape(), t.strides(), t.is_contiguous());
    return dtype().value_or(t.dtype()) == t.dtype() &&
           device().value_or(t.device()) == t.device() &&
           RequiresGrad().value_or(rg) == rg &&
           matched_strides &&
           IsNullOrEqual(shape().GetConcreteValue(), t.shape());
}


TensorTypePtr TensorType::Merge(const TensorType& other, bool merge_shape) const {
    auto data_type = MergePrimitiveValue(dtype(), other.dtype());
    auto shape = merge_shape ? GetSymbolicShape().Merge(other.GetSymbolicShape()) : GetSymbolicShape();
    auto stride_props = GetStrideProperties().Merge(other.GetStrideProperties());
    auto dev = MergePrimitiveValue(device(), other.device());
    auto gr = MergePrimitiveValue(RequiresGrad(), other.RequiresGrad());
    auto undef = MergePrimitiveValue(undefined(), other.undefined());
    return Create(data_type, dev, shape, stride_props, gr, undef);
}

TensorTypePtr TensorType::Contiguity() const {
    auto cloned = Clone();
    auto concrete_shape = shape().GetConcreteValue();
    CHECK(concrete_shape.has_value());
    cloned->strides_ = ComputeStrideProps(concrete_shape.value(),
                                          GetContiguousStrideOf(concrete_shape.value()));
    return cloned;
}

std::vector<int64_t> TensorType::GetContiguousStrideOf(IntArrayView shape, MemoryFormat memory_format) {
    if (shape.empty()) {
        return {};
    }

    const auto ndim = shape.size();
    std::vector<int64_t> stride_ascend_idx;
    if (memory_format == MemoryFormat::kChannelsLast) {
        stride_ascend_idx = {1, 3, 2, 0};
    } else if (memory_format == MemoryFormat::kChannelsLast3d) {
        stride_ascend_idx = {1, 4, 3, 2, 0};
    } else {
        stride_ascend_idx.reserve(ndim);
        for (size_t i = 0; i < ndim; ++i) {
            stride_ascend_idx.push_back(static_cast<int64_t>(ndim - i - 1));
        }
    }

    std::vector<int64_t> strides(ndim);
    strides[stride_ascend_idx[0]] = 1;
    for (int i = 1; i < ndim; ++i) {
        auto pre_idx = stride_ascend_idx[i - 1];
        auto cur_idx = stride_ascend_idx[i];
        strides[cur_idx] = strides[pre_idx] * shape[pre_idx];
    }
    return strides;
}

TensorTypePtr TensorType::Create(std::optional<DataType> dtype,
                                 std::optional<Device> device,
                                 SymbolicShape shape,
                                 VaryingShape<Stride> strides,
                                 std::optional<bool> requires_grad,
                                 std::optional<bool> undefined) {
    //NOLINTBEGIN
    auto* ptr = new TensorType(dtype, device, std::move(shape), std::move(strides),
                               requires_grad, undefined);
    //NOLINTEND
    return TensorTypePtr(ptr);
}

TensorTypePtr TensorType::Create(std::optional<DataType> dtype,
                                 std::optional<Device> device,
                                 const VaryingShape<int64_t>& shape,
                                 const VaryingShape<int64_t>& strides,
                                 std::optional<bool> requires_grad,
                                 std::optional<bool> undefined,
                                 bool tensor_contiguity) {
    auto concrete_stride = strides.GetConcreteValue();
    if (concrete_stride.has_value()) {
        auto concrete_shape = shape.GetConcreteValue();
        CHECK(concrete_shape.has_value() && concrete_shape->size() == concrete_stride->size());
        auto sprops = ComputeStrideProps(concrete_shape.value(),
                                         concrete_stride.value(),
                                         tensor_contiguity);
        auto symbol_shape = SymbolicShape(concrete_shape.value());
        return Create(dtype, device, std::move(symbol_shape), std::move(sprops), requires_grad, undefined);
    }

    // strides are all null, but still have number of strides equal to number of ranks
    const auto& shape_opt = shape.shape();
    CHECK(shape_opt.has_value() && shape.size().has_value());
    auto symbol_shape = SymbolicShape(shape_opt.value());
    return Create(dtype, device, std::move(symbol_shape),
                  VaryingShape<Stride>(shape_opt->size()), requires_grad, undefined);
}

TensorTypePtr TensorType::Create(std::optional<DataType> dtype,
                                 std::optional<Device> device,
                                 std::optional<size_t> dim,
                                 std::optional<bool> requires_grad) {
    return Create(dtype, device, SymbolicShape(dim), VaryingShape<Stride>(dim), requires_grad);
}


TensorTypePtr TensorType::Create(const Tensor& t) {
    VaryingShape<bool> contiguity;
    VaryingShape<size_t> stride_indices;

    if (t.layout() == kStrided && !t.is_nested()) {
        auto shape = VaryingShape<int64_t>{t.shape().vec()};
        auto strides = VaryingShape<int64_t>{t.strides().vec()};
        return Create(t.dtype(), t.device(), shape, strides,
                      t.requires_grad(), false, t.is_contiguous());
    }

    return Create(t.dtype(), t.device(), SymbolicShape(), VaryingShape<Stride>{},
                  t.requires_grad(), false);
}

bool IsContiguousStride(IntArrayView shape, IntArrayView strides) {
    if (shape.empty()) {
        return true;
    }

    const auto ndim = static_cast<int>(shape.size());
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

TensorTypePtr TensorType::CreateContiguous(DataType dtype, Device device, IntArrayView shape) {
    const auto strides = GetContiguousStrideOf(shape);
    CHECK(shape.size() == strides.size());
    return Create(dtype, device, VaryingShape(shape), VaryingShape(strides), std::nullopt);
}

TypePtr TensorType::CreateFromBoolType() {
    return CreateContiguous(DataType::Bool(), Device::CPU(), {});
}

TypePtr TensorType::CreateFromNumberType(const Type& t) {
    if (t.IsSubtypeOf(*IntType::Global())) {
        return CreateContiguous(DataType::Int(64), Device::CPU(), {});
    }

    if (t.IsSubtypeOf(*FloatType::Global())) {
        return CreateContiguous(DataType::Double(), Device::CPU(), {});
    }

    if (t.IsSubtypeOf(*BoolType::Global())) {
        return CreateContiguous(DataType::Bool(), Device::CPU(), {});
    }

    if (t.kind() == NumberType::Kind) {
        return Create(std::nullopt, Device::CPU(), {}, std::nullopt);
    }

    CHECK(false) << "Unknown number type: " << t.str();
    AETHERMIND_UNREACHABLE();
}


TensorTypePtr TensorType::WithRequiresGrad(std::optional<bool> s) const {
    auto cloned = Clone();
    cloned->requires_grad_ = s;
    return cloned;
}

TensorTypePtr TensorType::WithDataType(const std::optional<DataType>& dtype) const {
    auto cloned = Clone();
    cloned->dtype_ = dtype;
    return cloned;
}

TensorTypePtr TensorType::WithDim(std::optional<size_t> d) const {
    auto cloned = Clone();
    cloned->shape_ = SymbolicShape(d);
    cloned->strides_ = VaryingShape<Stride>(d);
    return cloned;
}

TensorTypePtr TensorType::WithStrides(VaryingShape<Stride> strides) const {
    auto cloned = Clone();
    cloned->strides_ = std::move(strides);
    return cloned;
}

TensorTypePtr TensorType::WithShape(IntArrayView shape) const {
    return WithShapeAndStrides(shape, GetContiguousStrideOf(shape));
}

TensorTypePtr TensorType::WithDevice(std::optional<Device> device) const {
    auto cloned = Clone();
    cloned->device_ = device;
    return cloned;
}

TensorTypePtr TensorType::WithShapeAndStrides(IntArrayView shape, IntArrayView strides) const {
    auto cloned = Clone();
    cloned->shape_ = SymbolicShape(shape);
    cloned->strides_ = ComputeStrideProps(shape, strides);
    return cloned;
}

TensorTypePtr TensorType::WithSymbolicShape(SymbolicShape symbolic_shape) const {
    auto cloned = Clone();
    cloned->shape_ = std::move(symbolic_shape);
    return cloned;
}

TensorTypePtr TensorType::WithUndefined() const {
    auto cloned = Clone();
    cloned->undefined_ = true;
    return cloned;
}

TensorTypePtr TensorType::WithPossibleUndefined() const {
    auto cloned = Clone();
    cloned->undefined_ = std::nullopt;
    return cloned;
}

TensorTypePtr TensorType::WithDimensionOnly() const {
    auto cloned = Clone();
    cloned->shape_ = SymbolicShape(shape().size());
    cloned->strides_ = VaryingShape<Stride>(shape().size());
    return cloned;
}

const TensorTypePtr& TensorType::Get() {
    static auto ptr = Create({}, {}, SymbolicShape(), VaryingShape<Stride>{}, {});
    return ptr;
}

}// namespace aethermind