#ifndef AETHERMIND_BACKEND_BACKEND_H
#define AETHERMIND_BACKEND_BACKEND_H

/// @file backend.h
/// @brief Backend abstraction for planning-time kernel resolution.
///
/// Defines the `Backend` interface used during execution planning to resolve
/// a concrete `ResolvedKernel` from an `OpType`, `KernelSelector`, and typed
/// `OpParams`. The backend owns capability reporting and kernel lookup; it
/// does not execute kernels.

#include "aethermind/backend/backend_fwd.h"
#include "aethermind/backend/packed_weight.h"
#include "aethermind/backend/resolved_kernel.h"
#include "aethermind/base/macros.h"
#include "aethermind/operators/op_params.h"

#include <span>

namespace aethermind {

class TensorView;

/// @brief Abstract backend for planning-time kernel selection and preparation.
///
/// Each backend handles one `DeviceType` and exposes its capabilities. The
/// sole planning-time entry point is `PrepareKernel`, which resolves and
/// freezes a `ResolvedKernel` for later execution without further lookup.
class Backend {
public:
    virtual ~Backend() = default;

    /// @brief Returns the device type handled by this backend.
    ///
    /// @return Device type (e.g. `kCPU`).
    AM_NODISCARD virtual DeviceType device_type() const noexcept = 0;

    /// @brief Resolves a kernel and freezes metadata for execution.
    ///
    /// Freezes type-erased metadata derived from typed semantic parameters
    /// into the returned value, including its scratch-space requirement.
    /// This is the planning-time kernel-resolution entry point; the returned
    /// `ResolvedKernel` is later owned by an `ExecutionStep` and invoked without
    /// another backend lookup. Packed artifacts use GetPackingRecipe during
    /// model preparation before the request is passed to PackWeights.
    ///
    /// @param op_type Operator type to resolve.
    /// @param selector Selector describing device, dtype, and layout constraints.
    /// @param params Typed operator parameters for metadata derivation.
    /// @return Resolved kernel on success, or an error when no eligible kernel
    ///         exists or metadata derivation fails.
    AM_NODISCARD virtual StatusOr<ResolvedKernel> PrepareKernel(
            OpType op_type,
            const KernelSelector& selector,
            const OpParams& params) const = 0;

    /// @brief Returns the packing recipe selected for this operator/selector.
    ///
    /// Backends must use the same descriptor eligibility and priority policy
    /// as PrepareKernel. This query is used during model preparation, where
    /// operator params and tensor shapes are not available yet.
    virtual StatusOr<PackingRecipe> GetPackingRecipe(
            OpType op_type, const KernelSelector& selector) const {
        UNUSED(op_type);
        UNUSED(selector);
        return Status::Unimplemented(
                "Backend does not expose descriptor packing recipes");
    }

    /// @brief Packs logical weight views using an explicit consumer recipe.
    ///
    /// `components` carries weight views in logical binding order: exactly one
    /// for direct bindings, several (Q/K/V or Gate/Up) for composite bindings.
    /// The backend owns the packing layout authority: it validates
    /// rank/dtype/feature-count agreement, materializes any composite
    /// concatenation, allocates aligned storage, and returns an artifact whose
    /// `recipe()` the caller must use when building the `WeightArtifactKey`.
    ///
    /// Callers supply the recipe selected for the consumer through
    /// GetPackingRecipe. This is the only packing entry; packing support is
    /// optional, and the default returns Unimplemented.
    ///
    /// @param op_type Operator the packed weight serves.
    /// @param components Weight views in logical binding order.
    /// @param selector Selector requesting WeightFormat::kPacked.
    /// @param recipe Exact physical layout and alignment selected for the consumer.
    /// @return An artifact produced with the requested recipe, or an error when
    ///         the backend cannot satisfy the packing contract.
    virtual StatusOr<std::unique_ptr<PackedWeight>> PackWeights(
            OpType op_type,
            std::span<const TensorView> components,
            const KernelSelector& selector,
            const PackingRecipe& recipe) const {
        UNUSED(op_type);
        UNUSED(components);
        UNUSED(selector);
        UNUSED(recipe);
        return Status::Unimplemented(
                "Backend does not implement weight packing");
    }

    /// @brief Returns the kernel registry for debug inspection, if available.
    ///
    /// @return Pointer to the registry, or `nullptr` when the backend does not
    ///         expose one.
    AM_NODISCARD virtual const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept = 0;
};

} // namespace aethermind
#endif
