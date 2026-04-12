//
// Created by richard on 4/12/26.
//

#ifndef AETHERMIND_TENSOR_COMPAT_H
#define AETHERMIND_TENSOR_COMPAT_H

#include "aethermind/base/tensor.h"
#include "aethermind/memory/buffer.h"
#include "memory/storage_impl.h"
#include "tensor_bk.h"

namespace aethermind {

/// Converts legacy Storage to the new Buffer abstraction.
///
/// The returned Buffer aliases the legacy Storage memory and keeps the Storage
/// alive through the Buffer deleter context. The legacy storage offset is not
/// part of this conversion.
AM_NODISCARD Buffer BufferFromLegacyStorage(const Storage& storage);

/// Converts legacy Tensor_BK to the new Tensor abstraction.
///
/// storage_offset() is expressed in elements on the legacy side and is
/// converted to byte_offset() in bytes on the new side.
AM_NODISCARD Tensor TensorFromLegacy(const Tensor_BK& legacy);

/// Converts the new Tensor abstraction to legacy Tensor_BK.
///
/// This conversion preserves shape/stride metadata. Because DataPtr cannot hold
/// an arbitrary deleter context, the current implementation materializes a new
/// legacy Storage allocation and copies CPU-visible bytes into it.
AM_NODISCARD Tensor_BK LegacyTensorFromTensor(const Tensor& tensor);

}// namespace aethermind

#endif// AETHERMIND_TENSOR_COMPAT_H
