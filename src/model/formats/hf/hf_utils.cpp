#include "aethermind/model/formats/hf/hf_utils.h"

#include <string>

namespace aethermind::hf {

StatusOr<DataType> ParseSafetensorsDType(std::string_view dtype_text) {
    const auto equals = [dtype_text](std::string_view expected) noexcept {
        return dtype_text.compare(expected) == 0;
    };

    if (equals("F16")) {
        return DataType::Float(16);
    }

    if (equals("BF16")) {
        return DataType::BFloat(16);
    }

    if (equals("F32")) {
        return DataType::Float32();
    }

    if (equals("F64")) {
        return DataType::Float(64);
    }

    if (equals("I16")) {
        return DataType::Int(16);
    }

    if (equals("I32")) {
        return DataType::Int(32);
    }

    if (equals("I64")) {
        return DataType::Int(64);
    }

    if (equals("I8")) {
        return DataType::Int(8);
    }

    if (equals("U8")) {
        return DataType::UInt(8);
    }

    if (equals("U16")) {
        return DataType::UInt(16);
    }

    if (equals("U32")) {
        return DataType::UInt(32);
    }

    if (equals("U64")) {
        return DataType::UInt(64);
    }

    if (equals("BOOL")) {
        return DataType::Bool();
    }

    if (equals("F8_E5M2")) {
        return DataType::Float8E5M2();
    }

    if (equals("F8_E4M3")) {
        return DataType::Float8E4M3();
    }

    return Status::InvalidArgument(
            std::string("Unsupported safetensors dtype: ") + std::string(dtype_text));
}

}// namespace aethermind::hf
