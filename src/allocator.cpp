//
// Created by 赵丹 on 25-1-23.
//

#include "allocator.h"

namespace aethermind {

REGISTER_ALLOCATOR(DeviceType::kUndefined, UndefinedAllocator);

static void delete_general_data_ptr(void* ptr) {
    delete static_cast<GeneralDataPtrContext*>(ptr);
}

DataPtr GeneralDataPtrContext::make_data_ptr(void* ptr, deleter_type deleter, Device device) {
    auto* ctx = new GeneralDataPtrContext(ptr, deleter);
    return {ptr, ctx, delete_general_data_ptr, device};
}

}// namespace aethermind
