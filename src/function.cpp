//
// Created by richard on 9/24/25.
//

#include "function.h"

namespace aethermind {

bool Function::defined() const noexcept {
    return pimpl_;
}

uint32_t Function::use_count() const noexcept {
    return pimpl_.use_count();
}

bool Function::unique() const noexcept {
    return pimpl_.unique();
}

FunctionImpl* Function::get_impl_ptr_unsafe() const noexcept {
    return pimpl_.get();
}

FunctionImpl* Function::release_impl_unsafe() noexcept {
    return pimpl_.release();
}

}// namespace aethermind
