//
// Created by richard on 2/6/26.
//

#ifndef AMMALLOC_AMMALLOC_H
#define AMMALLOC_AMMALLOC_H

#include <cstddef>

namespace ammalloc {

void* am_malloc(size_t original_size);

void am_free(void* ptr);

}// namespace ammalloc

#endif// AMMALLOC_AMMALLOC_H
