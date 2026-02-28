//
// Created by richard on 2/6/26.
//

#ifndef AETHERMIND_MALLOC_AMMALLOC_H
#define AETHERMIND_MALLOC_AMMALLOC_H

#include <cstddef>

namespace aethermind {

void* am_malloc(size_t size);

void am_free(void* ptr);

}// namespace aethermind

#endif//AETHERMIND_MALLOC_AMMALLOC_H
