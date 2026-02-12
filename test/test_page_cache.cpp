//
// Created by richard on 2/12/26.
//
#include "ammalloc/page_cache.h"

#include <atomic>
#include <cstring>
#include <gtest/gtest.h>

namespace aethermind {

class PageCacheTest : public ::testing::Test {
protected:
    PageCache& cache_ = PageCache::GetInstance();

    bool IsBucketEmpty(size_t page_num) const {
        return cache_.span_lists_[page_num].empty();
    }
};

}// namespace aethermind


namespace {
using namespace aethermind;


}// namespace