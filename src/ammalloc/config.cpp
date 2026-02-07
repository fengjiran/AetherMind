//
// Created by richard on 2/7/26.
//
#include "ammalloc/config.h"
#include "ammalloc/common.h"

#include <cctype>
#include <cstdlib>

namespace aethermind {

void RuntimeConfig::InitFromEnv() {
    if (const char* env = std::getenv("AM_TC_SIZE")) {
        if (const auto val = details::ParseSize(env); val > 0) {
            max_tc_size_ = val < SizeConfig::MAX_TC_SIZE ? val : SizeConfig::MAX_TC_SIZE;
        }
    }

    if (const char* env = std::getenv("AM_USE_MAP_POPULATE")) {
        use_map_populate = details::ParseBool(env);
    }
}

}// namespace aethermind