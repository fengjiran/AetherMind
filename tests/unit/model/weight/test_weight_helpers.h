#ifndef AETHERMIND_TESTS_UNIT_MODEL_WEIGHT_TEST_WEIGHT_HELPERS_H
#define AETHERMIND_TESTS_UNIT_MODEL_WEIGHT_TEST_WEIGHT_HELPERS_H

#include "aethermind/model/raw_weight.h"

#include <cstddef>
#include <vector>

namespace aethermind::test {

struct TestStorage : RawStorage {
    explicit TestStorage(size_t nbytes) : data(nbytes) {}
    std::vector<std::byte> data;
};

} // namespace aethermind::test

#endif
