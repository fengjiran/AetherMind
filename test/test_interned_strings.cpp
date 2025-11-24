//
// Created by richard on 11/24/25.
//
#include "interned_strings.h"

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

TEST(InternedString, basic) {
    for (const auto& sym_name: InternedStrings::Global().ListAllSymbolNames()) {
        std::cout << sym_name << std::endl;
    }
}

}// namespace
