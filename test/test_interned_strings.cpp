//
// Created by richard on 11/24/25.
//
#include "interned_strings.h"

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

TEST(InternedString, basic) {
    // const auto& interned_string = InternedStrings::Global();
    auto all_sym_name = InternedStrings::Global().ListAllSymbols();
    for (const auto& sym_name : all_sym_name) {
        std::cout << sym_name << std::endl;
    }
}

}// namespace
