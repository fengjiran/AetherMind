//
// Created by 赵丹 on 2025/8/24.
//
#include "../include/memory/storage.h"

#include <gtest/gtest.h>

using namespace aethermind;
namespace {

TEST(Storage, init) {
    Storage s1;
    EXPECT_EQ(s1.use_count(), 0);
    EXPECT_FALSE(s1.defined());
    EXPECT_EQ(s1.nbytes(), 0);
    EXPECT_TRUE(s1.data() == nullptr);
    EXPECT_TRUE(s1.device() == Device(kUndefined));
}

// TEST(Storage, unique_void_ptr) {
//     struct Context {
//         explicit Context(void* ptr) : data(ptr) {}
//         void delete_ptr() const {
//             // std::cout << "call free.\n";
//             free(data);
//         }
//         void* data;
//     };
//
//     auto default_deleter = [](void* ptr) {
//         static_cast<Context*>(ptr)->delete_ptr();
//     };
//
//     auto* ptr = malloc(10);
//     Context ctx(ptr);
//     UniqueVoidPtr p(ptr, static_cast<void*>(&ctx), default_deleter);
// }

TEST(Storage, data_ptr_context) {
    auto deleter1 = [](void* ptr) {
        LOG(INFO) << "call test deleter1, free ptr: " << ptr;
        free(ptr);
    };

    auto deleter2 = [](void* ptr) {
        LOG(INFO) << "call test deleter2, free ptr: " << ptr;
        free(ptr);
    };

    auto* ptr = malloc(10);
    DataPtr data_ptr(ptr, deleter1, Device(kCPU));
    EXPECT_TRUE(data_ptr.device() == Device(kCPU));
    EXPECT_TRUE(data_ptr.get() == ptr);
    EXPECT_TRUE(data_ptr.get_deleter() == deleter1);

    data_ptr.compare_and_exchange_deleter(deleter1, deleter2);
    EXPECT_TRUE(data_ptr.get_deleter() == deleter2);
}

}