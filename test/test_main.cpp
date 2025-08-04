//
// Created by richard on 4/29/24.
//
#include "glog/logging.h"
#include "gtest/gtest.h"
// #include "thread_pool.h"

int main() {
    testing::InitGoogleTest();
    google::InitGoogleLogging("AetherMindTest");
    FLAGS_logtostderr = true;
    return RUN_ALL_TESTS();
}