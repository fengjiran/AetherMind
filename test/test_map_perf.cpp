//
// Created by richard on 1/10/26.
//
// #include "container/map.h"
#include "container/map_v1.h"
#include "test_utils.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

// class MapBenchmarkTest : public ::testing::Test {
// protected:
//     static constexpr size_t SMALL_SIZE = 1000;
//     static constexpr size_t MEDIUM_SIZE = 10000;
//     static constexpr size_t LARGE_SIZE = 100000;
//
//     std::vector<std::pair<int, int>> small_data_;
//     std::vector<std::pair<int, int>> medium_data_;
//     std::vector<std::pair<int, int>> large_data_;
//
//     void SetUp() override {
//         small_data_ = GenerateRandomData(SMALL_SIZE);
//         medium_data_ = GenerateRandomData(MEDIUM_SIZE);
//         large_data_ = GenerateRandomData(LARGE_SIZE);
//     }
// };
constexpr size_t SMALL_SIZE = 1000;
constexpr size_t MEDIUM_SIZE = 10000;
constexpr size_t LARGE_SIZE = 100000;

// insert performance test
TEST(MapBenchMarkTest, InsertPerformance) {
    // test small data insert
    {
        auto small_data = GenerateRandomData(SMALL_SIZE);
        {
            Timer timer("Map small insert");
            MapV1<int, int> map;
            for (const auto& [key, value]: small_data) {
                map.insert(key, value);
            }
        }

        {
            Timer timer("std::unordered_map small insert");
            std::unordered_map<int, int> map;
            for (const auto& [key, value]: small_data) {
                map.insert({key, value});
            }
        }
    }

    // test medium data insert
    {
        auto medium_data = GenerateRandomData(MEDIUM_SIZE);
        {
            Timer timer("Map medium insert");
            MapV1<int, int> map;
            for (const auto& [key, value]: medium_data) {
                map.insert(key, value);
            }
        }

        {
            Timer timer("std::unordered_map medium insert");
            std::unordered_map<int, int> map;
            for (const auto& [key, value]: medium_data) {
                map.insert({key, value});
            }
        }
    }

    // test large data insert
    {
        auto large_data = GenerateRandomData(LARGE_SIZE);
        {
            Timer timer("Map large insert");
            MapV1<int, int> map;
            for (const auto& [key, value]: large_data) {
                map.insert(key, value);
            }
        }

        {
            Timer timer("std::unordered_map large insert");
            std::unordered_map<int, int> map;
            for (const auto& [key, value]: large_data) {
                map.insert({key, value});
            }
        }
    }
}

// find performance test
TEST(MapBenchMarkTest, FindPerformance) {
    // prepare dataset
    MapV1<int, int> map;
    std::unordered_map<int, int> std_map;
    auto large_data = GenerateRandomData(LARGE_SIZE);

    for (const auto& [key, value]: large_data) {
        map.insert(key, value);
        std_map.insert({key, value});
    }

    {
        Timer timer("Map find");
        size_t found = 0;
        for (const auto& [key, value]: large_data) {
            if (map.find(key) != map.end()) {
                found++;
            }
        }
        EXPECT_EQ(found, large_data.size());
    }

    {
        Timer timer("std::unordered_map find");
        size_t found = 0;
        for (const auto& [key, value]: large_data) {
            if (std_map.find(key) != std_map.end()) {
                found++;
            }
        }
        EXPECT_EQ(found, large_data.size());
    }
}

// erase performance test
TEST(MapBenchMarkTest, ErasePerformance) {
    auto medium_data = GenerateRandomData(MEDIUM_SIZE);
    // 测试Map删除性能
    {
        Timer timer("Map erase");
        MapV1<int, int> map;
        for (const auto& [key, value]: medium_data) {
            map.insert(key, value);
        }

        for (const auto& [key, value]: medium_data) {
            map.erase(key);
        }

        EXPECT_TRUE(map.empty());
    }

    // 测试std::unordered_map删除性能
    {
        Timer timer("std::unordered_map erase");
        std::unordered_map<int, int> map;
        for (const auto& [key, value]: medium_data) {
            map.insert({key, value});
        }

        for (const auto& [key, value]: medium_data) {
            map.erase(key);
        }

        EXPECT_TRUE(map.empty());
    }
}

// traverse performance test
TEST(MapBenchMarkTest, TraversePerformance) {
    // 准备数据
    MapV1<int, int> map;
    std::unordered_map<int, int> std_map;
    auto large_data = GenerateRandomData(LARGE_SIZE);

    for (const auto& [key, value]: large_data) {
        map.insert(key, value);
        std_map.insert({key, value});
    }

    // 测试Map遍历性能
    {
        Timer timer("Map iterate");
        size_t count = 0;
        for (const auto& [key, value]: map) {
            count++;
        }
        EXPECT_EQ(count, map.size());
    }

    // 测试std::unordered_map遍历性能
    {
        Timer timer("std::unordered_map iterate");
        size_t count = 0;
        for (const auto& [key, value]: std_map) {
            count++;
        }
        EXPECT_EQ(count, std_map.size());
    }
}

TEST(MapBenchMarkTest, OperatorBracketPerformance) {
    auto medium_data_ = GenerateRandomData(MEDIUM_SIZE);
    // 测试Map的operator[]性能
    {
        Timer timer("Map operator[]");
        MapV1<int, int> map;
        for (size_t i = 0; i < medium_data_.size(); ++i) {
            map[static_cast<int>(i)] = static_cast<int>(i * 2);
        }

        // 读取测试
        size_t sum = 0;
        for (size_t i = 0; i < medium_data_.size(); ++i) {
            sum += map[static_cast<int>(i)];
        }
    }

    // 测试std::unordered_map的operator[]性能
    {
        Timer timer("std::unordered_map operator[]");
        std::unordered_map<int, int> map;
        for (size_t i = 0; i < medium_data_.size(); ++i) {
            map[static_cast<int>(i)] = static_cast<int>(i * 2);
        }

        // 读取测试
        size_t sum = 0;
        for (size_t i = 0; i < medium_data_.size(); ++i) {
            sum += map[static_cast<int>(i)];
        }
    }
}

TEST(MapBenchMarkTest, CopyAndMovePerformance) {
    auto medium_data = GenerateRandomData(MEDIUM_SIZE);
    // 准备数据
    MapV1<int, int> map;
    for (const auto& [key, value]: medium_data) {
        map.insert(key, value);
    }

    // 测试拷贝性能
    {
        Timer timer("Map copy");
        MapV1<int, int> map_copy(map);
        EXPECT_EQ(map_copy.size(), map.size());
    }

    // 测试移动性能
    {
        Timer timer("Map move");
        MapV1<int, int> map_move(std::move(map));
        EXPECT_LE(map_move.size(), medium_data.size());
        EXPECT_TRUE(map.empty());
    }
}

}// namespace