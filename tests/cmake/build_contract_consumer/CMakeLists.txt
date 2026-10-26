cmake_minimum_required(VERSION 3.28)
project(AetherMindContractConsumer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
find_package(Threads REQUIRED)
include("${AETHERMIND_CONTRACT_TARGETS}")

add_library(contract_cpp_consumer OBJECT consumer.cpp)
target_link_libraries(contract_cpp_consumer PRIVATE AetherMind)

add_library(contract_c_consumer OBJECT "${AETHERMIND_SOURCE_DIR}/tests/unit/c_api/test_c_api_contract.c")
target_link_libraries(contract_c_consumer PRIVATE AetherMind)
set_target_properties(contract_c_consumer PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
