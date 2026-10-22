add_executable(aethermind_infer
        inference_main.cpp
        inference_runner.cpp
)

target_link_libraries(aethermind_infer PRIVATE
        AetherMind
        spdlog::spdlog
)
