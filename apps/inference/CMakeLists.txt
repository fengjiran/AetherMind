add_executable(aethermind_infer
        inference_main.cpp
        inference_runner.cpp
)
aethermind_apply_sanitizer(aethermind_infer)

target_link_libraries(aethermind_infer PRIVATE
        AetherMind
        spdlog::spdlog
)
