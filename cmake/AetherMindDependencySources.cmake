include_guard(GLOBAL)

# Reuse prepared sources without letting independent build trees update them.
# FetchContent still owns each build tree's binary and population directories,
# and explicit source overrides and dependency providers retain their precedence.
function(aethermind_shared_dependency_source_options source_dir output)
    if(EXISTS "${source_dir}/CMakeLists.txt")
        set(${output}
                SOURCE_DIR "${source_dir}"
                DOWNLOAD_COMMAND "${CMAKE_COMMAND}" -E true
                UPDATE_COMMAND "${CMAKE_COMMAND}" -E true
                PARENT_SCOPE)
    else()
        set(${output} "" PARENT_SCOPE)
    endif()
endfunction()
