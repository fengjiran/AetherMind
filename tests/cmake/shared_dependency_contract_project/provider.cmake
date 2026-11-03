macro(provide_shared_dependency method dependency)
    if("${dependency}" STREQUAL "shared_probe")
        add_subdirectory("${PROVIDER_SOURCE}" "${CMAKE_BINARY_DIR}/provider-build")
        FetchContent_SetPopulated(shared_probe SOURCE_DIR "${PROVIDER_SOURCE}"
                BINARY_DIR "${CMAKE_BINARY_DIR}/provider-build")
    endif()
endmacro()
cmake_language(SET_DEPENDENCY_PROVIDER provide_shared_dependency
        SUPPORTED_METHODS FETCHCONTENT_MAKEAVAILABLE_SERIAL)
