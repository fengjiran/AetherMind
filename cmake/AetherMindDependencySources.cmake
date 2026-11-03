include_guard(GLOBAL)

include(FetchContent)

# Defer acquisition to FetchContent so overrides, providers and earlier
# declarations retain their precedence. Populate first, validate, then configure
# third-party code in aethermind_make_shared_dependency_available().
function(aethermind_declare_shared_dependency name source_dir repository tag)
    string(TOLOWER "${name}" key)
    if(NOT IS_ABSOLUTE "${source_dir}" OR "${tag}" STREQUAL "")
        message(FATAL_ERROR "${name} requires an absolute source directory and a version tag")
    endif()
    set(script "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/AcquireSharedDependency.cmake")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${script}")
    set(arguments
            "-DAETHERMIND_DEPENDENCY_NAME=${name}"
            "-DAETHERMIND_DEPENDENCY_SOURCE=${source_dir}"
            "-DAETHERMIND_DEPENDENCY_REPOSITORY=${repository}"
            "-DAETHERMIND_DEPENDENCY_TAG=${tag}")
    foreach(option LOCK_TIMEOUT DOWNLOAD_TIMEOUT)
        if(DEFINED AETHERMIND_DEPENDENCY_${option})
            list(APPEND arguments "-DAETHERMIND_DEPENDENCY_${option}=${AETHERMIND_DEPENDENCY_${option}}")
        endif()
    endforeach()
    if(GIT_EXECUTABLE)
        list(APPEND arguments "-DGIT_EXECUTABLE=${GIT_EXECUTABLE}")
    endif()
    list(APPEND arguments -P "${script}")
    FetchContent_Declare(${name}
            GIT_REPOSITORY "${repository}"
            GIT_TAG "${tag}"
            SOURCE_DIR "${source_dir}"
            # An absent subdirectory prevents MakeAvailable from executing
            # unvalidated CMakeLists.txt, including in disconnected mode.
            SOURCE_SUBDIR __aethermind_population_only__
            DOWNLOAD_COMMAND "${CMAKE_COMMAND}" -DAETHERMIND_DEPENDENCY_MODE=ENSURE ${arguments}
            UPDATE_COMMAND "${CMAKE_COMMAND}" -DAETHERMIND_DEPENDENCY_MODE=ENSURE ${arguments})
    get_property(already_declared GLOBAL PROPERTY AETHERMIND_SHARED_${key}_SOURCE SET)
    if(NOT already_declared)
        set_property(GLOBAL PROPERTY AETHERMIND_SHARED_${key}_SOURCE "${source_dir}")
        set_property(GLOBAL PROPERTY AETHERMIND_SHARED_${key}_ACQUIRE_ARGS "${arguments}")
    endif()
endfunction()

function(aethermind_make_shared_dependency_available _aethermind_name)
    cmake_parse_arguments(PARSE_ARGV 1 _aethermind_arg "" "" "TARGETS")
    if(NOT _aethermind_arg_TARGETS)
        message(FATAL_ERROR "${_aethermind_name}: specify the required dependency targets")
    endif()
    string(TOLOWER "${_aethermind_name}" _aethermind_key)
    string(TOUPPER "${_aethermind_name}" _aethermind_upper_name)
    get_property(_aethermind_shared_source GLOBAL PROPERTY AETHERMIND_SHARED_${_aethermind_key}_SOURCE)
    if("${_aethermind_shared_source}" STREQUAL "")
        message(FATAL_ERROR "Shared dependency ${_aethermind_name} has not been declared")
    endif()

    # ExternalProject runs UPDATE_COMMAND in SOURCE_DIR. An empty placeholder
    # allows recovery when a checkout was deleted but build stamps remain.
    # No acquisition occurs here, so providers may still satisfy the request.
    if(NOT FETCHCONTENT_FULLY_DISCONNECTED AND NOT FETCHCONTENT_SOURCE_DIR_${_aethermind_upper_name}
            AND NOT EXISTS "${_aethermind_shared_source}" AND NOT IS_SYMLINK "${_aethermind_shared_source}")
        file(MAKE_DIRECTORY "${_aethermind_shared_source}")
    endif()
    FetchContent_MakeAvailable(${_aethermind_name})
    FetchContent_GetProperties(${_aethermind_name}
            SOURCE_DIR _aethermind_source BINARY_DIR _aethermind_binary)

    set(_aethermind_present)
    foreach(_aethermind_target IN LISTS _aethermind_arg_TARGETS)
        if(TARGET ${_aethermind_target})
            list(APPEND _aethermind_present "${_aethermind_target}")
        endif()
    endforeach()
    list(LENGTH _aethermind_present _aethermind_present_count)
    list(LENGTH _aethermind_arg_TARGETS _aethermind_required_count)
    if(_aethermind_present_count EQUAL 0)
        if(NOT FETCHCONTENT_SOURCE_DIR_${_aethermind_upper_name})
            file(REAL_PATH "${_aethermind_source}" _aethermind_actual_source)
            file(REAL_PATH "${_aethermind_shared_source}" _aethermind_expected_source)
            if(NOT _aethermind_actual_source STREQUAL _aethermind_expected_source)
                message(FATAL_ERROR "${_aethermind_name}: provider or earlier declaration did not supply required targets")
            endif()
            # Native stamps and UPDATE_DISCONNECTED can skip population work.
            # Check the effective source before configuring third-party code;
            # only fully disconnected mode forbids acquiring missing contents.
            get_property(_aethermind_arguments GLOBAL PROPERTY AETHERMIND_SHARED_${_aethermind_key}_ACQUIRE_ARGS)
            if(FETCHCONTENT_FULLY_DISCONNECTED)
                set(_aethermind_mode VERIFY)
            else()
                set(_aethermind_mode ENSURE)
            endif()
            execute_process(COMMAND "${CMAKE_COMMAND}" "-DAETHERMIND_DEPENDENCY_MODE=${_aethermind_mode}"
                    ${_aethermind_arguments} RESULT_VARIABLE _aethermind_result
                    OUTPUT_VARIABLE _aethermind_output ERROR_VARIABLE _aethermind_error)
            if(NOT "${_aethermind_result}" STREQUAL "0")
                message(FATAL_ERROR "${_aethermind_name}: shared source validation failed:\n${_aethermind_output}\n${_aethermind_error}")
            endif()
        endif()
        if(NOT EXISTS "${_aethermind_source}/CMakeLists.txt")
            message(FATAL_ERROR "${_aethermind_name}: dependency source has no CMakeLists.txt: ${_aethermind_source}")
        endif()
        # Prefix local variables: subprojects inherit the calling scope.
        add_subdirectory("${_aethermind_source}" "${_aethermind_binary}")
    elseif(NOT _aethermind_present_count EQUAL _aethermind_required_count)
        message(FATAL_ERROR "${_aethermind_name}: only some required dependency targets were provided")
    endif()
    foreach(_aethermind_target IN LISTS _aethermind_arg_TARGETS)
        if(NOT TARGET ${_aethermind_target})
            message(FATAL_ERROR "${_aethermind_name}: dependency did not define ${_aethermind_target}")
        endif()
    endforeach()
    set(${_aethermind_key}_SOURCE_DIR "${_aethermind_source}" PARENT_SCOPE)
    set(${_aethermind_key}_BINARY_DIR "${_aethermind_binary}" PARENT_SCOPE)
    set(${_aethermind_key}_POPULATED TRUE PARENT_SCOPE)
endfunction()
