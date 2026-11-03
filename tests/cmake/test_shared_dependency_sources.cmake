cmake_minimum_required(VERSION 3.28)

foreach(required AETHERMIND_SOURCE_DIR AETHERMIND_TEST_BINARY_DIR)
    if(NOT DEFINED ${required} OR NOT IS_ABSOLUTE "${${required}}")
        message(FATAL_ERROR "Pass an absolute ${required}")
    endif()
endforeach()
find_package(Git REQUIRED)
set(root "${AETHERMIND_TEST_BINARY_DIR}/shared dependency contract")
file(REMOVE_RECURSE "${root}")
file(MAKE_DIRECTORY "${root}/origin" "${root}/override source")
set(dependency_project [=[
cmake_minimum_required(VERSION 3.28)
add_library(shared_probe_target INTERFACE)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/variant.txt" "${CMAKE_BUILD_TYPE}")
]=])
file(WRITE "${root}/origin/CMakeLists.txt" "${dependency_project}")
file(WRITE "${root}/override source/CMakeLists.txt" "${dependency_project}")
function(git_command output)
    execute_process(COMMAND "${GIT_EXECUTABLE}" ${ARGN}
            RESULT_VARIABLE result OUTPUT_VARIABLE text ERROR_VARIABLE error
            OUTPUT_STRIP_TRAILING_WHITESPACE TIMEOUT 30)
    if(NOT "${result}" STREQUAL "0")
        message(FATAL_ERROR "Test Git command failed: ${error}")
    endif()
    set(${output} "${text}" PARENT_SCOPE)
endfunction()
git_command(unused -c init.defaultObjectFormat=sha1 init --quiet "${root}/origin")
git_command(unused -C "${root}/origin" add CMakeLists.txt)
git_command(unused -C "${root}/origin" -c user.name=DependencyContract -c user.email=contract@example.invalid
        -c commit.gpgsign=false commit --quiet -m "Local dependency fixture")
set(tag v1.0.0)
git_command(unused -C "${root}/origin" tag "${tag}")
git_command(unused -C "${root}/origin" tag v1-alias)
set(shared "${root}/shared source")
set(fixture "${AETHERMIND_SOURCE_DIR}/tests/cmake/shared_dependency_contract_project")
set(missing_tag v9.9.9-missing)

function(configure_contract name build_type source expected_error)
    execute_process(COMMAND "${CMAKE_COMMAND}"
            -S "${fixture}" -B "${root}/${name}"
            "-DAETHERMIND_SOURCE_DIR=${AETHERMIND_SOURCE_DIR}"
            "-DSHARED_SOURCE=${source}" "-DEXPECTED_SOURCE=${source}"
            "-DTEST_REPOSITORY=${root}/origin" "-DTEST_TAG=${tag}"
            "-DGIT_EXECUTABLE=${GIT_EXECUTABLE}"
            "-DCMAKE_BUILD_TYPE=${build_type}" ${ARGN}
            RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 60)
    file(WRITE "${root}/${name}.log" "${output}\n${error}")
    string(REGEX REPLACE "[ \t\r\n]+" " " diagnostic "${output}\n${error}")
    if(NOT "${expected_error}" STREQUAL "")
        if("${result}" STREQUAL "0" OR NOT diagnostic MATCHES "${expected_error}")
            message(FATAL_ERROR "${name} did not fail as expected (${result}):\n${output}\n${error}")
        endif()
        return()
    elseif(NOT "${result}" STREQUAL "0")
        message(FATAL_ERROR "${name} failed (${result}):\n${output}\n${error}")
    endif()
    if(NOT name STREQUAL "disabled")
        file(READ "${root}/${name}/dependency-binary.txt" binary_dir)
        file(READ "${binary_dir}/variant.txt" variant)
        if(NOT variant STREQUAL build_type)
            message(FATAL_ERROR "${name} reused another build's generated files")
        endif()
    endif()
endfunction()

# Cold acquisition, a different build tree, and disconnected reuse of one tree.
configure_contract(debug Debug "${shared}" "")
file(SHA256 "${shared}/CMakeLists.txt" source_before)
configure_contract(alias Debug "${shared}" "" -DTEST_TAG=v1-alias)
configure_contract(release Release "${shared}" "" "-DTEST_REPOSITORY=${root}/must-not-fetch")
configure_contract(debug RelWithDebInfo "${shared}" ""
        "-DTEST_REPOSITORY=${root}/must-not-fetch" -DFETCHCONTENT_FULLY_DISCONNECTED=ON)
file(READ "${root}/release/dependency-binary.txt" release_binary)
file(READ "${release_binary}/variant.txt" release_variant)
if(NOT release_variant STREQUAL "Release")
    message(FATAL_ERROR "Independent build trees shared generated files")
endif()

# A cached/fully-disconnected configure must reject dirty or mismatched sources
# before any third-party CMakeLists.txt can run.
configure_contract(missing-tag Debug "${shared}" "Git failed" "-DTEST_TAG=${missing_tag}")
file(APPEND "${shared}/CMakeLists.txt" "\nfile(WRITE \"${shared}/must-not-execute\" poisoned)\n")
file(SHA256 "${shared}/CMakeLists.txt" dirty_before)
configure_contract(debug Debug "${shared}" "source has local changes" -DFETCHCONTENT_FULLY_DISCONNECTED=ON)
file(SHA256 "${shared}/CMakeLists.txt" dirty_after)
if(NOT dirty_before STREQUAL dirty_after OR EXISTS "${shared}/must-not-execute")
    message(FATAL_ERROR "Dirty source was overwritten or executed")
endif()
git_command(unused -C "${shared}" checkout -- CMakeLists.txt)
configure_contract(offline-cold Debug "${root}/offline source" "shared source is missing or empty"
        -DFETCHCONTENT_FULLY_DISCONNECTED=ON)

file(MAKE_DIRECTORY "${root}/unknown source")
file(WRITE "${root}/unknown source/CMakeLists.txt" "${dependency_project}\nfile(WRITE \"${root}/unknown source/must-not-execute\" poisoned)\n")
configure_contract(unknown Debug "${root}/unknown source" "Git failed|not its own Git checkout")
if(EXISTS "${root}/unknown source/must-not-execute")
    message(FATAL_ERROR "Unknown source was executed")
endif()

# Paths ending in -NOTFOUND are false constants to CMake's if(). Directory
# emptiness must use list length, or such user data could be mistaken for empty.
file(MAKE_DIRECTORY "${root}/sentinel source")
file(WRITE "${root}/sentinel source/user-data-NOTFOUND" keep)
configure_contract(sentinel Debug "${root}/sentinel source" "source has no CMakeLists.txt")
file(READ "${root}/sentinel source/user-data-NOTFOUND" sentinel_data)
if(NOT sentinel_data STREQUAL "keep" OR EXISTS "${root}/sentinel source/.git")
    message(FATAL_ERROR "False-valued path was treated as an empty source directory")
endif()
file(MAKE_DIRECTORY "${root}/unknown [source]")
file(WRITE "${root}/unknown [source]/user-data" keep)
configure_contract(literal-path Debug "${root}/unknown [source]" "source has no CMakeLists.txt")
file(READ "${root}/unknown [source]/user-data" literal_data)
if(NOT literal_data STREQUAL "keep" OR EXISTS "${root}/unknown [source]/.git")
    message(FATAL_ERROR "Glob metacharacters hid existing source contents")
endif()

# Failed clone/checkout cleans only its own staging directory; retries recover.
configure_contract(failed-clone Debug "${root}/failed source" "Git failed" "-DTEST_REPOSITORY=${root}/missing repository")
file(GLOB leftovers "${root}/.failed source.tmp-*")
if(leftovers OR EXISTS "${root}/failed source/CMakeLists.txt")
    message(FATAL_ERROR "Failed clone left a partially published checkout")
endif()
configure_contract(failed-clone Debug "${root}/failed source" "")
configure_contract(failed-checkout Debug "${root}/checkout source" "Git failed" "-DTEST_TAG=${missing_tag}")
file(GLOB leftovers "${root}/.checkout source.tmp-*")
if(leftovers OR EXISTS "${root}/checkout source/CMakeLists.txt")
    message(FATAL_ERROR "Failed checkout left a partially published checkout")
endif()
configure_contract(failed-checkout Debug "${root}/checkout source" "")

# Missing shared data must be repaired even when this build's stamps are intact.
configure_contract(stamps Debug "${root}/stamp source" "")
configure_contract(stamps Debug "${root}/stamp source" "")
file(REMOVE_RECURSE "${root}/stamp source")
configure_contract(stamps Debug "${root}/stamp source" "")

# Disabling remote updates must still permit restoring a missing checkout.
configure_contract(update-stamps Debug "${root}/update stamp source" "" -DFETCHCONTENT_UPDATES_DISCONNECTED=ON)
configure_contract(update-stamps Debug "${root}/update stamp source" "" -DFETCHCONTENT_UPDATES_DISCONNECTED=ON)
file(REMOVE_RECURSE "${root}/update stamp source")
configure_contract(update-stamps Debug "${root}/update stamp source" "" -DFETCHCONTENT_UPDATES_DISCONNECTED=ON)

# Explicit paths can be non-Git sources. Providers and existing targets must
# bypass acquisition even if the managed path is invalid or the repository fails.
configure_contract(override Debug "${root}/unused override source" ""
        "-DEXPECTED_SOURCE=${root}/override source"
        "-DFETCHCONTENT_SOURCE_DIR_SHARED_PROBE=${root}/override source"
        "-DTEST_REPOSITORY=${root}/must-not-fetch" "-DTEST_TAG=${missing_tag}")
configure_contract(provider Debug "${root}/unknown source" ""
        "-DEXPECTED_SOURCE=${root}/override source"
        "-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=${fixture}/provider.cmake"
        "-DPROVIDER_SOURCE=${root}/override source" "-DTEST_REPOSITORY=${root}/must-not-fetch")
configure_contract(existing Debug "${root}/unused existing source" ""
        -DCONTRACT_CASE=existing "-DEXPECTED_SOURCE=${root}/override source"
        "-DTEST_REPOSITORY=${root}/must-not-fetch")
configure_contract(earlier Debug "${root}/unused earlier source" ""
        -DCONTRACT_CASE=earlier "-DEXPECTED_SOURCE=${root}/override source"
        "-DTEST_REPOSITORY=${root}/must-not-fetch")
configure_contract(earlier-shared Debug "${root}/unused earlier shared source" ""
        -DCONTRACT_CASE=earlier-shared "-DEXPECTED_SOURCE=${root}/parent managed source")
file(WRITE "${root}/provider unused file" "preserve provider bypass")
configure_contract(provider-file Debug "${root}/provider unused file" ""
        "-DEXPECTED_SOURCE=${root}/override source"
        "-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=${fixture}/provider.cmake"
        "-DPROVIDER_SOURCE=${root}/override source" "-DTEST_REPOSITORY=${root}/must-not-fetch")
file(READ "${root}/provider unused file" provider_data)
if(NOT provider_data STREQUAL "preserve provider bypass")
    message(FATAL_ERROR "Provider bypass modified the unused managed source path")
endif()
configure_contract(disabled Debug "${root}/disabled source" "" -DCONTRACT_CASE=disabled)
if(EXISTS "${root}/disabled source" OR EXISTS "${root}/.disabled source.lock")
    message(FATAL_ERROR "Disabled dependency was acquired")
endif()
file(SHA256 "${shared}/CMakeLists.txt" source_after)
if(NOT source_before STREQUAL source_after OR EXISTS "${shared}/variant.txt")
    message(FATAL_ERROR "Shared source contents were modified")
endif()

# Annotated tags and multiple tags on one revision must both be supported.
# A clean checkout at the wrong tag must be rejected before CMake code runs.
file(APPEND "${root}/origin/CMakeLists.txt" [=[

if(DEFINED REVISION_MARKER)
    file(WRITE "${REVISION_MARKER}" configured)
endif()
]=])
git_command(unused -C "${root}/origin" add CMakeLists.txt)
git_command(unused -C "${root}/origin" -c user.name=DependencyContract -c user.email=contract@example.invalid
        -c commit.gpgsign=false commit --quiet -m "Second fixture revision")
git_command(unused -C "${root}/origin" -c user.name=DependencyContract -c user.email=contract@example.invalid
        -c tag.gpgsign=false tag --annotate v2.0.0 -m "Annotated fixture version")
git_command(unused -C "${shared}" fetch "${root}/origin" refs/tags/v2.0.0:refs/tags/v2.0.0)
git_command(unused -C "${shared}" checkout --detach refs/tags/v2.0.0)
configure_contract(debug Debug "${shared}" "tag mismatch" -DFETCHCONTENT_FULLY_DISCONNECTED=ON
        "-DREVISION_MARKER=${root}/wrong-tag-was-executed")
if(EXISTS "${root}/wrong-tag-was-executed")
    message(FATAL_ERROR "Wrong tag was configured before validation")
endif()
git_command(unused -C "${shared}" checkout --detach "refs/tags/${tag}")
configure_contract(annotated Debug "${root}/annotated source" "" -DTEST_TAG=v2.0.0
        "-DREVISION_MARKER=${root}/annotated-was-configured")
if(NOT EXISTS "${root}/annotated-was-configured")
    message(FATAL_ERROR "Annotated version tag was not accepted")
endif()

# Two independent configure processes must serialize a cold acquisition. Git's
# trace records the actual clone count without introducing a test-only fetcher.
set(parallel_source "${root}/parallel source")
set(trace "${root}/parallel-git.trace")
set(common
        "-DFIXTURE=${fixture}"
        "-DAETHERMIND_SOURCE_DIR=${AETHERMIND_SOURCE_DIR}"
        "-DSHARED_SOURCE=${parallel_source}" "-DEXPECTED_SOURCE=${parallel_source}"
        "-DTEST_REPOSITORY=${root}/origin" "-DTEST_TAG=${tag}"
        "-DGIT_EXECUTABLE=${GIT_EXECUTABLE}")
file(WRITE "${root}/parallel-configure.cmake" [=[
cmake_minimum_required(VERSION 3.28)
# execute_process pipelines run concurrently, but neither launcher writes to
# that pipe: a fast peer must not close stdout and SIGPIPE the other configure.
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${FIXTURE}" -B "${BINARY}"
        "-DAETHERMIND_SOURCE_DIR=${AETHERMIND_SOURCE_DIR}"
        "-DSHARED_SOURCE=${SHARED_SOURCE}" "-DEXPECTED_SOURCE=${EXPECTED_SOURCE}"
        "-DTEST_REPOSITORY=${TEST_REPOSITORY}" "-DTEST_TAG=${TEST_TAG}"
        "-DGIT_EXECUTABLE=${GIT_EXECUTABLE}" "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
        RESULT_VARIABLE result OUTPUT_FILE "${BINARY}.log" ERROR_FILE "${BINARY}.error.log" TIMEOUT 55)
if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "Parallel configure failed (${result}); see ${BINARY}.log and ${BINARY}.error.log")
endif()
]=])
execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env "GIT_TRACE=${trace}"
            "${CMAKE_COMMAND}" ${common} "-DBINARY=${root}/parallel-debug" -DBUILD_TYPE=Debug
            -P "${root}/parallel-configure.cmake"
        COMMAND "${CMAKE_COMMAND}" -E env "GIT_TRACE=${trace}"
            "${CMAKE_COMMAND}" ${common} "-DBINARY=${root}/parallel-release" -DBUILD_TYPE=Release
            -P "${root}/parallel-configure.cmake"
        RESULTS_VARIABLE results OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 60)
file(WRITE "${root}/parallel.log" "${output}\n${error}")
foreach(result IN LISTS results)
    if(NOT "${result}" STREQUAL "0")
        message(FATAL_ERROR "Concurrent configure failed (${results}):\n${output}\n${error}")
    endif()
endforeach()
file(STRINGS "${trace}" clones REGEX "built-in: git clone ")
list(LENGTH clones clone_count)
if(NOT clone_count EQUAL 1)
    message(FATAL_ERROR "Concurrent configure performed ${clone_count} clones; expected exactly one")
endif()
foreach(variant debug release)
    file(READ "${root}/parallel-${variant}/dependency-binary.txt" binary_dir)
    file(READ "${binary_dir}/variant.txt" actual_variant)
    string(TOLOWER "${actual_variant}" actual_variant)
    if(NOT actual_variant STREQUAL variant)
        message(FATAL_ERROR "Concurrent builds shared generated dependency state")
    endif()
endforeach()

# A crashed writer's staging directory is not a published checkout. Do not
# delete another process's staging data: use a fresh unique directory instead.
file(MAKE_DIRECTORY "${root}/.interrupted source.tmp-abandoned")
file(WRITE "${root}/.interrupted source.tmp-abandoned/CMakeLists.txt" "not a published checkout")
configure_contract(interrupted Debug "${root}/interrupted source" "")
if(NOT EXISTS "${root}/.interrupted source.tmp-abandoned/CMakeLists.txt")
    message(FATAL_ERROR "Acquisition deleted staging data it did not create")
endif()

# Test a real lock timeout and process-exit release, not just lock-file presence.
set(lock "${root}/.parallel source.lock")
set(held "${root}/lock-held")
file(WRITE "${root}/hold-lock.cmake" [=[
cmake_minimum_required(VERSION 3.28)
file(LOCK "${LOCK}" GUARD PROCESS TIMEOUT 5)
file(WRITE "${HELD}" held)
execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 3)
message(FATAL_ERROR "Intentional holder termination")
]=])
file(WRITE "${root}/lock-contender.cmake" [=[
cmake_minimum_required(VERSION 3.28)
foreach(attempt RANGE 1 150)
    if(EXISTS "${HELD}")
        break()
    endif()
    execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 0.02)
endforeach()
if(NOT EXISTS "${HELD}")
    message(FATAL_ERROR "Lock holder did not start")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}"
        -DAETHERMIND_DEPENDENCY_NAME=shared_probe
        "-DAETHERMIND_DEPENDENCY_SOURCE=${SOURCE}"
        "-DAETHERMIND_DEPENDENCY_REPOSITORY=${REPOSITORY}"
        "-DAETHERMIND_DEPENDENCY_TAG=${TAG}"
        -DAETHERMIND_DEPENDENCY_MODE=VERIFY -DAETHERMIND_DEPENDENCY_LOCK_TIMEOUT=1
        -P "${SCRIPT}"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 10)
if("${result}" STREQUAL "0" OR NOT "${output}${error}" MATCHES "lock acquisition failed")
    message(FATAL_ERROR "Contender did not report the expected lock timeout: ${output}${error}")
endif()
]=])
execute_process(
        COMMAND "${CMAKE_COMMAND}" "-DLOCK=${lock}" "-DHELD=${held}" -P "${root}/hold-lock.cmake"
        COMMAND "${CMAKE_COMMAND}" "-DHELD=${held}" "-DSOURCE=${parallel_source}"
            "-DREPOSITORY=${root}/origin" "-DTAG=${tag}"
            "-DSCRIPT=${AETHERMIND_SOURCE_DIR}/cmake/AcquireSharedDependency.cmake"
            -P "${root}/lock-contender.cmake"
        RESULTS_VARIABLE results OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 15)
file(WRITE "${root}/lock.log" "${output}\n${error}")
if(NOT "${results}" STREQUAL "1;0")
    message(FATAL_ERROR "Lock lifetime contract failed (${results}):\n${output}\n${error}")
endif()
configure_contract(after-lock Debug "${parallel_source}" "")
message(STATUS "Cold/shared acquisition, concurrency, validation, recovery, precedence and lock lifetime passed")
