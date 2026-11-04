cmake_minimum_required(VERSION 3.28)

foreach (required NAME SOURCE REPOSITORY TAG MODE)
    if (NOT DEFINED AETHERMIND_DEPENDENCY_${required} OR
            "${AETHERMIND_DEPENDENCY_${required}}" STREQUAL "")
        message(FATAL_ERROR "Pass AETHERMIND_DEPENDENCY_${required}")
    endif ()
endforeach ()

set(source "${AETHERMIND_DEPENDENCY_SOURCE}")
cmake_path(NORMAL_PATH source)
cmake_path(GET source PARENT_PATH parent)
cmake_path(GET source FILENAME directory)
if (NOT IS_ABSOLUTE "${source}" OR source STREQUAL parent OR
        NOT AETHERMIND_DEPENDENCY_MODE MATCHES "^(ENSURE|VERIFY)$")
    message(FATAL_ERROR "Invalid shared dependency source or acquisition mode")
endif ()

if (NOT DEFINED AETHERMIND_DEPENDENCY_LOCK_TIMEOUT)
    set(AETHERMIND_DEPENDENCY_LOCK_TIMEOUT 600)
endif ()

if (NOT DEFINED AETHERMIND_DEPENDENCY_DOWNLOAD_TIMEOUT)
    set(AETHERMIND_DEPENDENCY_DOWNLOAD_TIMEOUT 180)
endif ()

foreach (option LOCK_TIMEOUT DOWNLOAD_TIMEOUT)
    if (NOT AETHERMIND_DEPENDENCY_${option} MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "AETHERMIND_DEPENDENCY_${option} must be a positive integer")
    endif ()
endforeach ()

set(stage "")
set(stage_owned FALSE)
function(acquisition_failed reason)
    if (stage_owned AND NOT "${stage}" STREQUAL "")
        file(REMOVE_RECURSE "${stage}")
    endif ()
    message(FATAL_ERROR "Dependency ${AETHERMIND_DEPENDENCY_NAME} (${AETHERMIND_DEPENDENCY_TAG}) at ${source}: ${reason}")
endfunction()

file(MAKE_DIRECTORY "${parent}")
# Never unlink lock files: waiters must continue to lock the same inode.
# The operating system releases the advisory lock when this process exits.
file(LOCK "${parent}/.${directory}.lock" GUARD PROCESS
        TIMEOUT "${AETHERMIND_DEPENDENCY_LOCK_TIMEOUT}" RESULT_VARIABLE lock_result)
if (NOT "${lock_result}" STREQUAL "0")
    acquisition_failed("lock acquisition failed: ${lock_result}")
endif ()

find_package(Git QUIET)
if (NOT GIT_FOUND)
    acquisition_failed("Git is required for managed shared sources")
endif ()

function(run_git output)
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env GIT_OPTIONAL_LOCKS=0 GIT_TERMINAL_PROMPT=0
            "${GIT_EXECUTABLE}" ${ARGN}
            RESULT_VARIABLE result OUTPUT_VARIABLE text ERROR_VARIABLE error
            OUTPUT_STRIP_TRAILING_WHITESPACE
            TIMEOUT "${AETHERMIND_DEPENDENCY_DOWNLOAD_TIMEOUT}")
    if (NOT "${result}" STREQUAL "0")
        acquisition_failed("Git failed (${result}): ${error}")
    endif ()
    set(${output} "${text}" PARENT_SCOPE)
endfunction()

run_git(unused check-ref-format "refs/tags/${AETHERMIND_DEPENDENCY_TAG}")

function(source_entries path output)
    # Treat the directory itself literally; glob metacharacters in a checkout
    # path must not hide existing user data from the emptiness check.
    set(pattern "${path}")
    string(REPLACE "[" "[[]" pattern "${pattern}")
    string(REPLACE "?" "[?]" pattern "${pattern}")
    string(REPLACE "*" "[*]" pattern "${pattern}")
    file(GLOB entries LIST_DIRECTORIES TRUE "${pattern}/*" "${pattern}/.*")
    set(${output} "${entries}" PARENT_SCOPE)
endfunction()

function(validate_checkout path)
    if (NOT EXISTS "${path}/CMakeLists.txt")
        acquisition_failed("source has no CMakeLists.txt; existing content was preserved")
    endif ()

    run_git(top -C "${path}" rev-parse --show-toplevel)
    file(REAL_PATH "${path}" expected_top)
    file(REAL_PATH "${top}" actual_top)
    if (NOT actual_top STREQUAL expected_top)
        acquisition_failed("source is not its own Git checkout; existing content was preserved")
    endif ()

    run_git(head -C "${path}" rev-parse --verify HEAD)
    # Resolve the exact local tag, including annotated tags and aliases. Only
    # the tag is configured; no independent commit SHA is maintained or fetched.
    run_git(expected -C "${path}" rev-parse --verify "refs/tags/${AETHERMIND_DEPENDENCY_TAG}^{commit}")
    if (NOT head STREQUAL expected)
        acquisition_failed("tag mismatch: checkout is not at ${AETHERMIND_DEPENDENCY_TAG}; stop affected builds before an explicit upgrade")
    endif ()

    run_git(changes -C "${path}" status --porcelain=v1 --untracked-files=all)
    if (NOT "${changes}" STREQUAL "")
        acquisition_failed("source has local changes; existing content was preserved")
    endif ()
endfunction()

if (IS_SYMLINK "${source}")
    acquisition_failed("managed source is a symbolic link; use an explicit source override instead")
endif ()

if (EXISTS "${source}")
    if (NOT IS_DIRECTORY "${source}")
        acquisition_failed("source path is not a directory; existing content was preserved")
    endif ()

    source_entries("${source}" entries)
    list(LENGTH entries entry_count)
    if (entry_count GREATER 0)
        validate_checkout("${source}")
        return()
    endif ()
endif ()

if (AETHERMIND_DEPENDENCY_MODE STREQUAL "VERIFY")
    acquisition_failed("shared source is missing or empty; configure online to acquire ${AETHERMIND_DEPENDENCY_TAG}")
endif ()

string(RANDOM LENGTH 20 ALPHABET 0123456789abcdef token)
set(stage "${parent}/.${directory}.tmp-${token}")
if (EXISTS "${stage}")
    acquisition_failed("staging path already exists")
endif ()

set(stage_owned TRUE)
message(STATUS "Acquiring ${AETHERMIND_DEPENDENCY_NAME} (${AETHERMIND_DEPENDENCY_TAG}) into ${source}")
run_git(unused clone --no-hardlinks --no-checkout -- "${AETHERMIND_DEPENDENCY_REPOSITORY}" "${stage}")
run_git(unused -C "${stage}" checkout --detach "refs/tags/${AETHERMIND_DEPENDENCY_TAG}" --)
validate_checkout("${stage}")

# ExternalProject may create empty SOURCE_DIR placeholders outside this lock.
# Only empty placeholders may be removed; publication never replaces nonempty data.
foreach (attempt RANGE 1 16)
    if (IS_SYMLINK "${source}")
        acquisition_failed("source became a symbolic link during acquisition")
    endif ()

    if (EXISTS "${source}")
        source_entries("${source}" entries)
        list(LENGTH entries entry_count)
        if (entry_count GREATER 0 OR NOT IS_DIRECTORY "${source}")
            acquisition_failed("source changed during acquisition; existing content was preserved")
        endif ()
        file(REMOVE_RECURSE "${source}")
    endif ()

    # NO_REPLACE directory renames fail on some CMake 3.28 platforms. The
    # per-source lock serializes our writers; directory rename itself refuses
    # an existing nonempty directory.
    file(RENAME "${stage}" "${source}" RESULT result)
    if ("${result}" STREQUAL "0")
        set(stage_owned FALSE)
        return()
    endif ()

    if (NOT IS_DIRECTORY "${source}")
        acquisition_failed("atomic publication failed: ${result}")
    endif ()

    source_entries("${source}" entries)
    list(LENGTH entries entry_count)
    if (entry_count GREATER 0)
        acquisition_failed("atomic publication found nonempty destination: ${result}")
    endif ()
endforeach ()
acquisition_failed("atomic publication repeatedly collided with source placeholders")
