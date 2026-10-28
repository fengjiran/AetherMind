include_guard(GLOBAL)
cmake_policy(PUSH)
cmake_policy(SET CMP0057 NEW)

# Purpose: provide a single trusted Python3 interpreter for build-time and
# CTest consumers. CTest consumers are stdlib-only scripts, so this module
# does not probe torch/transformers-style third-party environments, nor does
# it call `uv python find` (deliberately: detection covers venvs and the
# system search only; the "uv-managed interpreter without a project venv"
# scenario is not supported).
#
# Resolution priority - every candidate that is adopted is probe-verified
# (runnable + version). Explicit inputs are exactly two:
#   - AETHERMIND_PYTHON_EXECUTABLE (module-owned cache option)
#   - user pin: -DPython3_EXECUTABLE (FindPython3 cache entry)
# A parent-scope *normal* variable Python3_EXECUTABLE is hidden at entry (it
# may be a leftover from an upstream project's earlier find_package) and is
# not an input. Explicit-input failure handling:
#   AETHERMIND_PYTHON_EXECUTABLE: FATAL_ERROR when REQUIRED, otherwise a
#     WARNING followed by fallback to lower-priority sources.
#   User pin, when adopted (i.e. no valid AETHERMIND_PYTHON_EXECUTABLE):
#     invalid / below MIN_VERSION / not absolute -> always FATAL_ERROR. The
#     pin cache entry would block FindPython3's system fallback, and a
#     graceful fallback would have to destroy the user's pin, so fallback is
#     rejected.
#   1. AETHERMIND_PYTHON_EXECUTABLE (overrides the pin, keeps the pin cache)
#   2. user pin: Python3_EXECUTABLE cache entry
#   3. active virtual environment     $ENV{VIRTUAL_ENV}
#   4. uv project environment         $ENV{UV_PROJECT_ENVIRONMENT}
#                                     (relative paths resolve against the
#                                     project root)
#   5. in-tree virtual environment    ${PROJECT_SOURCE_DIR}/.venv (the
#                                     subproject root in subproject builds)
#   6. system interpreter             find_package(Python3) default search
# Auto-detection (3-5) is skipped while either explicit input (1/2) is set.
# A final defensive version assertion after find_package prevents any future
# hint path from bypassing MIN_VERSION (FindPython3 does not validate the
# requested version for explicit hints).
#
# Sticky semantics (inherent to FindPython3): the resolution is persisted in
# _Python3_EXECUTABLE:INTERNAL, and configure runs without a candidate reuse
# the first result. This module attributes sources via an INTERNAL pair-cache
# (AETHERMIND_PYTHON_RESOLVED + AETHERMIND_PYTHON_SOURCE_LAST): an unchanged
# resolved path keeps the previous source label, a changed path recomputes.
# Switching interpreters:
#   - -DAETHERMIND_PYTHON_EXECUTABLE=<path> or -DPython3_EXECUTABLE=<path>
#   - or clear the Python3-related cache entries and reconfigure
#
# Outputs: propagates the consumed variables Python3_EXECUTABLE /
# Python3_Interpreter_FOUND / Python3_VERSION[_MAJOR|_MINOR|_PATCH], plus
# AETHERMIND_PYTHON_FOUND and AETHERMIND_PYTHON_SOURCE
# (custom | cached | virtualenv | uv-project-env | project-venv | system).
# Additional variables such as Python3_INTERPRETER_ID can be propagated here
# if consumers ever need them.
# A disabled optional lookup returns false FOUND flags and empty result values;
# a disabled REQUIRED lookup retains find_package's configuration error.

function(_aethermind_probe_python probe_exe out_ok out_ver out_err)
    execute_process(COMMAND "${probe_exe}" -c
            "import sys; sys.stdout.write('{}.{}.{}'.format(*sys.version_info[:3]))"
            OUTPUT_VARIABLE _probe_ver OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_VARIABLE _probe_err RESULT_VARIABLE _probe_rc TIMEOUT 15)
    set(${out_ok} FALSE PARENT_SCOPE)
    set(${out_ver} "" PARENT_SCOPE)
    set(${out_err} "" PARENT_SCOPE)
    if (_probe_rc EQUAL 0)
        set(${out_ok} TRUE PARENT_SCOPE)
        set(${out_ver} "${_probe_ver}" PARENT_SCOPE)
    elseif ("${_probe_err}" STREQUAL "")
        set(${out_err} "probe exited with code ${_probe_rc}" PARENT_SCOPE)
    else ()
        set(${out_err} "${_probe_err}" PARENT_SCOPE)
    endif ()
endfunction()

function(aethermind_find_python)
    set(options REQUIRED)
    set(one_value_args MIN_VERSION)
    cmake_parse_arguments(AFP "${options}" "${one_value_args}" "" ${ARGN})

    # CMake's regex engine lacks {m,n} quantifiers; use optional groups to
    # limit the version to at most three segments.
    if (AFP_MIN_VERSION)
        if (NOT AFP_MIN_VERSION MATCHES "^[0-9]+(\\.[0-9]+)?(\\.[0-9]+)?$")
            message(FATAL_ERROR
                    "aethermind_find_python: MIN_VERSION must be numeric 'X[.Y[.Z]]', got '${AFP_MIN_VERSION}'")
        endif ()
    endif ()

    if (CMAKE_DISABLE_FIND_PACKAGE_Python3)
        # Let CMake enforce REQUIRED (including CMAKE_REQUIRE_FIND_PACKAGE_*),
        # then discard results inherited from an earlier parent lookup.
        if (AFP_REQUIRED)
            find_package(Python3 COMPONENTS Interpreter REQUIRED)
        else ()
            find_package(Python3 COMPONENTS Interpreter QUIET)
        endif ()
        
        foreach (result Python3_EXECUTABLE Python3_VERSION Python3_VERSION_MAJOR
                Python3_VERSION_MINOR Python3_VERSION_PATCH AETHERMIND_PYTHON_SOURCE)
            set(${result} "" PARENT_SCOPE)
        endforeach ()
        set(Python3_Interpreter_FOUND FALSE PARENT_SCOPE)
        set(AETHERMIND_PYTHON_FOUND FALSE PARENT_SCOPE)
        return()
    endif ()

    # Hide a parent-scope normal-variable hint (possibly a leftover from an
    # upstream project's earlier find_package): FindPython3 inherits it as an
    # executable hint and does not validate the requested version. Only the
    # cache pin counts as an explicit input; it is read via $CACHE{} and is
    # unaffected by the unset below.
    unset(Python3_EXECUTABLE)
    if (NOT AFP_MIN_VERSION)
        unset(Python3_FIND_VERSION)    # likewise keep a stray version constraint from leaking in
    endif ()

    set(AETHERMIND_PYTHON_EXECUTABLE "" CACHE FILEPATH
            "Path to a specific Python interpreter (empty = auto-detect)")
    # The pair-cache is not initialized here: an INTERNAL set(... CACHE ...)
    # overwrites its value on every run. The FORCE writes in step 6 create it,
    # and until then it reads as undefined (expands to empty) in step 6.

    set(_candidate "")
    set(_source "system")
    set(_python_pin "$CACHE{Python3_EXECUTABLE}")
    # Interactive FindPython3 caches failed searches as NOTFOUND. Keep that
    # search result available to FindPython3 without treating it as a user pin.
    if (_python_pin MATCHES "(^|-)NOTFOUND$")
        set(_python_pin "")
    endif ()

    # 1) Explicit custom interpreter: the authoritative entry, never fails
    #    silently.
    if (AETHERMIND_PYTHON_EXECUTABLE)
        if (NOT IS_ABSOLUTE "${AETHERMIND_PYTHON_EXECUTABLE}")
            message(FATAL_ERROR "AETHERMIND_PYTHON_EXECUTABLE must be an absolute path")
        endif ()

        _aethermind_probe_python("${AETHERMIND_PYTHON_EXECUTABLE}" _ok _ver _err)
        if (NOT _ok)
            if (AFP_REQUIRED)
                message(FATAL_ERROR "AETHERMIND_PYTHON_EXECUTABLE"
                        "'${AETHERMIND_PYTHON_EXECUTABLE}' is not a runnable Python interpreter: ${_err}")
            endif ()
            message(WARNING "AETHERMIND_PYTHON_EXECUTABLE"
                    "'${AETHERMIND_PYTHON_EXECUTABLE}' is not runnable (${_err});"
                    " falling back to auto-detection")
        elseif (AFP_MIN_VERSION AND _ver VERSION_LESS "${AFP_MIN_VERSION}")
            if (AFP_REQUIRED)
                message(FATAL_ERROR "AETHERMIND_PYTHON_EXECUTABLE reports Python ${_ver},"
                        " below required ${AFP_MIN_VERSION}")
            endif ()
            message(WARNING "AETHERMIND_PYTHON_EXECUTABLE reports Python ${_ver},"
                    " below required ${AFP_MIN_VERSION}; falling back to auto-detection")
        else ()
            set(_candidate "${AETHERMIND_PYTHON_EXECUTABLE}")
            set(_source "custom")
        endif ()
    endif ()

    # 2) User pin: explicit input outranks auto-detection; validated when
    #    adopted, failures do not fall back (see header).
    if (NOT _candidate AND NOT "${_python_pin}" STREQUAL "")
        if (NOT IS_ABSOLUTE "${_python_pin}")
            message(FATAL_ERROR "Pinned Python3_EXECUTABLE must be an absolute path, got '${_python_pin}'")
        endif ()
        _aethermind_probe_python("${_python_pin}" _ok _ver _err)
        if (NOT _ok)
            message(FATAL_ERROR
                    "Pinned Python3_EXECUTABLE '${_python_pin}' is not a runnable Python interpreter: ${_err}")
        elseif (AFP_MIN_VERSION AND _ver VERSION_LESS "${AFP_MIN_VERSION}")
            message(FATAL_ERROR
                    "Pinned Python3_EXECUTABLE reports Python ${_ver}, below required ${AFP_MIN_VERSION};"
                    " unset it or pass -DAETHERMIND_PYTHON_EXECUTABLE with a compliant interpreter")
        endif ()
        set(_candidate "${_python_pin}")
        set(_source "cached")
    endif ()

    # 3) Pure filesystem venv probing, only while no explicit input is set.
    #    Absolute-normalization must not resolve symlinks (venv bin/python is
    #    one): REAL_PATH would swap in the system interpreter and lose
    #    sys.prefix.
    if (NOT _candidate)
        set(_venv_roots)
        set(_venv_sources)
        if (DEFINED ENV{VIRTUAL_ENV} AND NOT "$ENV{VIRTUAL_ENV}" STREQUAL "")
            set(_venv_dir "$ENV{VIRTUAL_ENV}")
            cmake_path(ABSOLUTE_PATH _venv_dir BASE_DIRECTORY "${PROJECT_SOURCE_DIR}"
                    NORMALIZE OUTPUT_VARIABLE _venv_dir)
            if (NOT _venv_dir IN_LIST _venv_roots)
                list(APPEND _venv_roots "${_venv_dir}")
                list(APPEND _venv_sources "virtualenv")
            endif ()
        endif ()

        if (DEFINED ENV{UV_PROJECT_ENVIRONMENT} AND NOT "$ENV{UV_PROJECT_ENVIRONMENT}" STREQUAL "")
            set(_venv_dir "$ENV{UV_PROJECT_ENVIRONMENT}")
            cmake_path(ABSOLUTE_PATH _venv_dir BASE_DIRECTORY "${PROJECT_SOURCE_DIR}"
                    NORMALIZE OUTPUT_VARIABLE _venv_dir)
            if (NOT _venv_dir IN_LIST _venv_roots)
                list(APPEND _venv_roots "${_venv_dir}")
                list(APPEND _venv_sources "uv-project-env")
            endif ()
        endif ()

        set(_venv_dir "${PROJECT_SOURCE_DIR}/.venv")
        cmake_path(ABSOLUTE_PATH _venv_dir NORMALIZE OUTPUT_VARIABLE _venv_dir)
        if (NOT _venv_dir IN_LIST _venv_roots)
            list(APPEND _venv_roots "${_venv_dir}")
            list(APPEND _venv_sources "project-venv")
        endif ()

        list(LENGTH _venv_roots _n)
        math(EXPR _last "${_n} - 1")
        foreach (_i RANGE ${_last})
            list(GET _venv_roots ${_i} _root)
            foreach (_rel "bin/python" "Scripts/python.exe")
                if (EXISTS "${_root}/${_rel}")
                    _aethermind_probe_python("${_root}/${_rel}" _ok _ver _err)
                    if (_ok AND (NOT AFP_MIN_VERSION OR NOT _ver VERSION_LESS "${AFP_MIN_VERSION}"))
                        set(_candidate "${_root}/${_rel}")
                        list(GET _venv_sources ${_i} _source)
                        break()
                    endif ()
                endif ()
            endforeach ()
            if (_candidate)
                break()
            endif ()
        endforeach ()
    endif ()

    # 4) Candidate submission. When custom overrides a non-empty user pin only
    #    a normal-variable hint is set (no unset): the pin stays cached and
    #    takes effect again the next time no candidate wins.
    if (_candidate)
        if (_source STREQUAL "custom"
                AND NOT "${_python_pin}" STREQUAL ""
                AND NOT "${_python_pin}" STREQUAL "${_candidate}")
            message(WARNING
                    "AETHERMIND_PYTHON_EXECUTABLE '${_candidate}' takes precedence over pinned"
                    " Python3_EXECUTABLE '${_python_pin}' for this configure;"
                    " the pin is preserved in the cache")
        endif ()
        set(Python3_EXECUTABLE "${_candidate}")
    endif ()

    # 5) Unified validation and caching. The version goes through the
    #    find_package version argument (semantics: >=).
    if (AFP_MIN_VERSION)
        if (AFP_REQUIRED)
            find_package(Python3 ${AFP_MIN_VERSION} COMPONENTS Interpreter REQUIRED)
        else ()
            find_package(Python3 ${AFP_MIN_VERSION} COMPONENTS Interpreter QUIET)
        endif ()
    else ()
        if (AFP_REQUIRED)
            find_package(Python3 COMPONENTS Interpreter REQUIRED)
        else ()
            find_package(Python3 COMPONENTS Interpreter QUIET)
        endif ()
    endif ()

    # Defensive version gate: FindPython3 does not validate the requested
    # version for explicit hints, so no future hint path may bypass
    # MIN_VERSION. Normal paths never trigger this assertion.
    if (Python3_Interpreter_FOUND AND AFP_MIN_VERSION AND Python3_VERSION VERSION_LESS "${AFP_MIN_VERSION}")
        message(FATAL_ERROR "aethermind_find_python: resolved Python ${Python3_VERSION}"
                " (${Python3_EXECUTABLE}) below required ${AFP_MIN_VERSION}")
    endif ()

    # 6) Sticky-reuse attribution: without a candidate, an unchanged resolved
    #    path keeps the previous source label. (Pins and venv hits exist as
    #    _candidate, so they never enter this branch.)
    if (Python3_Interpreter_FOUND AND NOT _candidate
            AND NOT "${AETHERMIND_PYTHON_RESOLVED}" STREQUAL ""
            AND "${Python3_EXECUTABLE}" STREQUAL "${AETHERMIND_PYTHON_RESOLVED}")
        set(_source "${AETHERMIND_PYTHON_SOURCE_LAST}")
    endif ()

    if (Python3_Interpreter_FOUND)
        set(AETHERMIND_PYTHON_RESOLVED "${Python3_EXECUTABLE}" CACHE INTERNAL
                "Last resolved interpreter; sticky-reuse source attribution" FORCE)
        set(AETHERMIND_PYTHON_SOURCE_LAST "${_source}" CACHE INTERNAL
                "Source label of the last resolved interpreter" FORCE)
    endif ()

    # 7) Result hand-off: find_package's normal variables land in this
    #    function scope and must be propagated explicitly.
    set(Python3_EXECUTABLE "${Python3_EXECUTABLE}" PARENT_SCOPE)
    set(Python3_Interpreter_FOUND "${Python3_Interpreter_FOUND}" PARENT_SCOPE)
    set(Python3_VERSION "${Python3_VERSION}" PARENT_SCOPE)
    set(Python3_VERSION_MAJOR "${Python3_VERSION_MAJOR}" PARENT_SCOPE)
    set(Python3_VERSION_MINOR "${Python3_VERSION_MINOR}" PARENT_SCOPE)
    set(Python3_VERSION_PATCH "${Python3_VERSION_PATCH}" PARENT_SCOPE)
    set(AETHERMIND_PYTHON_FOUND "${Python3_Interpreter_FOUND}" PARENT_SCOPE)
    if (NOT Python3_Interpreter_FOUND)
        set(_source "")
    endif ()
    set(AETHERMIND_PYTHON_SOURCE "${_source}" PARENT_SCOPE)

    if (Python3_Interpreter_FOUND)
        message(STATUS "Python3: ${Python3_EXECUTABLE} (version ${Python3_VERSION}, source=${_source})")
    endif ()
endfunction()

cmake_policy(POP)
