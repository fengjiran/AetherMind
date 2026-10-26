include_guard(GLOBAL)
cmake_policy(PUSH)
cmake_policy(SET CMP0057 NEW)

function(aethermind_configure_test_options)
    set(AETHERMIND_TEST_MODELS_DIR "${PROJECT_SOURCE_DIR}/.models" CACHE PATH
        "Root of the pinned, externally prepared HF test models")
    set(AETHERMIND_INFER_BINARY "" CACHE FILEPATH
        "Absolute external inference executable; the in-build target takes precedence")
    if(NOT IS_ABSOLUTE "${AETHERMIND_TEST_MODELS_DIR}")
        message(FATAL_ERROR "AETHERMIND_TEST_MODELS_DIR must be an absolute path")
    endif()
    if(AETHERMIND_INFER_BINARY AND NOT IS_ABSOLUTE "${AETHERMIND_INFER_BINARY}")
        message(FATAL_ERROR "AETHERMIND_INFER_BINARY must be an absolute path")
    endif()
endfunction()

# These cases require the glibc wrappers, rather than merely testing an OOM
# status with a custom allocator. Keep their identities explicit so sanitizer
# runs retain ordinary failure-recovery coverage.
function(aethermind_test_labels name output)
    set(labels unit)
    set(interposer_cases
        AMError.ObjectAllocationFailureReturnsNull
        AMError.MessageAllocationFailureReleasesObject
        CAbiRuntimeAllocationFailure.FailureDoesNotPublishHandleAndCanRetry
        CAbiRuntime.RegistryAllocationFailureReturnsResourceExhausted
        CAbiRuntime.ErrorObjectAllocationFailureLeavesDiagnosticUnavailable
        CAbiRuntime.ErrorMessageAllocationFailureReleasesDiagnosticObject
        CAbiModel.AllocationFailureDoesNotPublishModelAndRuntimeCanRetry
        CAbiModel.PackedPreparationAllocationFailureReleasesStorageAndCanRetry
        CAbiSession.AllocationFailureDoesNotPublishSessionAndCanRetry
        CAbiGenerate.ShortOutputThenRequestAllocationFailureProvesNoExecution
        CAbiGenerate.RequestAllocationFailureClearsLengthAndCanRetry
        CAbiGenerate.DiagnosticMessageAllocationFailureKeepsOriginalStatusAndUpperBound
        WorkspaceRequirementPlanning.SuccessfulPlanningDoesNotAllocate
        WorkspaceRequirementPlanningDiagnosticFailure.ReturnsResourceExhaustedAndPreservesOffsets
        OffsetAlignment.SuccessfulAlignmentDoesNotAllocate
        OffsetAlignmentDiagnosticFailure.ReturnsResourceExhaustedWithEmptyMessage
        ParseNumber.FallbackAllocationFailurePreservesOutputAndCanRetry
        KVCacheManagerInitAllocationFailure.LeavesEmptyStateAndCanRetry
        KVCacheManager.InitDiagnosticAllocationFailureLeavesEmptyStateAndCanRetry
        KVCacheManager.BufferMetadataFailureReturnsResourceExhaustedAndReleasesPayload
        KVCacheManager.OccupiedReserveDiagnosticFailurePreservesReservation
        KVCacheManager.ForeignResetDiagnosticFailurePreservesBothReservations
        KVCacheManager.ForeignReleaseDiagnosticFailurePreservesBothReservations
        KVCacheCommitTest.CommitDiagnosticFailurePreservesTheEntireState
        KVCacheCommitTest.StaleCommitDiagnosticFailurePreservesTheEntireState
        KVCacheCommitTest.MalformedReservationDiagnosticFailurePreservesTheEntireState
        KVCacheCommitTest.InvalidLayoutDiagnosticFailurePreservesTheEntireState
        TinyLlamaExecution.GenerateReturnsResourceExhaustedForRequestBufferAllocationFailure
        TinyLlamaExecution.GenerateReturnsResourceExhaustedForBindingAllocationFailure
        TinyLlamaExecution.GenerateReturnsResourceExhaustedForActivationMetadataFailure
        TinyLlamaExecution.GenerateReturnsResourceExhaustedForPrefillAndDecodeActivationFailure
        TinyLlamaExecution.SharedDecodeLoopHasNoMallocFamilyCallsAfterPreparation
        TinyWorkspaceSession.GenerateReturnsResourceExhaustedForWorkspaceAllocationFailure
        DirectPrefillDecode.MallocInterposerObservesCpuAllocatorCalls
        AlignedCpuBuffer.RejectsNonPowerOfTwoAlignmentBeforeAllocating
        AlignedCpuBuffer.SystemAllocationFailureReturnsStatusAndCanRetry
        AlignedCpuBuffer.MetadataFailureReleasesPayloadAndCanRetry
        CPUAllocatorDeathTest.HugePageWarningOutOfMemoryReleasesStorage
        CPUAllocator.SystemAllocationFailureThrowsBadAllocAndCanRetry
        Buffer.MetadataAllocationFailurePropagatesAndReleasesHandle
        RuntimeBuilderTryBuild.AllocatorProviderTransferOutOfMemoryStaysReusable
        RuntimeBuilderTryBuild.BackendFactoryTransferOutOfMemoryStaysReusable
        CpuWeightPacking.BufferMetadataFailureReturnsResourceExhaustedAndReleasesPayload
        CpuWeightPackingMetadataFailure.ReturnsResourceExhaustedAndReleasesPayload
        PackedWeightCollectionOwnership.AllocationFailureReleasesArtifactAndDoesNotBindSource
        WeightPacking.PrepackMetadataAllocationFailureReturnsResourceExhaustedBeforeBackendCall
    )
    foreach(case IN LISTS interposer_cases)
        string(REPLACE "." "\\." pattern "${case}")
        string(REPLACE "." "/[0-9]+\\." typed_pattern "${case}")
        if(name MATCHES "(^|/)${pattern}(/[^/]*)?(<[^>]*>)?$" OR
           name MATCHES "(^|/)${typed_pattern}$")
            list(APPEND labels interposer)
            break()
        endif()
    endforeach()
    if(name MATCHES "(^|/)KVCacheSessionViewOomTest\\.")
        list(APPEND labels interposer)
    endif()

    if(name MATCHES "(^|/)(ModelLoader_HfRealModelIntegrationTest|InferenceAdmissionFixtures?|Fp32HfAdmission|Fp32HfCompile|RunnerFp32Admission|RunnerKvAllocationFailure|InferenceRunnerEquivalence|InferenceRunnerSuccess|InferenceRunnerFailure|CAbiModel|CAbiSession|CAbiGenerate|CAbiKVValidation|CAbiRuntimeIntegration|LoadAndPrepareExecutableModel(Path)?|InferenceCliIntegration)\\." OR
       (name MATCHES "^ModelCompiler\\." AND NOT name MATCHES "^ModelCompiler\\.RejectsNullLoadedModel$"))
        list(APPEND labels model)
    endif()
    if(name MATCHES "(^|/)InferenceCliIntegration\\.")
        list(APPEND labels cli)
    endif()
    if(name MATCHES "(^|/)(CpuInfo|CpuCapabilities|CpuBackendFactory|InferenceRunOptions|InferenceRuntimeOptions|InferenceCpuPreset|CAbiRuntime|MallocInterposerContract|GlobalFunctionTable|FunctionRegistryOwnership)\\." OR
       name MATCHES "(TinyGenerateMatchesScalarOracle|LoadsPreparesAndGeneratesFromHfDirectory)")
        list(APPEND labels smoke)
    endif()
    if(name MATCHES "(ScalarOracle|CpuLinear|CpuRmsNorm|CpuAttention|CpuRoPE|CpuArgmax|AttentionKernel|RmsNormKernel|LinearKernel|RoPEKernel|ArgmaxKernel)" OR
       name MATCHES "(^|/)CPUKernel(Argmax|Attention|Linear|RmsNorm|RoPE|AddRmsNorm|GateUpLinear|QkvLinear|MatMul|GemmReference|GemmScalar|DotProduct|Silu)")
        list(APPEND labels numerical)
    endif()
    if(name MATCHES "(KVCache|Prefill|ExecutionContext|PreparedExecutionBindings|KernelInvoker|RuntimeBuilder|Allocator|Buffer\\.|CAbi|InferenceRunnerFailure|TinyLlamaExecution|TinyWorkspaceSession|GlobalFunctionTable|FunctionRegistryOwnership)")
        list(APPEND labels lifecycle)
    endif()
    if(name MATCHES "(Capacity|ContextLimit|Reservation|ContextTokens)")
        list(APPEND labels context_boundary)
    endif()
    if(name MATCHES "(Avx|AVX|PackedB)")
        list(APPEND labels isa-avx2)
    endif()
    if(name MATCHES "^(ThreadBasicTest|ThreadParamTest|ThreadMemberTest|ThreadMoveTest|JThreadBasicTest|JThreadStopTest)\\.")
        list(APPEND labels thread-smoke)
    endif()
    # A bounded vertical slice for the validation presets, not every unit case.
    if(name MATCHES "(^|/)(TinyLlamaExecution|TinyWorkspaceSession|DirectPrefillDecode|Prefill|PrefillInvalidToken|KVCacheCommitTest|KVCacheSessionView|KVCacheInvalidStateTest|KernelInvoker|ExecutionContext|RuntimeBuilderTryBuild|LoadAndPrepareExecutableModel(Path)?|InferenceRunnerFailure|InferenceRunnerEquivalence|InferenceCliIntegration|CAbiRuntimeIntegration|MallocInterposerContract|GlobalFunctionTable|FunctionRegistryOwnership)\\." OR
       name MATCHES "SlidingWindow" OR "interposer" IN_LIST labels)
        list(APPEND labels validation)
    endif()
    list(REMOVE_DUPLICATES labels)
    set(${output} "${labels}" PARENT_SCOPE)
endfunction()

function(aethermind_set_discovered_test_properties tests_variable sanitizer models_ready cli_ready interposer_ready required_assets)
    if(ARGC GREATER 7 AND NOT EXISTS "${ARGV7}")
        # PRE_TEST discovery emits this placeholder when the executable has not
        # been built. It must participate in the filtered validation entrypoint.
        set_tests_properties("${ARGV6}_NOT_BUILT" PROPERTIES LABELS "validation;build")
        return()
    elseif(NOT DEFINED ${tests_variable} OR "${${tests_variable}}" STREQUAL "")
        add_test(aethermind.gtest_empty "${ARGV8}" -E false)
        set_tests_properties(aethermind.gtest_empty PROPERTIES LABELS "validation;build")
        return()
    endif()
    foreach(name IN LISTS ${tests_variable})
        aethermind_test_labels("${name}" labels)
        set(reason "")
        if("interposer" IN_LIST labels AND (NOT sanitizer STREQUAL "none" OR NOT interposer_ready))
            set(reason unavailable_interposer)
        elseif("model" IN_LIST labels AND NOT models_ready AND NOT required_assets)
            set(reason missing_assets)
        elseif("cli" IN_LIST labels AND NOT cli_ready AND NOT required_assets)
            set(reason missing_cli)
        endif()
        if(reason)
            list(APPEND labels "${reason}")
            set_tests_properties("${name}" PROPERTIES DISABLED TRUE)
        endif()
        if(ARGV9)
            set_tests_properties("${name}" PROPERTIES LABELS "${labels}" TIMEOUT 60
                WORKING_DIRECTORY "${ARGV9}")
        else()
            set_tests_properties("${name}" PROPERTIES LABELS "${labels}" TIMEOUT 60)
        endif()
        if("validation" IN_LIST labels AND NOT reason)
            # CTest appends environment entries, retaining the CLI path supplied
            # by discovery. GTest converts required skips before printing status.
            set_tests_properties("${name}" PROPERTIES ENVIRONMENT "AETHERMIND_REQUIRED_VALIDATION=1")
        endif()
        if("model" IN_LIST labels)
            set_tests_properties("${name}" PROPERTIES FIXTURES_REQUIRED aethermind_model_assets)
        endif()
    endforeach()
endfunction()

function(aethermind_register_tests target)
    include(GoogleTest)
    include(CheckCXXSymbolExists)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        check_cxx_symbol_exists(__GLIBC__ "features.h" AETHERMIND_TEST_HAS_GLIBC)
    endif()
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(AETHERMIND_VALIDATION_REQUIRED_ASSETS AND NOT Python3_Interpreter_FOUND)
        message(FATAL_ERROR "Required validation assets need a Python3 interpreter for checksum verification")
    endif()

    set(models_ready TRUE)
    foreach(file config.json model.safetensors)
        if(NOT EXISTS "${AETHERMIND_TEST_MODELS_DIR}/tiny-random-LlamaForCausalLM/${file}")
            set(models_ready FALSE)
        endif()
    endforeach()
    file(READ "${PROJECT_SOURCE_DIR}/tests/fixtures/inference_admission/checksums.json" checksums)
    string(JSON variant_count LENGTH "${checksums}" variants)
    math(EXPR last_variant "${variant_count} - 1")
    foreach(index RANGE ${last_variant})
        string(JSON variant MEMBER "${checksums}" variants ${index})
        foreach(file config.json model.safetensors)
            if(NOT EXISTS "${AETHERMIND_TEST_MODELS_DIR}/inference-admission/${variant}/${file}")
                set(models_ready FALSE)
            endif()
        endforeach()
    endforeach()
    if(NOT Python3_Interpreter_FOUND)
        set(models_ready FALSE)
    endif()

    set(cli_ready FALSE)
    set(cli_binary "${AETHERMIND_INFER_BINARY}")
    if(TARGET aethermind_infer)
        set(cli_binary "$<TARGET_FILE:aethermind_infer>")
        set(cli_ready TRUE)
    elseif(cli_binary AND EXISTS "${cli_binary}")
        set(cli_ready TRUE)
    endif()
    if(AETHERMIND_VALIDATION_REQUIRED_ASSETS AND NOT cli_ready)
        message(FATAL_ERROR "Required validation needs BUILD_INFERENCE_TOOL=ON or an existing absolute AETHERMIND_INFER_BINARY")
    endif()

    # CMake 4.3 writes the PRE_TEST discovery JSON into WORKING_DIRECTORY, so
    # discover from the build tree. The discovered tests still run from the
    # repository root, restored by aethermind_set_discovered_test_properties.
    # A WORKING_DIRECTORY entry under PROPERTIES is not an option here: CMake
    # parses it as this keyword and overrides the discovery directory.
    gtest_discover_tests(${target}
        DISCOVERY_MODE PRE_TEST
        DISCOVERY_TIMEOUT 60
        NO_PRETTY_TYPES NO_PRETTY_VALUES
        TEST_LIST AETHERMIND_DISCOVERED_TESTS
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        PROPERTIES
            ENVIRONMENT "AETHERMIND_INFER_BINARY=${cli_binary}"
    )
    set(properties_script "${CMAKE_CURRENT_BINARY_DIR}/AetherMindTestProperties.cmake")
    set(properties_content
        "include(\"${PROJECT_SOURCE_DIR}/cmake/AetherMindTests.cmake\")\naethermind_set_discovered_test_properties(AETHERMIND_DISCOVERED_TESTS \"${AETHERMIND_SANITIZER}\" \"${models_ready}\" \"${cli_ready}\" \"${AETHERMIND_TEST_HAS_GLIBC}\" \"${AETHERMIND_VALIDATION_REQUIRED_ASSETS}\" \"${target}\" \"$<TARGET_FILE:${target}>\" \"${CMAKE_COMMAND}\" \"${PROJECT_SOURCE_DIR}\")\n")
    if(CMAKE_CONFIGURATION_TYPES)
        foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
            file(GENERATE OUTPUT "${properties_script}-${config}" CONTENT "${properties_content}" CONDITION "$<CONFIG:${config}>")
        endforeach()
        file(WRITE "${properties_script}" "include(\"${properties_script}-\${CTEST_CONFIGURATION_TYPE}\")\n")
    else()
        file(GENERATE OUTPUT "${properties_script}" CONTENT "${properties_content}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "${properties_script}")

    add_test(NAME cmake.test_classification COMMAND "${CMAKE_COMMAND}"
        -DAETHERMIND_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P "${PROJECT_SOURCE_DIR}/tests/cmake/test_test_classification.cmake")
    set_tests_properties(cmake.test_classification PROPERTIES LABELS "smoke;validation;build" TIMEOUT 30)
    add_test(NAME cmake.validation_skip_policy COMMAND "${CMAKE_COMMAND}"
        -DAETHERMIND_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -DAETHERMIND_CTEST_COMMAND=${CMAKE_CTEST_COMMAND}
        -DAETHERMIND_TEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
        -DAETHERMIND_REQUIRED_EXECUTION_PROBE=$<TARGET_FILE:aethermind_required_execution_probe>
        -P "${PROJECT_SOURCE_DIR}/tests/cmake/test_ctest_skip_policy.cmake")
    set_tests_properties(cmake.validation_skip_policy PROPERTIES LABELS "smoke;validation;build" TIMEOUT 30)

    if(Python3_Interpreter_FOUND)
        add_test(NAME python.fixture_converter COMMAND "${Python3_EXECUTABLE}" -m unittest discover
            -s "${PROJECT_SOURCE_DIR}/tests/unit/apps/inference"
            -p test_generate_fp32_admission_hf_fixtures.py -v)
        set_tests_properties(python.fixture_converter PROPERTIES LABELS "smoke;validation;fixture" TIMEOUT 60)
        add_test(NAME fixture.model_assets COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/tools/generate_fp32_admission_hf_fixtures.py"
            --source "${AETHERMIND_TEST_MODELS_DIR}/tiny-random-LlamaForCausalLM"
            --recipe "${PROJECT_SOURCE_DIR}/tests/fixtures/inference_admission/recipe.json"
            --output-dir "${AETHERMIND_TEST_MODELS_DIR}/inference-admission"
            --check "${PROJECT_SOURCE_DIR}/tests/fixtures/inference_admission/checksums.json")
    else()
        add_test(NAME python.fixture_converter COMMAND "${CMAKE_COMMAND}" -E false)
        set_tests_properties(python.fixture_converter PROPERTIES LABELS "fixture;missing_python" DISABLED TRUE)
        add_test(NAME fixture.model_assets COMMAND "${CMAKE_COMMAND}" -E false)
    endif()
    set_tests_properties(fixture.model_assets PROPERTIES LABELS fixture
        FIXTURES_SETUP aethermind_model_assets TIMEOUT 120)
    if(NOT models_ready AND NOT AETHERMIND_VALIDATION_REQUIRED_ASSETS)
        set_tests_properties(fixture.model_assets PROPERTIES DISABLED TRUE LABELS "fixture;missing_assets")
    endif()

    file(STRINGS "${PROJECT_SOURCE_DIR}/tests/unit/apps/inference/test_inference_cli.py"
        cli_methods REGEX "^    def test_[A-Za-z0-9_]+\\(self\\):")
    foreach(method IN LISTS cli_methods)
        string(REGEX REPLACE "^    def (test_[A-Za-z0-9_]+)\\(self\\):.*" "\\1" method "${method}")
        set(name "python.cli.${method}")
        if(Python3_Interpreter_FOUND)
            add_test(NAME "${name}" COMMAND "${Python3_EXECUTABLE}"
                "${PROJECT_SOURCE_DIR}/tests/unit/apps/inference/test_inference_cli.py"
                --infer "${cli_binary}" --fixtures "${AETHERMIND_TEST_MODELS_DIR}/inference-admission"
                "InferenceCliTests.${method}" -v)
        else()
            add_test(NAME "${name}" COMMAND "${CMAKE_COMMAND}" -E false)
        endif()
        set(labels "cli;model;validation;python")
        set(reason "")
        if(method MATCHES "^(test_logger_initialization_allocation_failure_precedes_model_loading|test_real_library_warning_uses_stderr_without_polluting_json)$")
            list(APPEND labels preload)
            if(NOT AETHERMIND_SANITIZER STREQUAL "none" OR NOT AETHERMIND_TEST_HAS_GLIBC)
                set(reason unavailable_preload)
            endif()
        endif()
        if(NOT Python3_Interpreter_FOUND)
            set(reason missing_python)
        elseif(NOT models_ready AND NOT AETHERMIND_VALIDATION_REQUIRED_ASSETS)
            set(reason missing_assets)
        elseif(NOT cli_ready AND NOT AETHERMIND_VALIDATION_REQUIRED_ASSETS)
            set(reason missing_cli)
        endif()
        if(reason)
            list(APPEND labels "${reason}")
            set_tests_properties("${name}" PROPERTIES DISABLED TRUE)
        endif()
        set_tests_properties("${name}" PROPERTIES LABELS "${labels}" TIMEOUT 120
            FIXTURES_REQUIRED aethermind_model_assets WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    endforeach()
endfunction()

cmake_policy(POP)
