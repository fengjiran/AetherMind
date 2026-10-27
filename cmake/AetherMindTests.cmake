include_guard(GLOBAL)
cmake_policy(PUSH)
cmake_policy(SET CMP0057 NEW)

function(aethermind_configure_test_options)
    set(AETHERMIND_TEST_MODELS_DIR "${PROJECT_SOURCE_DIR}/.models" CACHE PATH
            "Root of the pinned, externally prepared HF test models")
    if (DEFINED AETHERMIND_TEST_NUMERICAL_DIR AND NOT IS_ABSOLUTE "${AETHERMIND_TEST_NUMERICAL_DIR}")
        message(FATAL_ERROR "AETHERMIND_TEST_NUMERICAL_DIR must be an absolute path")
    endif ()
    set(AETHERMIND_TEST_NUMERICAL_DIR "${AETHERMIND_TEST_MODELS_DIR}/inference-numerical" CACHE PATH
            "Root of the reviewed, offline-generated HF numerical reference bundle")
    set(AETHERMIND_INFER_BINARY "" CACHE FILEPATH
            "Absolute external inference executable; the in-build target takes precedence")

    if (NOT IS_ABSOLUTE "${AETHERMIND_TEST_MODELS_DIR}")
        message(FATAL_ERROR "AETHERMIND_TEST_MODELS_DIR must be an absolute path")
    endif ()

    if (AETHERMIND_INFER_BINARY AND NOT IS_ABSOLUTE "${AETHERMIND_INFER_BINARY}")
        message(FATAL_ERROR "AETHERMIND_INFER_BINARY must be an absolute path")
    endif ()
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
    foreach (case IN LISTS interposer_cases)
        string(REPLACE "." "\\." pattern "${case}")
        string(REPLACE "." "/[0-9]+\\." typed_pattern "${case}")
        if (name MATCHES "(^|/)${pattern}(/[^/]*)?(<[^>]*>)?$" OR
                name MATCHES "(^|/)${typed_pattern}$")
            list(APPEND labels interposer)
            break()
        endif ()
    endforeach ()

    if (name MATCHES "(^|/)KVCacheSessionViewOomTest\\.")
        list(APPEND labels interposer)
    endif ()

    string(CONCAT model_pattern
            "(^|/)(ModelLoader_HfRealModelIntegrationTest|InferenceAdmissionFixtures?|"
            "Fp32HfAdmission|Fp32HfCompile|RunnerFp32Admission|RunnerKvAllocationFailure|"
            "InferenceRunnerEquivalence|InferenceRunnerSuccess|InferenceRunnerFailure|"
            "CAbiModel|CAbiSession|CAbiGenerate|CAbiKVValidation|CAbiRuntimeIntegration|"
            "LoadAndPrepareExecutableModel(Path)?|InferenceCliIntegration)\\.")
    if (name MATCHES "${model_pattern}" OR
    (name MATCHES "^ModelCompiler\\." AND NOT name MATCHES "^ModelCompiler\\.RejectsNullLoadedModel$"))
        list(APPEND labels model)
    endif ()

    if (name MATCHES "(^|/)InferenceCliIntegration\\.")
        list(APPEND labels cli)
    endif ()

    string(CONCAT smoke_pattern
            "(^|/)(CpuInfo|CpuCapabilities|CpuBackendFactory|"
            "InferenceRunOptions|InferenceRuntimeOptions|InferenceCpuPreset|CAbiRuntime|"
            "MallocInterposerContract|GlobalFunctionTable|FunctionRegistryOwnership)\\.")
    if (name MATCHES "${smoke_pattern}" OR
            name MATCHES "(TinyGenerateMatchesScalarOracle|LoadsPreparesAndGeneratesFromHfDirectory)")
        list(APPEND labels smoke)
    endif ()

    string(CONCAT numerical_pattern
            "(ScalarOracle|CpuLinear|CpuRmsNorm|CpuAttention|CpuRoPE|CpuArgmax|"
            "AttentionKernel|RmsNormKernel|LinearKernel|RoPEKernel|ArgmaxKernel)")
    string(CONCAT cpu_kernel_pattern
            "(^|/)CPUKernel(Argmax|Attention|Linear|RmsNorm|RoPE|AddRmsNorm|"
            "GateUpLinear|QkvLinear|MatMul|GemmReference|GemmScalar|DotProduct|Silu)")
    if (name MATCHES "${numerical_pattern}" OR
            name MATCHES "${cpu_kernel_pattern}")
        list(APPEND labels numerical)
    endif ()

    if (name MATCHES "(^|/)(HfFixedTrajectory|HfNearContextLimit|HfNumericalFaultInjection)\\.")
        list(APPEND labels model numerical numerical-model)
    endif ()

    if (name MATCHES "(^|/)(HfFixedTrajectory|HfNumericalFaultInjection|HfNumericalComparison)\\.")
        list(APPEND labels numerical validation)
    endif ()

    if (name MATCHES "(^|/)HfNearContextLimit\\.")
        list(APPEND labels long-context)
    endif ()

    string(CONCAT lifecycle_pattern
            "(KVCache|Prefill|ExecutionContext|PreparedExecutionBindings|KernelInvoker|RuntimeBuilder|"
            "Allocator|Buffer\\.|CAbi|InferenceRunnerFailure|TinyLlamaExecution|TinyWorkspaceSession|"
            "GlobalFunctionTable|FunctionRegistryOwnership)")
    if (name MATCHES "${lifecycle_pattern}")
        list(APPEND labels lifecycle)
    endif ()

    if (name MATCHES "(Capacity|ContextLimit|Reservation|ContextTokens)")
        list(APPEND labels context_boundary)
    endif ()

    # These suites call the AVX2/FMA translation units directly. Metadata and
    # feature-policy tests must remain runnable in a baseline-only build.
    string(CONCAT avx_suite_pattern
            "(^|/)(CPUKernelAvxLearning|CPUKernelGemmAvx2|CPUKernelGemmAvx2FastPathTest|"
            "CPUKernelGemmAvx2SmallMTest|CPUKernelGemmAvx2BlockedTest|CpuGemmPackedB|"
            "CpuQkvLinearPackedB|CpuGateUpLinearPackedB)\\.")
    string(CONCAT avx_rms_norm_pattern
            "^CPUKernelRmsNorm\\.(Avx2FmaPreparedKernelMatchesDoubleReferenceWithTail|"
            "Avx2FmaExactInPlaceWithTail)$")
    if (name MATCHES "${avx_suite_pattern}" OR
            name MATCHES "^CPUKernelDotProduct\\.(Avx2CoreMatchesGolden|MultiTargetAvx2CoreMatchesGolden)$" OR
            name MATCHES "${avx_rms_norm_pattern}" OR
            name MATCHES "^CPUKernelRmsNormEntry\\.Avx2RejectsNonUnitColumnStridesWhenAvailable$")
        list(APPEND labels isa-avx2)
    endif ()

    if (name MATCHES "(^|/)(CpuIsaFallback|CpuIsaFallbackPolicy)\\.")
        list(APPEND labels validation isa-policy)
    endif ()

    string(CONCAT thread_pattern
            "^(ThreadBasicTest|ThreadParamTest|ThreadMemberTest|ThreadMoveTest|"
            "JThreadBasicTest|JThreadStopTest)\\.")
    if (name MATCHES "${thread_pattern}")
        list(APPEND labels thread-smoke)
    endif ()

    # A bounded vertical slice for the validation presets, not every unit case.
    string(CONCAT validation_pattern
            "(^|/)(TinyLlamaExecution|TinyWorkspaceSession|DirectPrefillDecode|Prefill|"
            "PrefillInvalidToken|KVCacheCommitTest|KVCacheSessionView|KVCacheInvalidStateTest|"
            "KernelInvoker|ExecutionContext|RuntimeBuilderTryBuild|"
            "LoadAndPrepareExecutableModel(Path)?|InferenceRunnerFailure|InferenceRunnerEquivalence|"
            "InferenceCliIntegration|CAbiRuntimeIntegration|MallocInterposerContract|"
            "GlobalFunctionTable|FunctionRegistryOwnership)\\.")
    if (name MATCHES "${validation_pattern}" OR
            name MATCHES "SlidingWindow" OR "interposer" IN_LIST labels)
        list(APPEND labels validation)
    endif ()

    list(REMOVE_DUPLICATES labels)
    set(${output} "${labels}" PARENT_SCOPE)
endfunction()

function(aethermind_set_discovered_test_properties
        tests_variable sanitizer models_ready cli_ready interposer_ready required_assets)
    if (ARGC GREATER 7 AND NOT EXISTS "${ARGV7}")
        # PRE_TEST discovery emits this placeholder when the executable has not
        # been built. It must participate in the filtered validation entrypoint.
        set_tests_properties("${ARGV6}_NOT_BUILT" PROPERTIES LABELS "validation;build")
        add_test(aethermind.gtest_isa_not_built "${ARGV8}" -E false)
        set_tests_properties(aethermind.gtest_isa_not_built PROPERTIES LABELS "isa-avx2;build")
        add_test(aethermind.gtest_numerical_not_built "${ARGV8}" -E false)
        set_tests_properties(aethermind.gtest_numerical_not_built PROPERTIES LABELS "numerical-model;build")
        return()
    elseif (NOT DEFINED ${tests_variable} OR "${${tests_variable}}" STREQUAL "")
        add_test(aethermind.gtest_empty "${ARGV8}" -E false)
        set_tests_properties(aethermind.gtest_empty PROPERTIES LABELS "validation;build")
        add_test(aethermind.gtest_isa_empty "${ARGV8}" -E false)
        set_tests_properties(aethermind.gtest_isa_empty PROPERTIES LABELS "isa-avx2;build")
        add_test(aethermind.gtest_numerical_empty "${ARGV8}" -E false)
        set_tests_properties(aethermind.gtest_numerical_empty PROPERTIES LABELS "numerical-model;build")
        return()
    endif ()
    set(isa_case_count 0)
    set(numerical_case_count 0)
    foreach (name IN LISTS ${tests_variable})
        aethermind_test_labels("${name}" labels)
        set(reason "")
        if ("numerical-model" IN_LIST labels AND NOT ARGV11 AND NOT required_assets)
            set(reason missing_numerical_assets)
        elseif ("isa-avx2" IN_LIST labels AND NOT ARGV10)
            set(reason not_compiled_avx2)
        elseif ("interposer" IN_LIST labels AND (NOT sanitizer STREQUAL "none" OR NOT interposer_ready))
            set(reason unavailable_interposer)
        elseif ("model" IN_LIST labels AND NOT "numerical-model" IN_LIST labels AND
                NOT models_ready AND NOT required_assets)
            set(reason missing_assets)
        elseif ("cli" IN_LIST labels AND NOT cli_ready AND NOT required_assets)
            set(reason missing_cli)
        endif ()
        if (reason)
            list(APPEND labels "${reason}")
            if (reason STREQUAL "not_compiled_avx2")
                list(APPEND labels not_applicable)
            endif ()
            set_tests_properties("${name}" PROPERTIES DISABLED TRUE)
        endif ()
        if (ARGV9)
            set_tests_properties("${name}" PROPERTIES LABELS "${labels}" TIMEOUT 60
                    WORKING_DIRECTORY "${ARGV9}")
        else ()
            set_tests_properties("${name}" PROPERTIES LABELS "${labels}" TIMEOUT 60)
        endif ()
        if (("validation" IN_LIST labels OR "isa-avx2" IN_LIST labels OR
                "numerical-model" IN_LIST labels) AND NOT reason)
            # CTest appends environment entries, retaining the CLI path supplied
            # by discovery. GTest converts required skips before printing status.
            set_tests_properties("${name}" PROPERTIES ENVIRONMENT "AETHERMIND_REQUIRED_VALIDATION=1")
        endif ()
        if ("numerical-model" IN_LIST labels)
            set_tests_properties("${name}" PROPERTIES FIXTURES_REQUIRED aethermind_numerical_assets)
            math(EXPR numerical_case_count "${numerical_case_count} + 1")
            if ("long-context" IN_LIST labels)
                set_tests_properties("${name}" PROPERTIES TIMEOUT 600)
            endif ()
        elseif ("model" IN_LIST labels)
            set_tests_properties("${name}" PROPERTIES FIXTURES_REQUIRED aethermind_model_assets)
        endif ()
        if ("isa-avx2" IN_LIST labels)
            set_tests_properties("${name}" PROPERTIES FIXTURES_REQUIRED aethermind_avx2_host)
            if (NOT reason)
                math(EXPR isa_case_count "${isa_case_count} + 1")
            endif ()
        endif ()
    endforeach ()

    if (ARGV10 AND isa_case_count EQUAL 0)
        # Host capability alone is not execution evidence for a candidate.
        add_test(aethermind.avx2_cases_empty "${ARGV8}" -E false)
        set_tests_properties(aethermind.avx2_cases_empty PROPERTIES LABELS "isa-avx2;build")
    endif ()

    if (ARGC GREATER 11 AND numerical_case_count EQUAL 0)
        add_test(aethermind.numerical_cases_empty "${ARGV8}" -E false)
        set_tests_properties(aethermind.numerical_cases_empty PROPERTIES LABELS "numerical-model;build")
    endif ()
endfunction()

function(aethermind_register_tests target)
    include(GoogleTest)
    include(CheckCXXSymbolExists)
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        check_cxx_symbol_exists(__GLIBC__ "features.h" AETHERMIND_TEST_HAS_GLIBC)
    endif ()

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (AETHERMIND_VALIDATION_REQUIRED_ASSETS AND NOT Python3_Interpreter_FOUND)
        message(FATAL_ERROR "Required validation assets need a Python3 interpreter for checksum verification")
    endif ()

    # Presence only: the independent checker owns manifest/provenance/tensor
    # validation. These assets are separate from the FP32 admission fixtures.
    set(numerical_expected "${PROJECT_SOURCE_DIR}/tests/fixtures/inference_numerical/checksums.json")
    set(numerical_ready TRUE)
    if (NOT Python3_Interpreter_FOUND OR NOT EXISTS "${numerical_expected}" OR
            NOT EXISTS "${AETHERMIND_TEST_NUMERICAL_DIR}/manifest.json")
        set(numerical_ready FALSE)
    endif ()

    foreach (case mha-untied mha-tied gqa-tail mqa-tail mha-near-limit)
        foreach (file model/config.json model/model.safetensors reference.safetensors reference.json)
            if (NOT EXISTS "${AETHERMIND_TEST_NUMERICAL_DIR}/${case}/${file}")
                set(numerical_ready FALSE)
            endif ()
        endforeach ()
    endforeach ()

    if (Python3_Interpreter_FOUND)
        add_test(NAME fixture.numerical_assets COMMAND "${Python3_EXECUTABLE}"
                "${PROJECT_SOURCE_DIR}/tools/verify_llama_numerical_reference.py"
                --root "${AETHERMIND_TEST_NUMERICAL_DIR}" --expected "${numerical_expected}")
        add_test(NAME python.numerical_reference_contract COMMAND "${Python3_EXECUTABLE}"
                "${PROJECT_SOURCE_DIR}/tests/unit/inference/test_verify_llama_numerical_reference.py" -v)
    else ()
        add_test(NAME fixture.numerical_assets COMMAND "${CMAKE_COMMAND}" -E false)
        add_test(NAME python.numerical_reference_contract COMMAND "${CMAKE_COMMAND}" -E false)
        set_tests_properties(python.numerical_reference_contract PROPERTIES DISABLED TRUE)
    endif ()

    set_tests_properties(fixture.numerical_assets PROPERTIES LABELS "fixture;numerical"
            FIXTURES_SETUP aethermind_numerical_assets TIMEOUT 120)
    set_tests_properties(python.numerical_reference_contract PROPERTIES
            LABELS "validation;numerical;fixture" TIMEOUT 60)

    if (NOT Python3_Interpreter_FOUND)
        set_tests_properties(python.numerical_reference_contract PROPERTIES
                LABELS "validation;numerical;fixture;missing_python")
    endif ()

    if (NOT numerical_ready AND NOT AETHERMIND_VALIDATION_REQUIRED_ASSETS)
        set_tests_properties(fixture.numerical_assets PROPERTIES
                DISABLED TRUE LABELS "fixture;numerical;missing_numerical_assets")
    endif ()

    if (AETHERMIND_HAS_AVX2_FMA)
        set(expected_avx2 ON)
    else ()
        set(expected_avx2 OFF)
    endif ()

    if (Python3_Interpreter_FOUND)
        add_test(NAME fixture.cpu_isa_build COMMAND "${Python3_EXECUTABLE}"
                "${PROJECT_SOURCE_DIR}/tools/verify_cpu_isa_build.py"
                --source-dir "${PROJECT_SOURCE_DIR}" --build-dir "${PROJECT_BINARY_DIR}"
                --expect-avx2 "${expected_avx2}")
    else ()
        add_test(NAME fixture.cpu_isa_build COMMAND "${CMAKE_COMMAND}" -E false)
    endif ()

    set_tests_properties(fixture.cpu_isa_build PROPERTIES LABELS "validation;build;isa-build"
            FIXTURES_SETUP aethermind_cpu_isa_build TIMEOUT 30)
    if (NOT Python3_Interpreter_FOUND)
        set_tests_properties(fixture.cpu_isa_build PROPERTIES
                DISABLED TRUE LABELS "validation;build;isa-build;missing_python")
    elseif (NOT CMAKE_EXPORT_COMPILE_COMMANDS AND NOT AETHERMIND_VALIDATION_REQUIRED_ASSETS)
        set_tests_properties(fixture.cpu_isa_build PROPERTIES
                DISABLED TRUE LABELS "validation;build;isa-build;not_applicable;missing_compile_commands")
    endif ()

    # The probe itself uses baseline-safe code. Requiring this fixture prevents
    # a direct candidate call before real hardware/OS capability is established.
    add_executable(aethermind_avx2_host_probe EXCLUDE_FROM_ALL
            "${PROJECT_SOURCE_DIR}/tests/cmake/test_avx2_host_probe.cpp")
    target_link_libraries(aethermind_avx2_host_probe PRIVATE ${PROJECT_NAME})
    target_compile_definitions(aethermind_avx2_host_probe PRIVATE
            AETHERMIND_AVX2_KERNELS_COMPILED=$<BOOL:${AETHERMIND_HAS_AVX2_FMA}>)
    aethermind_apply_sanitizer(aethermind_avx2_host_probe)
    add_dependencies(${target} aethermind_avx2_host_probe)
    add_test(NAME fixture.avx2_host COMMAND aethermind_avx2_host_probe)
    set_tests_properties(fixture.avx2_host PROPERTIES LABELS "isa-avx2;build"
            FIXTURES_SETUP aethermind_avx2_host FIXTURES_REQUIRED aethermind_cpu_isa_build TIMEOUT 30)
    if (NOT AETHERMIND_ENABLE_AVX2)
        set_tests_properties(fixture.avx2_host PROPERTIES
                DISABLED TRUE LABELS "isa-avx2;build;not_applicable;not_compiled_avx2")
    endif ()

    set(models_ready TRUE)
    foreach (file config.json model.safetensors)
        if (NOT EXISTS "${AETHERMIND_TEST_MODELS_DIR}/tiny-random-LlamaForCausalLM/${file}")
            set(models_ready FALSE)
        endif ()
    endforeach ()
    file(READ "${PROJECT_SOURCE_DIR}/tests/fixtures/inference_admission/checksums.json" checksums)
    string(JSON variant_count LENGTH "${checksums}" variants)
    math(EXPR last_variant "${variant_count} - 1")
    foreach (index RANGE ${last_variant})
        string(JSON variant MEMBER "${checksums}" variants ${index})
        foreach (file config.json model.safetensors)
            if (NOT EXISTS "${AETHERMIND_TEST_MODELS_DIR}/inference-admission/${variant}/${file}")
                set(models_ready FALSE)
            endif ()
        endforeach ()
    endforeach ()
    if (NOT Python3_Interpreter_FOUND)
        set(models_ready FALSE)
    endif ()

    set(cli_ready FALSE)
    set(cli_binary "${AETHERMIND_INFER_BINARY}")
    if (TARGET aethermind_infer)
        set(cli_binary "$<TARGET_FILE:aethermind_infer>")
        set(cli_ready TRUE)
    elseif (cli_binary AND EXISTS "${cli_binary}")
        set(cli_ready TRUE)
    endif ()
    if (AETHERMIND_VALIDATION_REQUIRED_ASSETS AND NOT cli_ready)
        message(FATAL_ERROR
                "Required validation needs BUILD_INFERENCE_TOOL=ON or an existing absolute AETHERMIND_INFER_BINARY")
    endif ()

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
    string(CONCAT properties_content
            "include(\"${PROJECT_SOURCE_DIR}/cmake/AetherMindTests.cmake\")\n"
            "aethermind_set_discovered_test_properties("
            "AETHERMIND_DISCOVERED_TESTS "
            "\"${AETHERMIND_SANITIZER}\" "
            "\"${models_ready}\" "
            "\"${cli_ready}\" "
            "\"${AETHERMIND_TEST_HAS_GLIBC}\" "
            "\"${AETHERMIND_VALIDATION_REQUIRED_ASSETS}\" "
            "\"${target}\" "
            "\"$<TARGET_FILE:${target}>\" "
            "\"${CMAKE_COMMAND}\" "
            "\"${PROJECT_SOURCE_DIR}\" "
            "\"${AETHERMIND_HAS_AVX2_FMA}\" "
            "\"${numerical_ready}\")\n")
    if (CMAKE_CONFIGURATION_TYPES)
        foreach (config IN LISTS CMAKE_CONFIGURATION_TYPES)
            file(GENERATE
                    OUTPUT "${properties_script}-${config}"
                    CONTENT "${properties_content}"
                    CONDITION "$<CONFIG:${config}>")
        endforeach ()
        file(WRITE "${properties_script}" "include(\"${properties_script}-\${CTEST_CONFIGURATION_TYPE}\")\n")
    else ()
        file(GENERATE OUTPUT "${properties_script}" CONTENT "${properties_content}")
    endif ()
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

    set(build_contract_config "${CMAKE_CURRENT_BINARY_DIR}/AetherMindBuildContractConfig-$<CONFIG>.cmake")
    string(CONCAT build_contract_content
            "set(AETHERMIND_SOURCE_DIR [==[${PROJECT_SOURCE_DIR}]==])\n"
            "set(AETHERMIND_CONTRACT_BINARY_DIR [==[${CMAKE_CURRENT_BINARY_DIR}/build contracts]==])\n"
            "set(AETHERMIND_SPDLOG_SOURCE_DIR [==[${spdlog_SOURCE_DIR}]==])\n"
            "set(AETHERMIND_GENERATOR [==[${CMAKE_GENERATOR}]==])\n"
            "set(AETHERMIND_GENERATOR_PLATFORM [==[${CMAKE_GENERATOR_PLATFORM}]==])\n"
            "set(AETHERMIND_GENERATOR_TOOLSET [==[${CMAKE_GENERATOR_TOOLSET}]==])\n"
            "set(AETHERMIND_BUILD_CONFIG [==[$<CONFIG>]==])\n")
    foreach (variable CMAKE_MAKE_PROGRAM CMAKE_TOOLCHAIN_FILE CMAKE_C_COMPILER CMAKE_CXX_COMPILER
            CMAKE_C_FLAGS CMAKE_CXX_FLAGS CMAKE_SYSROOT CMAKE_C_COMPILER_TARGET CMAKE_CXX_COMPILER_TARGET)
        string(APPEND build_contract_content "set(AETHERMIND_${variable} [==[${${variable}}]==])\n")
    endforeach ()
    file(GENERATE OUTPUT "${build_contract_config}" CONTENT "${build_contract_content}")
    foreach (case alignment std fmt existing-std existing-fmt)
        set(name "cmake.build_contract.${case}")
        add_test(NAME "${name}" COMMAND "${CMAKE_COMMAND}"
                "-DAETHERMIND_CONTRACT_CONFIG=${build_contract_config}"
                "-DAETHERMIND_CONTRACT_CASE=${case}"
                -P "${PROJECT_SOURCE_DIR}/tests/cmake/test_build_contract.cmake")
        set_tests_properties("${name}" PROPERTIES LABELS "smoke;validation;build" TIMEOUT 600 RUN_SERIAL TRUE)
        if (NOT case STREQUAL "alignment" AND NOT EXISTS "${spdlog_SOURCE_DIR}/CMakeLists.txt")
            set_tests_properties("${name}" PROPERTIES DISABLED TRUE LABELS "build;missing_dependency_source")
        endif ()
    endforeach ()

    if (Python3_Interpreter_FOUND)
        add_test(NAME python.model_benchmark_runner_contract COMMAND "${Python3_EXECUTABLE}"
                "${PROJECT_SOURCE_DIR}/tests/unit/inference/test_run_model_benchmark.py" -v)
        set_tests_properties(python.model_benchmark_runner_contract PROPERTIES
                LABELS "python;smoke;validation;performance" TIMEOUT 60)
    else ()
        add_test(NAME python.model_benchmark_runner_contract COMMAND "${CMAKE_COMMAND}" -E false)
        set_tests_properties(python.model_benchmark_runner_contract PROPERTIES
                DISABLED TRUE LABELS "python;smoke;validation;performance;missing_python")
    endif ()

    if (Python3_Interpreter_FOUND)
        add_test(NAME python.model_benchmark_report_contract COMMAND "${Python3_EXECUTABLE}"
                "${PROJECT_SOURCE_DIR}/tests/unit/inference/test_verify_model_benchmark.py" -v)
        set_tests_properties(python.model_benchmark_report_contract PROPERTIES
                LABELS "python;smoke;validation;performance" TIMEOUT 60)
    else ()
        add_test(NAME python.model_benchmark_report_contract COMMAND "${CMAKE_COMMAND}" -E false)
        set_tests_properties(python.model_benchmark_report_contract PROPERTIES
                DISABLED TRUE LABELS "python;smoke;validation;performance;missing_python")
    endif ()

    if (Python3_Interpreter_FOUND)
        add_test(NAME python.cpu_isa_build_contract COMMAND "${Python3_EXECUTABLE}"
                "${PROJECT_SOURCE_DIR}/tests/unit/cmake/test_verify_cpu_isa_build.py" -v)
        set_tests_properties(python.cpu_isa_build_contract PROPERTIES LABELS "validation;build;isa-build" TIMEOUT 30)
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
    else ()
        add_test(NAME python.fixture_converter COMMAND "${CMAKE_COMMAND}" -E false)
        set_tests_properties(python.fixture_converter PROPERTIES LABELS "fixture;missing_python" DISABLED TRUE)
        add_test(NAME fixture.model_assets COMMAND "${CMAKE_COMMAND}" -E false)
    endif ()
    set_tests_properties(fixture.model_assets PROPERTIES LABELS fixture
            FIXTURES_SETUP aethermind_model_assets TIMEOUT 120)
    if (NOT models_ready AND NOT AETHERMIND_VALIDATION_REQUIRED_ASSETS)
        set_tests_properties(fixture.model_assets PROPERTIES DISABLED TRUE LABELS "fixture;missing_assets")
    endif ()

    file(STRINGS "${PROJECT_SOURCE_DIR}/tests/unit/apps/inference/test_inference_cli.py"
            cli_methods REGEX "^    def test_[A-Za-z0-9_]+\\(self\\):")
    string(CONCAT preload_pattern
            "^(test_logger_initialization_allocation_failure_precedes_model_loading|"
            "test_real_library_warning_uses_stderr_without_polluting_json)$")
    foreach (method IN LISTS cli_methods)
        string(REGEX REPLACE "^    def (test_[A-Za-z0-9_]+)\\(self\\):.*" "\\1" method "${method}")
        set(name "python.cli.${method}")
        if (Python3_Interpreter_FOUND)
            add_test(NAME "${name}" COMMAND "${Python3_EXECUTABLE}"
                    "${PROJECT_SOURCE_DIR}/tests/unit/apps/inference/test_inference_cli.py"
                    --infer "${cli_binary}" --fixtures "${AETHERMIND_TEST_MODELS_DIR}/inference-admission"
                    "InferenceCliTests.${method}" -v)
        else ()
            add_test(NAME "${name}" COMMAND "${CMAKE_COMMAND}" -E false)
        endif ()
        set(labels "cli;model;validation;python")
        set(reason "")
        if (method MATCHES "${preload_pattern}")
            list(APPEND labels preload)
            if (NOT AETHERMIND_SANITIZER STREQUAL "none" OR NOT AETHERMIND_TEST_HAS_GLIBC)
                set(reason unavailable_preload)
            endif ()
        endif ()

        if (NOT Python3_Interpreter_FOUND)
            set(reason missing_python)
        elseif (NOT models_ready AND NOT AETHERMIND_VALIDATION_REQUIRED_ASSETS)
            set(reason missing_assets)
        elseif (NOT cli_ready AND NOT AETHERMIND_VALIDATION_REQUIRED_ASSETS)
            set(reason missing_cli)
        endif ()
        if (reason)
            list(APPEND labels "${reason}")
            set_tests_properties("${name}" PROPERTIES DISABLED TRUE)
        endif ()
        set_tests_properties("${name}" PROPERTIES LABELS "${labels}" TIMEOUT 120
                FIXTURES_REQUIRED aethermind_model_assets WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    endforeach ()
endfunction()

cmake_policy(POP)
