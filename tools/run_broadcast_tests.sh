#!/usr/bin/env bash

# Builds the unit-test target (if needed) and runs the broadcast test suites.
#
# Usage: ./tools/run_broadcast_tests.sh [options]
#
# Options:
#   --build-dir <dir>    Build directory (default: build)
#   --build-type <type>  CMake build type: Debug | Release |
#                        RelWithDebInfo | MinSizeRel (default: Debug)
#   --no-build           Skip configure/build; run only (assumes binary exists)
#   --filter <name>      gtest filter (default: BroadcastShapes.*:BroadcastInputStrides.*:InferBroadcastShape.*)
#   --gtest-args <args>  Extra arguments passed through to the test binary
#   -h, --help           Show this help message

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD_DIR="build"
BUILD_TYPE="Debug"
NO_BUILD="false"
FILTER="BroadcastShapes.*:BroadcastInputStrides.*:InferBroadcastShape.*"
GTEST_ARGS=()

usage() {
    cat <<'EOF'
Usage: ./tools/run_broadcast_tests.sh [options]

Builds (configure + build the unit-test target if needed) and runs the
broadcast test suites (BroadcastShapes, BroadcastInputStrides,
InferBroadcastShape).

Options:
  --build-dir <dir>    Build directory (default: build)
  --build-type <type>  CMake build type: Debug | Release |
                       RelWithDebInfo | MinSizeRel (default: Debug)
  --no-build           Skip configure/build; run only (binary must exist)
  --filter <name>      gtest filter (default: Broadcast.*)
  --gtest-args <args>  Extra arguments passed through to the test binary
  -h, --help           Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --build-dir)
        BUILD_DIR="$2"
        shift 2
        ;;
    --build-type)
        BUILD_TYPE="$2"
        shift 2
        ;;
    --no-build)
        NO_BUILD="true"
        shift
        ;;
    --filter)
        FILTER="$2"
        shift 2
        ;;
    --gtest-args)
        # Remaining args are forwarded verbatim to the test binary.
        shift
        while [[ $# -gt 0 ]]; do
            GTEST_ARGS+=("$1")
            shift
        done
        ;;
    -h | --help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown option: $1" >&2
        usage
        exit 1
        ;;
    esac
done

TEST_BINARY="$ROOT_DIR/$BUILD_DIR/tests/unit/aethermind_unit_tests"

if [[ "$NO_BUILD" != "true" ]]; then
    echo "======================================"
    echo "AetherMind Broadcast Test Runner"
    echo "======================================"
    echo "Root directory : $ROOT_DIR"
    echo "Build directory: $BUILD_DIR"
    echo "Build type     : $BUILD_TYPE"
    echo "Filter         : $FILTER"
    echo

    cmake -S "$ROOT_DIR" -B "$ROOT_DIR/$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

    cmake --build "$ROOT_DIR/$BUILD_DIR" --target aethermind_unit_tests -j
fi

if [[ ! -x "$TEST_BINARY" ]]; then
    echo "Test binary not found: $TEST_BINARY" >&2
    echo "Run without --no-build, or build the aethermind_unit_tests target first." >&2
    exit 1
fi

echo
echo "Running: $TEST_BINARY --gtest_filter=$FILTER ${GTEST_ARGS[*]:-}"
echo

"$TEST_BINARY" --gtest_filter="$FILTER" "${GTEST_ARGS[@]:+${GTEST_ARGS[@]}}"
