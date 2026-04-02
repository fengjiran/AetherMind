#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD_DIR="build"
TARGET="all"
BUILD_TYPE="Debug"
BUILD_TESTS="ON"
BUILD_BENCHMARKS="ON"
USE_LIBBACKTRACE="ON"
BACKTRACE_ON_SEGFAULT="ON"
ENABLE_TSAN="OFF"
CONFIGURE_ONLY="false"

usage() {
    cat <<'EOF'
Usage: ./tools/build_project.sh [options]

Options:
  --build-dir <dir>         Build directory (default: build)
  --build-type <type>       CMake build type: Debug | Release |
                            RelWithDebInfo | MinSizeRel
                            (default: Debug)
  --debug                   Shortcut for --build-type Debug
  --release                 Shortcut for --build-type Release
  --relwithdebinfo          Shortcut for --build-type RelWithDebInfo
  --minsizerel              Shortcut for --build-type MinSizeRel
  --target <name>           Build target: all | AetherMind | ammalloc |
                            aethermind_unit_tests | aethermind_benchmark
                            (default: all)
  --skip-tests              Configure with BUILD_TESTS=OFF
  --skip-benchmarks         Configure with BUILD_BENCHMARKS=OFF
  --disable-libbacktrace    Configure with USE_LIBBACKTRACE=OFF
  --disable-segfault-hook   Configure with BACKTRACE_ON_SEGFAULT=OFF
  --tsan                    Configure with ENABLE_TSAN=ON
  --configure-only          Only run cmake configure
  -h, --help                Show this help message

Examples:
  ./tools/build_project.sh
  ./tools/build_project.sh --release
  ./tools/build_project.sh --target AetherMind
  ./tools/build_project.sh --skip-benchmarks --target aethermind_unit_tests
  ./tools/build_project.sh --tsan --build-dir build-tsan --target aethermind_unit_tests
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
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --relwithdebinfo)
            BUILD_TYPE="RelWithDebInfo"
            shift
            ;;
        --minsizerel)
            BUILD_TYPE="MinSizeRel"
            shift
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        --skip-tests)
            BUILD_TESTS="OFF"
            shift
            ;;
        --skip-benchmarks)
            BUILD_BENCHMARKS="OFF"
            shift
            ;;
        --disable-libbacktrace)
            USE_LIBBACKTRACE="OFF"
            shift
            ;;
        --disable-segfault-hook)
            BACKTRACE_ON_SEGFAULT="OFF"
            shift
            ;;
        --tsan)
            ENABLE_TSAN="ON"
            shift
            ;;
        --configure-only)
            CONFIGURE_ONLY="true"
            shift
            ;;
        -h|--help)
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

case "$BUILD_TYPE" in
    Debug|Release|RelWithDebInfo|MinSizeRel)
        ;;
    *)
        echo "Unsupported build type: $BUILD_TYPE" >&2
        usage
        exit 1
        ;;
esac

echo "======================================"
echo "AetherMind Build Script"
echo "======================================"
echo "Root directory          : $ROOT_DIR"
echo "Build directory         : $BUILD_DIR"
echo "Build type              : $BUILD_TYPE"
echo "Target                  : $TARGET"
echo "BUILD_TESTS             : $BUILD_TESTS"
echo "BUILD_BENCHMARKS        : $BUILD_BENCHMARKS"
echo "USE_LIBBACKTRACE        : $USE_LIBBACKTRACE"
echo "BACKTRACE_ON_SEGFAULT   : $BACKTRACE_ON_SEGFAULT"
echo "ENABLE_TSAN             : $ENABLE_TSAN"
echo "CONFIGURE_ONLY          : $CONFIGURE_ONLY"
echo

cmake -S "$ROOT_DIR" -B "$ROOT_DIR/$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DBUILD_TESTS="$BUILD_TESTS" \
    -DBUILD_BENCHMARKS="$BUILD_BENCHMARKS" \
    -DUSE_LIBBACKTRACE="$USE_LIBBACKTRACE" \
    -DBACKTRACE_ON_SEGFAULT="$BACKTRACE_ON_SEGFAULT" \
    -DENABLE_TSAN="$ENABLE_TSAN"

if [[ "$CONFIGURE_ONLY" == "true" ]]; then
    echo
    echo "Configure complete."
    exit 0
fi

if [[ "$TARGET" == "all" ]]; then
    cmake --build "$ROOT_DIR/$BUILD_DIR" -j
else
    cmake --build "$ROOT_DIR/$BUILD_DIR" --target "$TARGET" -j
fi

echo
echo "Build complete."
