#!/usr/bin/bash

set -e

BUILD_TYPE=${1:-Debug}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BUILD_DIR="$SCRIPT_DIR/../cmake-build-$BUILD_TYPE"
PRELOAD_LIB="$BUILD_DIR/libsafecuda.so"

if [[ ! -f "$PRELOAD_LIB" ]]; then
    echo "Preload library '$PRELOAD_LIB' not found. Build the library first."
    exit 1
fi

echo "Running tests for SafeCUDA..."
cd "$BUILD_DIR"
LD_PRELOAD="$PRELOAD_LIB" ctest --output-on-failure

# shellcheck disable=SC2181
if [[ $? -ne 0 ]]; then
        echo "Tests failed. Please check the output above for details."
        exit 1
else
        echo "All tests passed successfully!"
fi
