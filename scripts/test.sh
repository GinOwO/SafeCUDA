#!/usr/bin/bash

set -e

BUILD_TYPE=${1:-Debug}
BUILD_DIR="cmake-build-$BUILD_TYPE"

echo "Running tests for SafeCUDA..."
cd "$BUILD_DIR"
ctest --output-on-failure

# shellcheck disable=SC2181
if [[ $? -ne 0 ]]; then
        echo "Tests failed. Please check the output above for details."
        exit 1
else
        echo "All tests passed successfully!"
fi
