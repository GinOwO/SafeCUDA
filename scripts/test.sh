#!/usr/bin/bash

set -e

echo "Running tests for SafeCUDA..."
cd build
ctest --output-on-failure

# shellcheck disable=SC2181
if [[ $? -ne 0 ]]; then
        echo "Tests failed. Please check the output above for details."
        exit 1
else
        echo "All tests passed successfully!"
fi
