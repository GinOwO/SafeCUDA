#!/usr/bin/bash
set -e

BUILD_TYPE=${1:-Release}

sh ./scripts/build.sh "$BUILD_TYPE"
sh ./scripts/test.sh "$BUILD_TYPE"
