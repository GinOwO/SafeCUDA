#!/usr/bin/bash
set -e

BUILD_TYPE=${1:-Debug}

sh ./scripts/build.sh "$BUILD_TYPE"
sh ./scripts/test.sh
