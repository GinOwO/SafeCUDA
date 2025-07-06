#!/bin/bash

set -e

extensions=(
        "*.c" "*.cpp" "*.cc" "*.cxx" "*.c++"
        "*.h" "*.hpp" "*.hh" "*.hxx" "*.h++"
        "*.cu" "*.cuh"
)

find_cmd="find . -type d -name '*build*' -prune -o \( "
for ext in "${extensions[@]}"; do
        find_cmd+=" -name \"$ext\" -o"
done
find_cmd=${find_cmd::-3}
find_cmd+=" \) -print0"

eval "$find_cmd" | while IFS= read -r -d '' file; do
        echo "Formatting: $file"
        clang-format -i "$file"
done
