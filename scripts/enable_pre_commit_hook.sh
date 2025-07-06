#!/usr/bin/bash

cat <<'EOF' >.git/hooks/pre-commit
#!/bin/bash
set -e

echo "[pre-commit] Running format script..."
sh ./scripts/format.sh

echo "[pre-commit] Running build and test script..."
sh ./scripts/build_and_test.sh Release

echo "[pre-commit] Pre-commit checks passed."
EOF

chmod +x .git/hooks/pre-commit
