#!/usr/bin/env bash

set -euo pipefail

# IBCDFO pins the exact MINQ commit it supports, and refuses to run (via
# sys.exit) if the MINQ clone is at any other commit. Read the required SHA
# from the installed ibcdfo package rather than hardcoding it here, so that
# bumping ibcdfo in pixi.lock can never silently desynchronize the two.
MINQ_COMMIT=$(python -c "
import pathlib, ibcdfo
print((pathlib.Path(ibcdfo.__file__).parent / 'PkgData' / 'REQUIRED_MINQ_COMMIT').read_text().strip())
")

if [ -z "$MINQ_COMMIT" ]; then
    echo "ERROR: could not determine required MINQ commit from installed ibcdfo" >&2
    exit 1
fi

echo "Installing MINQ at ibcdfo-required commit ${MINQ_COMMIT}"

git clone https://github.com/POptUS/MINQ
git -C MINQ checkout "$MINQ_COMMIT"
pushd MINQ/py/minq5/
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
echo "PYTHONPATH=$PYTHONPATH" >> "${GITHUB_ENV:-/dev/null}"
popd
