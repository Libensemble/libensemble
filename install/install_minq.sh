#!/usr/bin/env bash

git clone https://github.com/POptUS/MINQ
pushd MINQ/py/minq5/
export PYTHONPATH="$PYTHONPATH:$(pwd)"
echo "PYTHONPATH=$PYTHONPATH" >> $GITHUB_ENV
popd
