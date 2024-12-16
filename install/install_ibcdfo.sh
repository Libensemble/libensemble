#!/usr/bin/env bash

git clone --recurse-submodules -b develop https://github.com/POptUS/IBCDFO.git
pushd IBCDFO/minq/py/minq5/
export PYTHONPATH="$PYTHONPATH:$(pwd)"
echo "PYTHONPATH=$PYTHONPATH" >> $GITHUB_ENV
popd
pushd IBCDFO/ibcdfo_pypkg/
pip install -e .
popd
