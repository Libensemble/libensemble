#!/usr/bin/env bash

git clone https://github.com/POptUS/MINQ
git -C MINQ checkout 7749b83645ea21e303a94e1200542f7028499bb8
pushd MINQ/py/minq5/
export PYTHONPATH="$PYTHONPATH:$(pwd)"
echo "PYTHONPATH=$PYTHONPATH" >> $GITHUB_ENV
popd
