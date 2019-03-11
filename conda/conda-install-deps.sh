#!/usr/bin/env bash

# Install dependencies for libEnsemble and tests in Conda.
# Note: libEnsemble itself is currently not in Conda
# To replicate travis builds - see directory run_travis_locally/

# Note - This assumes miniconda - with anaconda some will already be installed
# You must have installed miniconda: https://conda.io/docs/install/quick.html

# To isolate from local installs run this before activating environment
# export PYTHONNOUSERSITE=1

# To create and activate new environment called py3.6
# > conda create --name py3.6 python=3.6
# > source activate py3.6
# To enable running this script without prompts:
# > conda config --set always_yes yes
# Then source this script: . ./conda-install-deps.sh

# To come out of env: source deactivate
# To see envs: conda info --envs

# You may need to add the conda-forge chanel
# conda config --add channels conda-forge

#--------------------------------------------------------------------------

# Install packages for libensemble. Do in this order!!!

# This should work for mpich as of v0.4.1
# For other MPI libraries, some packages, such as mpi4py and PETSc may
# need to be pip installed.

export MPI=MPICH
export LIBE_BRANCH=master

conda install gcc_linux-64 || return
conda install $MPI || return
#conda install numpy || return #scipy includes numpy
conda install --no-update-deps scipy || return
conda install --no-update-deps  mpi4py || return
conda install --no-update-deps petsc4py petsc || return
conda install --no-update-deps nlopt || return

# pip install these as the conda installs downgrade pytest on python3.4
pip install pytest || return
pip install pytest-cov || return
pip install pytest-timeout || return
pip install mock || return
# pip install coveralls || return # for online

# Install libEnsemble

# From source
git clone -b $LIBE_BRANCH https://github.com/Libensemble/libensemble.git || return
cd libensemble/ || return
pip install -e . || return
# OR
# pip install libEnsemble

echo -e "\nDo 'conda list' to see installed packages"
echo -e "Do 'libensemble/tests/run-tests.sh' to run the tests\n"
