#!/usr/bin/env bash

#Note - This assumes miniconda - with anaconda some will already be installed
#You must have installed miniconda: https://conda.io/docs/install/quick.html

#To create and activate new environment called py3.6
#> conda create --name py3.6 python=3.6
#> source activate py3.6

#To come out of env: source deactivate
#To see envs: conda info --envs

#--------------------------------------------------------------------------

#Install packages for libensemble. Do in this order!!!
  conda install --yes mpi4py
  conda install --yes -c conda-forge petsc4py
  conda install --yes -c conda-forge nlopt
  conda install --yes pytest pytest-cov 
  conda install --yes scipy 
  
  #To use dev version of mpi4py
  #conda install --yes cython
  #pip install git+https://bitbucket.org/mpi4py/mpi4py.git@master
  
  echo -e "\nDo 'conda list' to see installed packages"
  echo -e "Do './run-tests.sh' to run the tests\n"
