#!/usr/bin/env bash
# Script to run same steps as travis CI - at time of writing.
# Note: Travis steps are wrapped in some checks as a precaution
# Note: This does NOT guarantee Travis will pass
# Future - Separate kernel and call from travis and here.
#        - Maybe use docker container like travis

  TRAVIS_PYTHON_VERSION=3.5
  RUN_TESTS_SCRIPT=../code/tests/run-tests.sh
  
  if [ -n $1 ]; then  
    TRAVIS_PYTHON_VERSION=$1
  fi;
  
  # Checks -----------------------------------------------------------------------------------------
  cont=true
  
  #Check can find run-tests.sh - normally run in conda/ dir.
  if [ ! -e $RUN_TESTS_SCRIPT ]; then
    echo -e 'Cannot find $RUN_TESTS_SCRIPT - aborting'
    cont=false  
  fi;
  
  #Check for existing miniconda  
  if [ $cont = 'true' ]; then
    # Prompt if miniconda already installed - up to user to delete - won't do automatically
    # If miniconda2/miniconda3 etc dirs exist this should not remove - but will prepend on path.
    miniconda_exists=false
    #miniconda_name=miniconda
    if [ -d ~/miniconda ]; then
      miniconda_exists=true
      echo -e 'A miniconda installation already exists - script will not delete it - carry on without re-install y/n?'
      read var
      if [ $var = 'y' ] || [ $var = 'Y' ]; then
        echo -e 'Miniconda will still check for updates - but existing packages  will not be removed'
        cont=true;
      else
        echo -e 'Aborting - run: Remove ~/miniconda and re-run'
        cont=false
      fi;
    else
      cont=true;
    fi;
  fi;
  
  #Check if env called condaenv exists in this miniconda
  if [ $cont = 'true' ]; then
    if [ $miniconda_exists = 'true' ]; then  
      conda info -e
      if [[ $(conda info -e | egrep condaenv) ]]; then
        echo -e 'Environment condaenv already exists - remove y/n?'
        read var
        if [ $var = 'y' ] || [ $var = 'Y' ]; then
          echo -e 'Removing condaenv'
          if [[ $(conda remove --name condaenv --all) ]]; then
            cont=true;
          else
	         echo -e 'Failed to remove condaenv'
	         cont=false
	        fi;
        else
          echo -e 'Aborting - run: Remove env condaenv and re-run'
          cont=false
        fi;      
      fi; 
    fi;
  fi;

  # Install and run tests --------------------------------------------------------------------------
  
  if [ $cont = 'true' ]; then  
    echo -e "\nRunning Travis CI steps for python $TRAVIS_PYTHON_VERSION\n"

    if [ $miniconda_exists = "false" ]; then
      #Do this conditionally because it saves some downloading if the version is the same.
      if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then
         wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh;
      else
         wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
      fi;
    fi;

    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    hash -r
    conda config --set always_yes yes --set changeps1 no
    conda update -q conda
    conda info -a # For debugging any issues with conda
    conda create --yes --name condaenv python=$TRAVIS_PYTHON_VERSION 
    source activate condaenv

  # Currently mpi4py 2.0.0 fails with mpi_init error on some platforms need dev version from source. 
  # Means installing dependencies sep. including MPI lib.
  #install:
    conda install --yes mpi4py
    conda install --yes -c conda-forge petsc4py
    conda install --yes -c conda-forge nlopt
    conda install --yes pytest pytest-cov
    conda install --yes -c conda-forge coveralls
    conda install --yes scipy 
    conda install --yes cython
    pip install git+https://bitbucket.org/mpi4py/mpi4py.git@master
  #  python setup.py install

  # Run test
  #script:
  $RUN_TESTS_SCRIPT  
fi;
