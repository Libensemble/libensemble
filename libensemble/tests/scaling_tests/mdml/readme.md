## Running test run_libe_mdml.py

This libEnsemble workflow is inspired by, and runs components of DeepDriveMD (https://deepdrivemd.github.io/),
an adaptive machine-learning driven molecular dynamics loop. Through this test,
we hope to evaluate libEnsemble's effectiveness at coordinating workflows
involving GPUs, large amounts of data, dynamic resource sets, etc.

## Getting started on Swing

# Create a new python environment
```
$ module load anaconda3/2020.11
$ conda create -name new_env python=3.8
...
$ conda activate new_env
```

# Install libEnsemble, DeepDriveMD, and other initial dependencies

We'll be running components of DeepDriveMD as tasks within libEnsemble. The MD components
require additional dependencies.

```
$ pip install libensemble
...
$ pip install git+https://github.com/DeepDriveMD/DeepDriveMD-pipeline.git
...
$ pip install git+https://github.com/braceal/MD-tools.git
...
$ pip install git+https://github.com/braceal/molecules.git
```

# Install OpenMM from source

The binaries of OpenMM available on conda-forge or other distribution sources
were compiled with a version of CUDA that is not supported on Swing's GPUs. Therefore,
we need to build OpenMM from source with the expected CUDA version (11.0)

Load the following modules:
1) gcc/9.2.0-r4tyw54
2) cuda/11.0.2-4szlv2t

The drivers on Swing's GPUs expect CUDA 11.0. Other versions from the other modules won't work for this purpose.

Follow the instructions from here: http://docs.openmm.org/7.1.0/userguide/library.html#compiling-openmm-from-source-code

Notes:

1) Use the ``cmake`` module for cmake: ``module load cmake; cmake -i ..``
2) See http://docs.openmm.org/7.1.0/userguide/library.html#other-required-software for instructions on compiling the Python API wrappers. SWIG and Doxygen will need to be downloaded and installed separately.
3) In the event you receive an error regarding ``CUDA_CUDA_LIBRARY`` being set to ``NOTFOUND``,
set it to ``/gpfs/fs1/soft/swing/spack-0.16.1/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.2.0/cuda-11.0.2-4szlv2t/lib64/stubs``.
