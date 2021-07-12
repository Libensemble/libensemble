## Running test run_libe_ddmd.py

This libEnsemble workflow is inspired by, and runs components of DeepDriveMD (https://deepdrivemd.github.io/),
an adaptive machine-learning driven molecular dynamics loop. Through this test,
we hope to evaluate libEnsemble's effectiveness at coordinating workflows
involving GPUs, large amounts of data, dynamic resource sets, multiple applications,
etc.

## Getting started on Swing

# Create a new python environment
```
$ module load anaconda3/2020.11
$ conda create -name new_env python=3.8
...
$ conda activate new_env
```

Note: Set ``export PYTHONNOUSERSITE=1`` as a preventative measure
for package conflicts in ``.local``.

# Install DeepDriveMD, and other initial dependencies

We'll be running components of DeepDriveMD as tasks within libEnsemble. The MD components
require additional dependencies.

```
$ pip install git+https://github.com/DeepDriveMD/DeepDriveMD-pipeline.git
...
$ pip install git+https://github.com/braceal/MD-tools.git
...
$ pip install git+https://github.com/braceal/molecules.git
...
$ conda install scikit-learn mpi4py pandas numpy==1.20.3
```

# Install OpenMM

The binaries of OpenMM available on conda-forge or other distribution sources
were compiled with a version of CUDA that is not supported on Swing's GPUs. Therefore,
we need to build OpenMM from source with the expected CUDA version (11.0)

Helpful pointers for installing OpenMM on Swing (or other systems): https://gist.github.com/lee212/4bbfe520c8003fbb91929731b8ea8a1e

Load the following modules:
1) gcc/9.2.0-r4tyw54
2) cuda/11.0.2-4szlv2t
3) cmake

Obtain the source code from: https://github.com/openmm/openmm/releases/tag/7.5.1

Do the following:

```
$ mkdir build_openmm
$ mkdir install_openmm
$ conda install cython swig doxygen
...
$ cd build_openmm
$ ccmake -i ../openmm-7.5.1/
```

Follow the instructions from here: http://docs.openmm.org/7.1.0/userguide/library.html#compiling-openmm-from-source-code

Notes:

1) In the event you receive an error regarding ``CUDA_CUDA_LIBRARY`` being set to ``NOTFOUND``,
set it (under Advanced Options) to ``/gpfs/fs1/soft/swing/spack-0.16.1/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.2.0/cuda-11.0.2-4szlv2t/lib64/stubs``. If the option doesn't persist, append the above to your ``PATH``.
2) The first time running ``ccmake``, the initial set of options may not be very large, but configure anyway. A subsequent set of options will appear for configuration. Make sure that ``CMAKE_INSTALL_PREFIX``, ``DOXYGEN_EXECUTABLE``, ``PYTHON_EXECUTABLE``, ``SWIG_EXECUTABLE``, and others are accurate.
4) A configuration warning stating ``Could NOT find OPENCL (missing: OPENCL_LIBRARY)`` can be ignored.

# Install libEnsemble

```
$ pip install libensemble
```

# Executing the test

The test can be found in ``libensemble/libensemble/tests/scaling_tests/ddmd``,
whereever libEnsemble was installed.

Feel free to adjust ``MD_BATCH_SIZE``, ``'sim_max'`` or ``sim_specs['user']['sim_length_ns']`` to customize
the length of the routine.

Currently, ``swing_submit_central.sh`` is the only batch submission script known to work.
Adjust the account and number of workers within this file, then run ``sbatch`` on it
to submit ``run_libe_mdml.py`` to the scheduler.
