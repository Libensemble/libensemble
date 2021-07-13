# Running test run_libe_ddmd.py

This libEnsemble workflow coordinates components of DeepDriveMD (https://deepdrivemd.github.io/),
an adaptive machine-learning driven molecular dynamics loop. Through this test,
we hope to evaluate libEnsemble's effectiveness at coordinating workflows
involving GPUs, large amounts of data, dynamic resource sets, multiple applications,
and other critical workflow characteristics.

# Getting started on Swing

## Create a new python environment
```
$ module load anaconda3/2020.11
$ conda create -name new_env python=3.8
...
$ conda activate new_env
```

Note: Set ``export PYTHONNOUSERSITE=1`` as a preventative measure
for package conflicts in ``.local``.

## Install DeepDriveMD and other initial dependencies

We'll be running components of DeepDriveMD as tasks within libEnsemble. Each component
requires additional dependency installations.

```
$ pip install git+https://github.com/DeepDriveMD/DeepDriveMD-pipeline.git
...
$ pip install git+https://github.com/braceal/MD-tools.git
...
$ pip install git+https://github.com/braceal/molecules.git
...
$ conda install mpi4py scikit-learn pandas numpy==1.20.3
```

## Install OpenMM

The binaries of OpenMM available on conda-forge or other distribution sources
were compiled with a version of CUDA that is not supported on Swing's GPUs. Therefore,
we need to build OpenMM from source with the expected CUDA version (11.0)

Helpful pointers for installing OpenMM on Swing (or other systems): https://gist.github.com/lee212/4bbfe520c8003fbb91929731b8ea8a1e

Load the following modules:
1) gcc/9.2.0-r4tyw54
2) cuda/11.0.2-4szlv2t
3) cmake/3.19.5-sfky2zq

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

## Install libEnsemble

```
$ git clone https://github.com/Libensemble/libensemble.git
```

## Executing the test

The test can be found in ``libensemble/libensemble/tests/scaling_tests/ddmd``,
whereever libEnsemble was installed.

Feel free to adjust ``MD_BATCH_SIZE``, ``'sim_max'`` or ``sim_specs['user']['sim_length_ns']`` to customize
the length of the routine.

Currently, ``swing_submit_central.sh`` is the only batch submission script known to work.
Adjust the account and number of workers within this file, then run ``sbatch`` on it
to submit ``run_libe_mdml.py`` to the scheduler.

## Submit multiple tasks to a single node.

Users who wish to submit multiple MD tasks to a single node must a MPI configured for Swing.
Note that current iterations of the generator and simulator user functions set ``CUDA_VISIBLE_DEVICES``
by ``workerID``.

```
$ conda uninstall mpi4py mpich mpi
...
$ export PATH=/soft/openmpi/4.1.1/swing/bin:/soft/ucx/1.10.0/swing-defaults/bin:$PATH
$ export LD_LIBRARY_PATH=/soft/openmpi/4.1.1/swing/lib:/soft/ucx/1.10.0/swing-defaults/lib:$LD_LIBRARY_PATH
```

The ``mpirun`` used by libEnsemble's ``MPIExecutor`` should now be able to concurrently
submit multiple tasks to a node.

# Test Explanation

This libEnsemble workflow (along with most others) contains three primary Python files.
``run_libe_ddmd.py`` is the libEnsemble *calling script*, where we configure and parameterize
our libEnsemble routine then launch the workflow. ``openmm_md_simf.py`` contains the
*simulation function*, which in this case configures and submits ``run_openmm.py`` from
DeepDriveMD for execution, organizes output, and notifies libEnsemble on completion.
``keras_cvae_ml_genf.py`` contains the *generator function*, and is responsible for producing
an initial set of MD simulations for our simulator function, running each of DeepDriveMD's
ML and data processing tasks on the resulting output, then initiating subsequent MD tasks.

Notably, the ``keras_cvae_ml_genf`` generator function is configured as a *persistent*
function, which means that it continuously loops and waits for output from simulations, as
opposed to simulation function instances which are currently configured to return
on completion of their tasks. The generator function therefore maintains its state
and access to all data from prior runs, a necessary capability.
