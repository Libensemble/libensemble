===========
libEnsemble
===========

.. image::  https://travis-ci.org/Libensemble/libensemble.svg?branch=master
   :target: https://travis-ci.org/Libensemble/libensemble

.. image:: https://coveralls.io/repos/github/Libensemble/libensemble/badge/?maxAge=2592000/?branch=master
   :target: https://coveralls.io/github/Libensemble/libensemble?branch=master
   
.. image::  https://readthedocs.org/projects/libensemble/badge/?maxAge=2592000
   :target: https://libensemble.readthedocs.org/en/latest/
   :alt: Documentation Status


Library for managing ensemble-like collections of computations.


Dependencies
------------

* Python_ 2.7, 3.4 or above.

* A functional MPI 1.x/2.x/3.x implementation like `MPICH
  <http://www.mpich.org/>`_ or `Open MPI <http://www.open-mpi.org/>`_
  built with shared/dynamic libraries.

* mpi4py_ v2.0.0 or above

* NumPy_

The examples and tests require the following dependencies:

* SciPy_
* petsc4py_
* PETSc_ - This can optionally be installed by pip along with petsc4py
* nlopt_ - Installed with `shared libraries enabled <http://ab-initio.mit.edu/wiki/index.php/NLopt_Installation#Shared_libraries>`_.

PETSc and nlopt must be built with shared libraries enabled and present in sys.path (eg. via setting the PYTHONPATH environment variable). nlopt should produce a file nlopt.py if python is found on the system.

.. _PETSc:  http://www.mcs.anl.gov/petsc
.. _Python: http://www.python.org
.. _nlopt: http://ab-initio.mit.edu/wiki/index.php/NLopt
.. _NumPy:  http://www.numpy.org
.. _SciPy:  http://www.scipy.org
.. _mpi4py:  https://bitbucket.org/mpi4py/mpi4py
.. _petsc4py:  https://bitbucket.org/petsc/petsc4py


Installation
------------

pip install can be used to install libEnsemble and access the libensemble module. However, to access the examples and tests the source distribution is required. This can be obtained via PYPI or github. The simplest way of obtaining a tarball for the latest release from PYPI is::

    pip download libensemble

You can also download the source code from `github <https://github.com/Libensemble/libensemble>`_

The examples and tests are set up to be run from the source distribution and do not currently require the libEnsemble package to be installed. Installing the package, however, will download any python dependencies required. This can be done from the top level directory using::

    pip install .

Conda: Conda can also be used for simple fast installation of dependencies using mpich (see conda/conda-install-deps.sh). This is probably the fastest approach for a clean installation from scratch as conda can install both the Python and non-Python dependencies - see conda directory for dependent packages/instructions. However, to use an existing MPI, care must be taken to ensure the installed packages do not install their own MPI dependencies - this may not be trivial. In particular,  mpi4py should be configured to point to your systems MPI if that already exists. This can be checked by locating the mpi.cfg file in the mpi4py installation. TravisCI testing has also been configured to use Conda (combined with pip to work with multiple MPI libraries) with the `Miniconda <https://conda.io/docs/install/quick.html>`_ distribution.

Spack: Libensemble is also available in the Spack_ distribution.

.. _Spack: https://spack.readthedocs.io/en/latest


Testsuite
---------

The testsuite includes both unit and regression tests and is run regularly on

* `Travis CI <https://travis-ci.org/Libensemble/libensemble>`_

The testsuite requires the pytest and pytest-cov packages to be installed and can be run from the code/tests directory of the source distribution using the following methods::

    ./run-tests.sh (optionally specify eg. -p 3 for Python3)

    python3 setup.py test (run from top level directory)

Coverage reports are produced separately for unit tests and regression tests under the relevant directories. For parallel tests, the union of all processors is taken. Furthermore, a combined coverage report is created at the top level, which can be viewed online in `Coveralls <https://coveralls.io/github/Libensemble/libensemble?branch=master>`_.


Basic Usage
-----------

Examples can be found under code/examples. 


Documentation
-------------
* http://libensemble.readthedocs.org/


Support 
-------

You can join the libEnsemble mailing list at:

* https://lists.mcs.anl.gov/mailman/listinfo/libensemble 

or email questions to:

* libensemble@lists.mcs.anl.gov

