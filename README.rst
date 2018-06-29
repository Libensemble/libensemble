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

* A functional MPI 1.x/2.x/3.x implementation such as `MPICH
  <http://www.mpich.org/>`_  built with shared/dynamic libraries.

* mpi4py_ v2.0.0 or above

* NumPy_

Optional dependency:

* Balsam_
 
From v0.2.0, libEnsemble has the option of using the Balsam job manager. This
is required for running on some supercomputing platforms (eg. Cray XC40);
platforms which do not support launching jobs on compute nodes.

The example sim and gen functions and tests require the following dependencies:

* SciPy_
* petsc4py_
* PETSc_ - This can optionally be installed by pip along with petsc4py
* NLopt_ - Installed with `shared libraries enabled <http://ab-initio.mit.edu/wiki/index.php/NLopt_Installation#Shared_libraries>`_.

PETSc and NLopt must be built with shared libraries enabled and present in
sys.path (eg. via setting the PYTHONPATH environment variable). NLopt should
produce a file nlopt.py if Python is found on the system.

.. _PETSc:  http://www.mcs.anl.gov/petsc
.. _Python: http://www.python.org
.. _nlopt: http://ab-initio.mit.edu/wiki/index.php/NLopt
.. _NumPy:  http://www.numpy.org
.. _SciPy:  http://www.scipy.org
.. _mpi4py:  https://bitbucket.org/mpi4py/mpi4py
.. _petsc4py:  https://bitbucket.org/petsc/petsc4py
.. _Balsam: https://www.alcf.anl.gov/balsam


Installation
------------

You can use pip to install libEnsemble and its dependencies::

    pip install libensemble

Libensemble is also available in the Spack_ distribution.

.. _Spack: https://spack.readthedocs.io/en/latest

The tests and examples can be accessed in the `github <https://github.com/Libensemble/libensemble>`_ repository. A `tarball <https://github.com/Libensemble/libensemble/releases/latest>`_ of the most recent release is also available.
    

Testing
---------

The provided test suite includes both unit and regression tests and is run
regularly on:

* `Travis CI <https://travis-ci.org/Libensemble/libensemble>`_

The test suite requires the pytest and pytest-cov packages to be installed and
can be run from the libensemble/tests directory of the source distribution
using the following methods::

    ./run-tests.sh (optionally specify eg. -p 3 for Python3)

To clean the test repositories run::

    ./run-tests.sh -c

Coverage reports are produced separately for unit tests and regression tests
under the relevant directories. For parallel tests, the union of all processors
is taken. Furthermore, a combined coverage report is created at the top level,
which can be viewed after running the tests via the html file
libensemble/tests/cov_merge/index.html. The Travis CI coverage results are
given online at
`Coveralls <https://coveralls.io/github/Libensemble/libensemble?branch=master>`_. 

Note for v0.2.0: The job_controller tests can be run using the direct-launch or
Balsam job controllers. However, currently only the direct-launch versions can
be run on Travis CI, which reduces the test coverage results.


Basic Usage
-----------

The examples directory contains example libEnsemble calling scripts, sim functions, gen functions, alloc functions and job submission scripts.

See the `user-guide <http://libensemble.readthedocs.org>`_ for more information.


Documentation
-------------

* http://libensemble.readthedocs.org/


Support 
-------

You can join the libEnsemble mailing list at:

* https://lists.mcs.anl.gov/mailman/listinfo/libensemble 

or email questions to:

* libensemble@lists.mcs.anl.gov

