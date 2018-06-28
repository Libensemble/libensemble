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

Optional dependency:

* Balsam_
 
From v0.2.0, libEnsemble has the option of using the Balsam job manager. This is required for running on some supercomputing platforms (eg. Cray XC40); platforms which do not support launching jobs on compute nodes.

The example sim/gen functions and tests require the following dependencies:

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
.. _Balsam: https://www.alcf.anl.gov/balsam


Installation
------------

You can use pip to install libEnsemble and its dependencies::

    pip install libensemble

Spack: Libensemble is also available in the Spack_ distribution.

.. _Spack: https://spack.readthedocs.io/en/latest

The tests and examples can be accessed by the `github <https://github.com/Libensemble/libensemble>`_ repository. A tarball is available at::

    wget https://github.com/Libensemble/libensemble/releases/tag/v0.2.0/libensemble-0.2.0.tar.gz
    

Testsuite
---------

The testsuite includes both unit and regression tests and is run regularly on

* `Travis CI <https://travis-ci.org/Libensemble/libensemble>`_

The testsuite requires the pytest and pytest-cov packages to be installed and can be run from the libensemble/tests directory of the source distribution using the following methods::

    ./run-tests.sh (optionally specify eg. -p 3 for Python3)

    python3 setup.py test (run from top level directory)
    
To clean the test repositories run::

    ./run-tests.sh -c

Coverage reports are produced separately for unit tests and regression tests under the relevant directories. For parallel tests, the union of all processors is taken. Furthermore, a combined coverage report is created at the top level, which can be viewed after running the tests via the html file libensemble/tests/cov_merge/index.html. The travis results are given online in `Coveralls <https://coveralls.io/github/Libensemble/libensemble?branch=master>`_. 

Note for v0.2.0: The job_controller tests can be run using the direct-launch or Balsam job controllers. However, currently only the direct-launch versions can be run on Travis CI, which reduces the test coverage results.


Basic Usage
-----------

The best example user scripts are the regression tests. These can be found under libensemble/tests directory. 

Example submission scripts can be found in examples/job_submission_scripts

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

