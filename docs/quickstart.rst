.. image:: images/libE_logo.png
  :alt: libEnsemble

|

.. image::  https://travis-ci.org/Libensemble/libensemble.svg?branch=master
   :target: https://travis-ci.org/Libensemble/libensemble

.. image:: https://coveralls.io/repos/github/Libensemble/libensemble/badge/?maxAge=2592000/?branch=master
   :target: https://coveralls.io/github/Libensemble/libensemble?branch=master
   
.. image::  https://readthedocs.org/projects/libensemble/badge/?maxAge=2592000
   :target: https://libensemble.readthedocs.org/en/latest/
   :alt: Documentation Status

|

====================
What is libEnsemble?
====================

libEnsemble is a Python library to coordinate the concurrent evaluation of ensembles of computations.
Designed with flexibility in mind, libEnsemble can utilize massively parallel resources to accelerate
the solution of design, decision, and inference problems.

A visual overview is given in the libEnsemble poster_.

.. _poster:  https://figshare.com/articles/LibEnsemble_PETSc_TAO-_Sustaining_a_library_for_dynamic_ensemble-based_computations/7765454

libEnsemble aims for:

• Extreme scaling
• Resilience/Fault tolerance
• Monitoring/killing jobs (and recovering resources)
• Portability and flexibility
• Exploitation of persistent data/control flow.

The user selects or supplies a generation function that produces simulation input as well as
a simulation function that performs and monitors the simulations. The generation function
may contain, for example, an optimization method to generate new simulation parameters
on-the-fly and based on the results of previous simulations. Examples and templates of these
functions are included in the library.

libEnsemble employs a manager-worker scheme that can run on various communication media
(including MPI, multiprocessing, and TCP). Each worker can control and monitor any type
of job from small sub-node jobs to huge many-node simulations. A job controller
interface is provided to ensure scripts are portable, resilient and flexible; it also
enables automatic detection of the nodes and cores in a system and can split up
jobs automatically if nodes/cores are not supplied.


Dependencies
------------

Required dependencies:

* Python_ 3.4 or above.

* NumPy_

For libEnsemble running with the mpi4py parallelism:

* A functional MPI 1.x/2.x/3.x implementation such as `MPICH
  <http://www.mpich.org/>`_  built with shared/dynamic libraries.

* mpi4py_ v2.0.0 or above


Optional dependency:

* Balsam_
 
From v0.2.0, libEnsemble has the option of using the Balsam job manager. This
is required for running libEnsemble on the compute nodes of some supercomputing
platforms (e.g., Cray XC40); platforms that do not support launching jobs from compute nodes.
Note that as of v0.5.0, libEnsemble can also be run on the launch nodes using multiprocessing.

The example sim and gen functions and tests require the following dependencies:

* SciPy_
* petsc4py_
* PETSc_ - This can optionally be installed by pip along with petsc4py
* NLopt_ - Installed with `shared libraries enabled <http://ab-initio.mit.edu/wiki/index.php/NLopt_Installation#Shared_libraries>`_.

PETSc and NLopt must be built with shared libraries enabled and present in
sys.path (e.g., via setting the PYTHONPATH environment variable). NLopt should
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

Use pip to install libEnsemble and its dependencies::

    pip install libensemble

libEnsemble is also available in the Spack_ distribution.

.. _Spack: https://spack.readthedocs.io/en/latest

The tests and examples can be accessed in the `github <https://github.com/Libensemble/libensemble>`_ repository.
A `tarball <https://github.com/Libensemble/libensemble/releases/latest>`_ of the most recent release is also available.
    

Testing
---------

The provided test suite includes both unit and regression tests and is run
regularly on:

* `Travis CI <https://travis-ci.org/Libensemble/libensemble>`_

The test suite requires the mock, pytest, pytest-cov and pytest-timeout
packages to be installed and can be run from the libensemble/tests directory of
the source distribution by running::

    ./run-tests.sh

To clean the test repositories run::

    ./run-tests.sh -c
    
Further options are available. To see a complete list of options run::

    ./run-tests.sh -h

Coverage reports are produced separately for unit tests and regression tests
under the relevant directories. For parallel tests, the union of all processors
is taken. Furthermore, a combined coverage report is created at the top level,
which can be viewed after running the tests via the html file
libensemble/tests/cov_merge/index.html. The Travis CI coverage results are
given online at
`Coveralls <https://coveralls.io/github/Libensemble/libensemble?branch=master>`_. 

Note: The job_controller tests can be run using the direct-launch or
Balsam job controllers. However, currently only the direct-launch versions can
be run on Travis CI, which reduces the test coverage results.


Basic Usage
-----------

The examples directory contains example libEnsemble calling scripts, sim
functions, gen functions, alloc functions and job submission scripts.

The user will create a python script to call the libEnsemble :doc:`libE<libE_module>` function.
This must supply the :ref:`sim_specs<datastruct-sim-specs>` and :ref:`gen_specs<datastruct-gen-specs>`, 
and optionally :ref:`libE_specs<datastruct-libe-specs>`, :ref:`alloc_specs<datastruct-alloc-specs>` and :ref:`persis_info<datastruct-persis-info>`.

The default manager/worker communications mode is MPI. The user script is launched as::

    mpiexec -np N python myscript.py
 
where ``N`` is the number of processors. This will launch one manager and ``N-1`` workers.

If running in local mode, which uses Python's multiprocessing module, the 'local'
comms option and the number of workers must be specified in :ref:`libE_specs<datastruct-libe-specs>`.
The script can then be run as a regular python script::

    python myscript.py
    
When specifying these options via command line options, one may use the ``parse_args``
function used in the regression tests, which can be found in
``libensemble/tests/regression_tests/common.py``


See the `user-guide <https://libensemble.readthedocs.io/en/latest/user_guide.html>`_ for more information.


Documentation
-------------

* http://libensemble.readthedocs.org/

Citing libEnsemble
------------------
Please use the following to cite libEnsemble in a publication:

.. code-block:: bibtex

  @techreport{libEnsemble,
    author      = {Stephen Hudson and Jeffrey Larson and Stefan M. Wild and David Bindel},
    title       = {{libEnsemble} Users Manual},
    institution = {Argonne National Laboratory},
    number      = {Revision 0.5.0},
    year        = {2019},
    url         = {https://buildmedia.readthedocs.org/media/pdf/libensemble/latest/libensemble.pdf}
  }


Support 
-------

Join the libEnsemble mailing list at:

* https://lists.mcs.anl.gov/mailman/listinfo/libensemble 

or email questions to:

* libensemble@lists.mcs.anl.gov

or communicate (and establish a private channel, if desired) at:

* https://libensemble.slack.com 
