Dependencies
~~~~~~~~~~~~

Required dependencies:

* Python_ 3.5 or above.
* NumPy_

For libEnsemble running with the mpi4py parallelism:

* A functional MPI 1.x/2.x/3.x implementation, such as MPICH_, built with shared/dynamic libraries.
* mpi4py_ v2.0.0 or above

Optional dependency:

* Balsam_

From v0.2.0, libEnsemble has the option of using the Balsam job manager. This
is required for running libEnsemble on the compute nodes of some supercomputing
platforms (e.g., Cray XC40); platforms that do not support launching jobs from
compute nodes. Note that as of v0.5.0, libEnsemble can also be run on the
launch nodes using multiprocessing.

The example sim and gen functions and tests require the following dependencies:

* SciPy_
* petsc4py_
* PETSc_ - This can optionally be installed by pip along with petsc4py
* NLopt_ - Installed with `shared libraries enabled`_.

PETSc and NLopt must be built with shared libraries enabled and present in
``sys.path`` (e.g., via setting the ``PYTHONPATH`` environment variable). NLopt
should produce a file ``nlopt.py`` if Python is found on the system. NLopt may also
require SWIG_ to be installed on certain systems.

Installation
~~~~~~~~~~~~

Use pip to install libEnsemble and its dependencies::

    pip install libensemble

libEnsemble is also available in the Spack_ distribution. It can be installed from Spack with::

    spack install py-libensemble

The tests and examples can be accessed in the GitHub_ repository.
If necessary, you may install all optional dependencies (listed above) at once with::

    pip install libensemble[extras]

A tarball_ of the most recent release is also available.

Testing
~~~~~~~

The provided test suite includes both unit and regression tests and is run
regularly on:

* `Travis CI`_

The test suite requires the mock_, pytest_, pytest-cov_, and pytest-timeout_
packages to be installed and can be run from the libensemble/tests directory of
the source distribution by running::

    ./run-tests.sh

To clean the test repositories run::

    ./run-tests.sh -c

Further options are available. To see a complete list of options run::

    ./run-tests.sh -h

If you have the source distribution, you can download (but not install) the testing
prerequisites and run the tests with::

    python setup.py test

in the top-level directory containing the setup script.

Coverage reports are produced separately for unit tests and regression tests
under the relevant directories. For parallel tests, the union of all processors
is taken. Furthermore, a combined coverage report is created at the top level,
which can be viewed after ``run_tests.sh`` is completed at
``libensemble/tests/cov_merge/index.html``. The Travis CI coverage results are
available online at Coveralls_.

.. note::
    The job_controller tests can be run using the direct-launch or
    Balsam job controllers. Balsam integration with libEnsemble is now tested
    via ``test_balsam_hworld.py``.

Basic Usage
~~~~~~~~~~~

The examples directory contains example libEnsemble calling scripts, sim
functions, gen functions, alloc functions and job submission scripts.

The default manager/worker communications mode is MPI. The user script is
launched as::

    mpiexec -np N python myscript.py

where ``N`` is the number of processors. This will launch one manager and
``N-1`` workers.

If running in local mode, which uses Python's multiprocessing module, the
``local`` comms option and the number of workers must be specified. The script
can then be run as a regular python script::

    python myscript.py

When specifying these options via command line options, one may use the
``parse_args`` function used in the regression tests, which can be found in
`common.py`_ in the ``libensemble/tests/regression_tests`` directory.

See the `user guide`_ for more information.


Resources
~~~~~~~~~

**Support:**

- The best way to receive support is to email questions to ``libEnsemble@lists.mcs.anl.gov``.
- Communicate (and establish a private channel, if desired) at the `libEnsemble Slack page`_.
- Join the `libEnsemble mailing list`_ for updates about new releases.

**Further Information:**

- Documentation is provided by ReadtheDocs_.
- A visual overview of libEnsemble is given in this poster_.

**Citation:**

- Please use the following to cite libEnsemble in a publication:

.. code-block:: bibtex

  @techreport{libEnsemble,
    author      = {Stephen Hudson and Jeffrey Larson and Stefan M. Wild and
                   David Bindel and John-Luke Navarro},
    title       = {{libEnsemble} Users Manual},
    institution = {Argonne National Laboratory},
    number      = {Revision 0.5.2},
    year        = {2019},
    url         = {https://buildmedia.readthedocs.org/media/pdf/libensemble/latest/libensemble.pdf}
  }


.. after_resources_rst_tag



.. _Balsam: https://www.alcf.anl.gov/balsam
.. _common.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/common.py
.. _Coveralls: https://coveralls.io/github/Libensemble/libensemble?branch=master
.. _GitHub: https://github.com/Libensemble/libensemble
.. _libEnsemble mailing list: https://lists.mcs.anl.gov/mailman/listinfo/libensemble
.. _libEnsemble Slack page: https://libensemble.slack.com
.. _mock: https://pypi.org/project/mock
.. _mpi4py: https://bitbucket.org/mpi4py/mpi4py
.. _MPICH: http://www.mpich.org/
.. _nlopt: http://ab-initio.mit.edu/wiki/index.php/NLopt
.. _NumPy: http://www.numpy.org
.. _petsc4py: https://bitbucket.org/petsc/petsc4py
.. _PETSc: http://www.mcs.anl.gov/petsc
.. _poster: https://figshare.com/articles/LibEnsemble_PETSc_TAO-_Sustaining_a_library_for_dynamic_ensemble-based_computations/7765454
.. _pytest-cov: https://pypi.org/project/pytest-cov/
.. _pytest-timeout: https://pypi.org/project/pytest-timeout/
.. _pytest: https://pypi.org/project/pytest/
.. _Python: http://www.python.org
.. _ReadtheDocs: http://libensemble.readthedocs.org/
.. _SciPy: http://www.scipy.org
.. _shared libraries enabled: http://ab-initio.mit.edu/wiki/index.php/NLopt_Installation#Shared_libraries
.. _Spack: https://spack.readthedocs.io/en/latest
.. _SWIG: http://swig.org/
.. _tarball: https://github.com/Libensemble/libensemble/releases/latest
.. _Travis CI: https://travis-ci.org/Libensemble/libensemble
.. _user guide: https://libensemble.readthedocs.io/en/latest/user_guide.html
