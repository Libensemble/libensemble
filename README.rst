.. image:: docs/images/libE_logo.png
   :align: center
   :alt: libEnsemble

|

.. image:: https://img.shields.io/pypi/v/libensemble.svg?color=blue
   :target: https://pypi.org/project/libensemble

.. image:: https://github.com/Libensemble/libensemble/workflows/libEnsemble-CI/badge.svg?branch=main
   :target: https://github.com/Libensemble/libensemble/actions

.. image:: https://coveralls.io/repos/github/Libensemble/libensemble/badge.svg?branch=main
   :target: https://coveralls.io/github/Libensemble/libensemble?branch=main

.. image:: https://readthedocs.org/projects/libensemble/badge/?maxAge=2592000
   :target: https://libensemble.readthedocs.org/en/latest/
   :alt: Documentation Status

|

.. after_badges_rst_tag

===========================
Introduction to libEnsemble
===========================

libEnsemble is a Python library to coordinate the concurrent evaluation of
dynamic ensembles of calculations. The library is developed to use massively
parallel resources to accelerate the solution of design, decision, and
inference problems and to expand the class of problems that can benefit from
increased concurrency levels.

libEnsemble aims for the following:

• Extreme scaling
• Resilience/fault tolerance
• Monitoring/killing of tasks (and recovering resources)
• Portability and flexibility
• Exploitation of persistent data/control flow

The user selects or supplies a *generator function* that produces
input parameters for a *simulator function* that performs and monitors
simulations. For example, the generator function may contain an
optimization routine to generate new simulation parameters on-the-fly based on
the results of previous simulations. Examples and templates of such functions are
included in the library.

libEnsemble employs a manager/worker scheme that can run on various
communication media (including MPI, multiprocessing, and TCP); interfacing with
user-provided executables is also supported. Each worker can
control and monitor any level of work, from small subnode tasks to huge
many-node simulations. An executor interface is provided to ensure that scripts
are portable, resilient, and flexible; it also enables automatic detection of
the nodes and cores available to the user, and can dynamically assign resources
to workers.

.. before_dependencies_rst_tag

Dependencies
~~~~~~~~~~~~

Required dependencies:

* Python_ 3.6 or above
* NumPy_
* psutil_

For libEnsemble running with the mpi4py parallelism:

* A functional MPI 1.x/2.x/3.x implementation, such as MPICH_, built with shared/dynamic libraries
* mpi4py_ v2.0.0 or above

Optional dependencies:

* Balsam_

From v0.2.0, libEnsemble has the option of using the Balsam job manager. Balsam
is required in order to run libEnsemble on the compute nodes of some supercomputing
platforms that do not support launching tasks from compute nodes. As of v0.5.0,
libEnsemble can also be run on launch nodes using multiprocessing.

* pyyaml_

As of v0.8.0, an alternative interface is available. An Ensemble object is
created and can be parameterized by a YAML file.

* funcX_

As of v0.8.0+dev, workers can optionally submit generator or simulator
function instances to remote funcX_ endpoints, distributing an ensemble across
systems and heterogenous resources.

The example simulation and generation functions and tests require the following:

* SciPy_
* mpmath_
* petsc4py_
* DEAP_
* DFO-LS_
* Tasmanian_
* NLopt_
* `PETSc/TAO`_ - Can optionally be installed by pip along with petsc4py
* Surmise_

PETSc and NLopt must be built with shared libraries enabled and present in
``sys.path`` (e.g., via setting the ``PYTHONPATH`` environment variable). NLopt
should produce a file ``nlopt.py`` if Python is found on the system. See the
`NLopt documentation` for information about building NLopt with shared
libraries. NLopt may also require SWIG_ to be installed on certain systems.

Installation
~~~~~~~~~~~~

libEnsemble can be installed or accessed from a variety of sources.

Install libEnsemble and its dependencies from PyPI_ using pip::

    pip install libensemble

Install libEnsemble with Conda_ from the conda-forge channel::

    conda config --add channels conda-forge
    conda install -c conda-forge libensemble

Install libEnsemble using the Spack_ distribution::

    spack install py-libensemble

libEnsemble is included in the `xSDK Extreme-scale Scientific Software Development Kit`_
from xSDK version 0.5.0 onward. Install the xSDK and load the environment with ::

    spack install xsdk
    spack load -r xsdk

The codebase, tests and examples can be accessed in the GitHub_ repository.
If necessary, you may install all optional dependencies (listed above) at once
with ::

    pip install libensemble[extras]

A tarball_ of the most recent release is also available.

Testing
~~~~~~~

The provided test suite includes both unit and regression tests and is run
regularly on:

* `GitHub Actions`_

The test suite requires the mock_, pytest_, pytest-cov_, and pytest-timeout_
packages to be installed and can be run from the ``libensemble/tests`` directory
of the source distribution by running ::

    ./run-tests.sh

Further options are available. To see a complete list of options, run ::

    ./run-tests.sh -h

The regression tests also work as good example libEnsemble scripts and can
be run directly in ``libensemble/tests/regression_tests``. For example::

    cd libensemble/tests/regression_tests
    python test_uniform_sampling.py --comms local --nworkers 3

The ``libensemble/tests/scaling_tests`` directory includes some examples that make
use of the executor to run compiled applications. These are tested regularly on
HPC systems.

If you have the source distribution, you can download (but not install) the testing
prerequisites and run the tests with ::

    python setup.py test

in the top-level directory containing the setup script.

Coverage reports are produced separately for unit tests and regression tests
under the relevant directories. For parallel tests, the union of all processors
is taken. Furthermore, a combined coverage report is created at the top level,
which can be viewed at ``libensemble/tests/cov_merge/index.html``
after ``run_tests.sh`` is completed. The coverage results are available
online at Coveralls_.

.. note::
    The executor tests can be run by using the direct-launch or
    Balsam executors. Balsam integration with libEnsemble is now tested
    via ``test_balsam_hworld.py``.

Basic Usage
~~~~~~~~~~~

The examples directory contains example libEnsemble calling scripts, simulation
functions, generation functions, allocation functions, and libEnsemble submission scripts.

The default manager/worker communications mode is MPI. The user script is
launched as ::

    mpiexec -np N python myscript.py

where ``N`` is the number of processors. This will launch one manager and
``N-1`` workers.

If running in local mode, which uses Python's multiprocessing module, the
``local`` comms option and the number of workers must be specified. The script
can then be run as a regular Python script::

    python myscript.py

These options may be specified via the command line by using the ``parse_args()``
convenience function within libEnsemble's ``tools`` module.

See the `user guide`_ for more information.

Resources
~~~~~~~~~

**Support:**

- The best way to receive support is to email questions to ``libEnsemble@lists.mcs.anl.gov``.
- Communicate (and establish a private channel, if desired) at the `libEnsemble Slack page`_.
- Join the `libEnsemble mailing list`_ for updates about new releases.

**Further Information:**

- Documentation is provided by ReadtheDocs_.
- An overview of libEnsemble's structure and capabilities is given in this manuscript_ and poster_

**Citation:**

- Please use the following to cite libEnsemble:

.. code-block:: bibtex

  @techreport{libEnsemble,
    title   = {{libEnsemble} Users Manual},
    author  = {Stephen Hudson and Jeffrey Larson and Stefan M. Wild and
               David Bindel and John-Luke Navarro},
    institution = {Argonne National Laboratory},
    number  = {Revision 0.8.0+dev},
    year    = {2021},
    url     = {https://buildmedia.readthedocs.org/media/pdf/libensemble/latest/libensemble.pdf}
  }

  @article{Hudson2022,
    title   = {{libEnsemble}: A Library to Coordinate the Concurrent
               Evaluation of Dynamic Ensembles of Calculations},
    author  = {Stephen Hudson and Jeffrey Larson and John-Luke Navarro and Stefan Wild},
    journal = {{IEEE} Transactions on Parallel and Distributed Systems},
    volume  = {33},
    number  = {4},
    pages   = {977--988},
    year    = {2022},
    doi     = {10.1109/tpds.2021.3082815}
  }

**Capabilities:**

libEnsemble generation capabilities include:

- APOSMM_ Asynchronously parallel optimization solver for finding multiple minima. Supported local optimization routines include:

  - DFO-LS_ Derivative-free solver for (bound constrained) nonlinear least-squares minimization
  - NLopt_ Library for nonlinear optimization, providing a common interface for various methods
  - scipy.optimize_ Open-source solvers for nonlinear problems, linear programming,
    constrained and nonlinear least-squares, root finding, and curve fitting.
  - `PETSc/TAO`_ Routines for the scalable (parallel) solution of scientific applications

- DEAP_ Distributed evolutionary algorithms
- Distributed optimization methods for minimizing sums of convex functions. Methods include:

  - Primal-dual sliding (https://arxiv.org/pdf/2101.00143).
  - Distributed gradient descent with gradient tracking (https://arxiv.org/abs/1908.11444).
  - Proximal sliding (https://arxiv.org/abs/1406.0919).

- ECNoise_ Estimating Computational Noise in Numerical Simulations
- Surmise_ Modular Bayesian calibration/inference framework
- Tasmanian_ Toolkit for Adaptive Stochastic Modeling and Non-Intrusive ApproximatioN
- VTMOP_ Fortran package for large-scale multiobjective multidisciplinary design optimization

libEnsemble has also been used to coordinate many computationally expensive
simulations. Select examples include:

- OPAL_ Object Oriented Parallel Accelerator Library. (See this `IPAC manuscript`_.)
- WarpX_ Advanced electromagnetic particle-in-cell code. (See example `WarpX + libE scripts`_.)

See a complete list of `example user scripts`_.

.. after_resources_rst_tag

.. _APOSMM: https://link.springer.com/article/10.1007/s12532-017-0131-4
.. _AWA: https://link.springer.com/article/10.1007/s12532-017-0131-4
.. _Balsam: https://www.alcf.anl.gov/support-center/theta/balsam
.. _Conda: https://docs.conda.io/en/latest/
.. _Coveralls: https://coveralls.io/github/Libensemble/libensemble?branch=main
.. _DEAP: https://deap.readthedocs.io/en/master/overview.html
.. _DFO-LS: https://github.com/numericalalgorithmsgroup/dfols
.. _ECNoise: https://www.mcs.anl.gov/~wild/cnoise/
.. _example user scripts: https://libensemble.readthedocs.io/en/main/examples/examples_index.html
.. _funcX: https://funcx.org/
.. _GitHub: https://github.com/Libensemble/libensemble
.. _GitHub Actions: https://github.com/Libensemble/libensemble/actions
.. _IPAC manuscript: https://doi.org/10.18429/JACoW-ICAP2018-SAPAF03
.. _libEnsemble mailing list: https://lists.mcs.anl.gov/mailman/listinfo/libensemble
.. _libEnsemble Slack page: https://libensemble.slack.com
.. _manuscript: https://arxiv.org/abs/2104.08322
.. _mock: https://pypi.org/project/mock
.. _mpi4py: https://bitbucket.org/mpi4py/mpi4py
.. _MPICH: http://www.mpich.org/
.. _mpmath: http://mpmath.org/
.. _NLopt documentation: http://ab-initio.mit.edu/wiki/index.php/NLopt_Installation#Shared_libraries
.. _nlopt: http://ab-initio.mit.edu/wiki/index.php/NLopt
.. _NumPy: http://www.numpy.org
.. _OPAL: http://amas.web.psi.ch/docs/opal/opal_user_guide-1.6.0.pdf
.. _petsc4py: https://bitbucket.org/petsc/petsc4py
.. _PETSc/TAO: http://www.mcs.anl.gov/petsc
.. _poster: https://figshare.com/articles/libEnsemble_A_Python_Library_for_Dynamic_Ensemble-Based_Computations/12559520
.. _psutil: https://pypi.org/project/psutil/
.. _PyPI: https://pypi.org
.. _pytest-cov: https://pypi.org/project/pytest-cov/
.. _pytest-timeout: https://pypi.org/project/pytest-timeout/
.. _pytest: https://pypi.org/project/pytest/
.. _Python: http://www.python.org
.. _pyyaml: https://pyyaml.org/
.. _ReadtheDocs: http://libensemble.readthedocs.org/
.. _SciPy: http://www.scipy.org
.. _scipy.optimize: https://docs.scipy.org/doc/scipy/reference/optimize.html
.. _Spack: https://spack.readthedocs.io/en/latest
.. _Surmise: https://surmise.readthedocs.io/en/latest/index.html
.. _SWIG: http://swig.org/
.. _tarball: https://github.com/Libensemble/libensemble/releases/latest
.. _Tasmanian: https://tasmanian.ornl.gov/
.. _user guide: https://libensemble.readthedocs.io/en/latest/programming_libE.html
.. _VTMOP: https://informs-sim.org/wsc20papers/311.pdf
.. _WarpX: https://warpx.readthedocs.io/en/latest/
.. _WarpX + libE scripts: https://warpx.readthedocs.io/en/latest/usage/workflows/libensemble.html
.. _xSDK Extreme-scale Scientific Software Development Kit: https://xsdk.info
