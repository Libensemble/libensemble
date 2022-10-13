.. image:: docs/images/libEnsemble_Logo.svg
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

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

|

.. after_badges_rst_tag

===========================
Introduction to libEnsemble
===========================

libEnsemble is a Python_ toolkit for coordinating workflows of asynchronous and dynamic ensembles of calculations.

libEnsemble helps users take advantage of massively parallel resources to solve design,
decision, and inference problems and expands the class of problems that can benefit from
increased parallelism.

libEnsemble aims for:

• **Extreme scaling**: Run on or across_ laptops, clusters, and leadership-class machines.
• **Dynamic Ensembles**: Generate new tasks on-the-fly based on previous computations.
• **Dynamic Resource Management**: Reassign resource partitions of any size for tasks.
• **Monitoring/killing of applications**: Ensemble members can poll or kill running apps.
• **Resilience/fault tolerance**: libEnsemble can restart incomplete tasks or entire ensembles.
• **Portability and flexibility**: Run identical libEnsemble scripts on different machines.
• **Exploitation of persistent data/control flow**: libEnsemble can pass data between ensemble members.
• **Low start-up cost**: Default single-machine deployments don't require additional services.

libEnsemble's users select or supply **generator** and **simulator** Python
functions; these respectively produce candidate parameters and perform/monitor
computations that use those parameters. Generator functions can train
models, perform optimizations, and test candidate solutions in a batch or streaming
fashion based on simulation results.
Simulator functions can themselves use parallel resources and involve libraries
or executables that are not written in Python.

With a basic familiarity of Python and NumPy_, users can easily incorporate
any other mathematics, machine-learning, or resource-management libraries into libEnsemble
workflows.

libEnsemble employs a manager/worker scheme that communicates via MPI, multiprocessing,
or TCP. Workers control and monitor any level of work using the aforementioned
generator and simulator functions, from small subnode tasks to huge many-node computations.

libEnsemble includes an Executor interface so application-launching functions are
portable, resilient, and flexible; it also automatically detects available nodes
and cores, and can dynamically assign resources to workers.

libEnsemble performs best on Unix-like systems like Linux and macOS. See the
:ref:`FAQ<faqwindows>` for more information.

.. before_dependencies_rst_tag

Dependencies
~~~~~~~~~~~~

**Required dependencies**:

* Python_ 3.7 or above
* NumPy_
* psutil_
* setuptools_

When using  ``mpi4py`` for libEnsemble communications:

* A functional MPI 1.x/2.x/3.x implementation, such as MPICH_, built with shared/dynamic libraries
* mpi4py_ v2.0.0 or above

**Optional dependencies**:

* Balsam_

As of v0.9.0, libEnsemble features an updated `Balsam Executor`_
for workers to schedule and launch applications to *anywhere* with a running
Balsam site, including to remote machines.

* pyyaml_

libEnsemble is typically configured and parameterized via Python dictionaries. libEnsemble can also be parameterized via yaml.

* funcX_

As of v0.9.0, libEnsemble features a cross-system capability powered by funcX_,
a function-as-a-service platform to which workers can submit remote generator or
simulator function instances. This feature can help distribute an ensemble
across systems and heterogeneous resources.

* `psi-j-python`_

As of v0.9.3, libEnsemble features a set of command-line utilities for submitting
libEnsemble jobs to almost any system or scheduler via a `PSI/J`_ Python interface. tqdm_
is also required.

The example simulation and generation functions and tests require the following:

* SciPy_
* mpmath_
* petsc4py_
* DEAP_
* DFO-LS_
* Tasmanian_
* NLopt_
* `PETSc/TAO`_ - Can optionally be installed by pip along with ``petsc4py``
* Surmise_

PETSc and NLopt must be built with shared libraries enabled and be present in
``sys.path`` (e.g., via setting the ``PYTHONPATH`` environment variable). NLopt
should produce a file ``nlopt.py`` if Python is found on the system. See the
`NLopt documentation` for information about building NLopt with shared
libraries. NLopt may also require SWIG_ to be installed on certain systems.

Installation
~~~~~~~~~~~~

libEnsemble can be installed or accessed from a variety of sources.

Install libEnsemble and its dependencies from PyPI_ using pip::

    pip install libensemble

Install libEnsemble with Conda_ from the conda-forge_ channel::

    conda config --add channels conda-forge
    conda install -c conda-forge libensemble

Install libEnsemble using the Spack_ distribution::

    spack install py-libensemble

libEnsemble is included in the `xSDK Extreme-scale Scientific Software Development Kit`_
from xSDK version 0.5.0 onward. Install the xSDK and load the environment with::

    spack install xsdk
    spack load -r xsdk

The codebase, tests and examples can be accessed in the GitHub_ repository.
If necessary, you may install all optional dependencies (listed above) at once
with::

    pip install libensemble[extras]

A tarball_ of the most recent release is also available.

Testing
~~~~~~~

The provided test suite includes both unit and regression tests and is run
regularly on:

* `GitHub Actions`_

The test suite requires the mock_, pytest_, pytest-cov_, and pytest-timeout_ packages
to be installed and can be run from the ``libensemble/tests`` directory
of the source distribution by running::

    ./run-tests.sh

Further options are available. To see a complete list of options, run::

    ./run-tests.sh -h

The regression tests also work as good example libEnsemble scripts and can
be run directly in ``libensemble/tests/regression_tests``. For example::

    cd libensemble/tests/regression_tests
    python test_uniform_sampling.py --comms local --nworkers 3

The ``libensemble/tests/scaling_tests`` directory includes example scripts that
use the executor to run compiled applications. These are tested regularly on
HPC systems.

If you have the libEnsemble source code, you can download (but not install) the testing
prerequisites and run the tests with::

    python setup.py test

in the top-level directory containing the setup script.

Coverage reports are produced separately for unit tests and regression tests
under the relevant directories. For parallel tests, the union of all processors
is taken. Furthermore, a combined coverage report is created at the top level,
which can be viewed at ``libensemble/tests/cov_merge/index.html``
after ``run_tests.sh`` is completed. The coverage results are available
online at Coveralls_.

Basic Usage
~~~~~~~~~~~

The default manager/worker communications mode is MPI. The user script is
launched as::

    mpiexec -np N python myscript.py

where ``N`` is the number of processors. This will launch one manager and
``N-1`` workers.

If running in local mode, which uses Python's multiprocessing module, the
``local`` comms option and the number of workers must be specified, either in `libE_specs`_
or via the command-line using the ``parse_args()`` function. The script
can then be run as a regular Python script::

    python myscript.py --comms local --nworkers N

This will launch one manager and N workers.

See the `user guide`_ for more information.

Resources
~~~~~~~~~

**Support:**

- Email questions or request `libEnsemble Slack page`_ access from ``libEnsemble@lists.mcs.anl.gov``.
- Open issues on GitHub_.
- Join the `libEnsemble mailing list`_ for updates about new releases.

**Further Information:**

- Documentation is provided by ReadtheDocs_.
- Contributions_ to libEnsemble are welcome.
- An overview of libEnsemble's structure and capabilities is given in this manuscript_ and poster_.
- Examples of production user functions and complete workflows can be viewed, downloaded, and contributed to in the libEnsemble `Community Examples repository`_.

**Citation:**

- Please use the following to cite libEnsemble:

.. code-block:: bibtex

  @techreport{libEnsemble,
    title   = {{libEnsemble} Users Manual},
    author  = {Stephen Hudson and Jeffrey Larson and Stefan M. Wild and
               David Bindel and John-Luke Navarro},
    institution = {Argonne National Laboratory},
    number  = {Revision 0.9.3},
    year    = {2022},
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

**Example Compatible Packages**

.. before_examples_rst_tag

libEnsemble and the `Community Examples repository`_ include example generator
functions for the following libraries:

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

.. _across: https://libensemble.readthedocs.io/en/develop/platforms/platforms_index.html#funcx-remote-user-functions
.. _APOSMM: https://link.springer.com/article/10.1007/s12532-017-0131-4
.. _AWA: https://link.springer.com/article/10.1007/s12532-017-0131-4
.. _Balsam: https://balsam.readthedocs.io/en/latest/
.. _Balsam Executor: https://libensemble.readthedocs.io/en/develop/executor/balsam_2_executor.html
.. _Community Examples repository: https://github.com/Libensemble/libe-community-examples
.. _Conda: https://docs.conda.io/en/latest/
.. _conda-forge: https://conda-forge.org/
.. _Contributions: https://github.com/Libensemble/libensemble/blob/main/CONTRIBUTING.rst
.. _Coveralls: https://coveralls.io/github/Libensemble/libensemble?branch=main
.. _DEAP: https://deap.readthedocs.io/en/master/overview.html
.. _DFO-LS: https://github.com/numericalalgorithmsgroup/dfols
.. _ECNoise: https://www.mcs.anl.gov/~wild/cnoise/
.. _example user scripts: https://libensemble.readthedocs.io/en/main/examples/examples_index.html
.. _funcX: https://funcx.org/
.. _GitHub: https://github.com/Libensemble/libensemble
.. _GitHub Actions: https://github.com/Libensemble/libensemble/actions
.. _here: https://libensemble.readthedocs.io/projects/libe-community-examples/en/latest/
.. _IPAC manuscript: https://doi.org/10.18429/JACoW-ICAP2018-SAPAF03
.. _libEnsemble mailing list: https://lists.mcs.anl.gov/mailman/listinfo/libensemble
.. _libEnsemble Slack page: https://libensemble.slack.com
.. _libE_specs: https://libensemble.readthedocs.io/en/main/data_structures/libE_specs.html
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
.. _PSI/J: https://exaworks.org/psij
.. _psi-j-python: https://github.com/ExaWorks/psi-j-python
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
.. _setuptools: https://setuptools.pypa.io/en/latest/
.. _Spack: https://spack.readthedocs.io/en/latest
.. _Summit: https://www.olcf.ornl.gov/olcf-resources/compute-systems/summit/
.. _Surmise: https://surmise.readthedocs.io/en/latest/index.html
.. _SWIG: http://swig.org/
.. _tarball: https://github.com/Libensemble/libensemble/releases/latest
.. _Tasmanian: https://tasmanian.ornl.gov/
.. _Theta: https://www.alcf.anl.gov/alcf-resources/theta
.. _tqdm: https://tqdm.github.io/
.. _user guide: https://libensemble.readthedocs.io/en/latest/programming_libE.html
.. _VTMOP: https://github.com/Libensemble/libe-community-examples#vtmop
.. _WarpX: https://warpx.readthedocs.io/en/latest/
.. _WarpX + libE scripts: https://warpx.readthedocs.io/en/latest/usage/workflows/libensemble.html
.. _xSDK Extreme-scale Scientific Software Development Kit: https://xsdk.info
