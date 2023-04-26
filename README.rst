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

========================================================
A complete toolkit for dynamic ensembles of calculations
========================================================

Easy construction of *adaptive*, *scalable* workflows that connect "deciders" to experiments or simulations.

• **Adaptive ensembles**: Generate parallel tasks *on-the-fly* based on previous computations.
• **Portable, extreme scaling**: Run on or across **laptops**, **clusters**, and **leadership-class machines**.
• **Dynamic resource management**: Adaptively assign and reassign resources (including **GPUs**) to tasks.
• **Application monitoring**: Ensemble members can **run, monitor, and cancel apps**.
• **Coordinated data-flow between tasks**: libEnsemble can pass data between stateful ensemble members.
• **Low start-up cost**: No additional background services or processes required.

libEnsemble is *especially effective* at solving design, decision, and inference problems on massively parallel resources.

.. before_dependencies_rst_tag

Installation
============

Install libEnsemble and its dependencies from PyPI_ using pip::

    pip install libensemble

Install libEnsemble with Conda_ from the conda-forge_ channel::

    conda config --add channels conda-forge
    conda install -c conda-forge libensemble

Install libEnsemble using the Spack_ distribution::

    spack install py-libensemble

libEnsemble is included in the `xSDK Extreme-scale Scientific Software Development Kit`_.
..  Install the xSDK and load the environment with::

.. ..     spack install xsdk
.. ..     spack load -r xsdk

.. Dependencies
.. ============

.. libEnsemble performs best on Unix-like systems like Linux and macOS. See the
.. FAQ_ for more information.

.. **Required dependencies**:

.. * Python_ 3.8 or above
.. * NumPy_
.. * psutil_
.. * setuptools_
.. * pydantic_

.. When using  ``mpi4py`` for libEnsemble communications:

.. * A functional MPI 1.x/2.x/3.x implementation, such as MPICH_, built with shared/dynamic libraries
.. * mpi4py_ v2.0.0 or above

.. **Optional dependencies**:

.. * Balsam_ - Manage and submit applications to the Balsam service with our BalsamExecutor
.. * pyyaml_ and tomli_ - Parameterize libEnsemble via yaml or toml
.. * funcX_ - Submit simulation or generator function instances to remote funcX endpoints
.. * `psi-j-python`_ and `tqdm`_ - Use `liberegister` and `libesubmit` to submit libEnsemble jobs to any scheduler

Resources
=========

**Support:**

- Open issues or ask questions on GitHub_.
- Email questions or request `libEnsemble Slack page`_ access from ``libEnsemble@lists.mcs.anl.gov``.
- Join the `libEnsemble mailing list`_ for updates about new releases.

**Further Information:**

- Documentation is provided by ReadtheDocs_.
- Contributions_ to libEnsemble are welcome.
- Production functions and complete workflows can be viewed and submitted in the libEnsemble `Community Examples repository`_.

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

.. **Example Compatible Packages**

.. .. before_examples_rst_tag

.. libEnsemble and the `Community Examples repository`_ include example generator
.. functions for the following libraries:

.. - APOSMM_ Asynchronously parallel optimization solver for finding multiple minima. Supported local optimization routines include:

..   - DFO-LS_ Derivative-free solver for (bound constrained) nonlinear least-squares minimization
..   - NLopt_ Library for nonlinear optimization, providing a common interface for various methods
..   - scipy.optimize_ Open-source solvers for nonlinear problems, linear programming,
..     constrained and nonlinear least-squares, root finding, and curve fitting.
..   - `PETSc/TAO`_ Routines for the scalable (parallel) solution of scientific applications

.. - DEAP_ Distributed evolutionary algorithms
.. - Distributed optimization methods for minimizing sums of convex functions. Methods include:

..   - Primal-dual sliding (https://arxiv.org/pdf/2101.00143).
..   - Distributed gradient descent with gradient tracking (https://arxiv.org/abs/1908.11444).
..   - Proximal sliding (https://arxiv.org/abs/1406.0919).

.. - ECNoise_ Estimating Computational Noise in Numerical Simulations
.. - Surmise_ Modular Bayesian calibration/inference framework
.. - Tasmanian_ Toolkit for Adaptive Stochastic Modeling and Non-Intrusive ApproximatioN
.. - VTMOP_ Fortran package for large-scale multiobjective multidisciplinary design optimization

.. libEnsemble has also been used to coordinate many computationally expensive
.. simulations. Select examples include:

.. - OPAL_ Object Oriented Parallel Accelerator Library. (See this `IPAC manuscript`_.)
.. - WarpX_ Advanced electromagnetic particle-in-cell code. (See example `WarpX + libE scripts`_.)

.. See a complete list of `example user scripts`_.

.. after_resources_rst_tag

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
.. _FAQ: https://libensemble.readthedocs.io/en/main/FAQ.html
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
.. _pydantic: https://pydantic-docs.helpmanual.io/
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
.. _Tasmanian: https://tasmanian.ornl.gov/
.. _Theta: https://www.alcf.anl.gov/alcf-resources/theta
.. _tomli: https://pypi.org/project/tomli/
.. _tqdm: https://tqdm.github.io/
.. _user guide: https://libensemble.readthedocs.io/en/latest/programming_libE.html
.. _VTMOP: https://github.com/Libensemble/libe-community-examples#vtmop
.. _WarpX: https://warpx.readthedocs.io/en/latest/
.. _WarpX + libE scripts: https://warpx.readthedocs.io/en/latest/usage/workflows/libensemble.html
.. _xSDK Extreme-scale Scientific Software Development Kit: https://xsdk.info
