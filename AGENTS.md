Agent Contributor Guidelines and Information
============================================

Read the ``README.rst`` for an overview of libEnsemble.

Repository Layout
-----------------

- ``libensemble/`` - source code.
-   ``/alloc_funcs`` - allocation functions. Policies for passing work between the manager and workers.
-   ``/comms`` - modules and abstractions for communication between the manager and workers.
-   ``/executors`` - an interface for launching executables, often simulations.
-   ``/gen_classes`` - generators that adhere to the `gest-api` standard.
                       Recommended over entries from ``/gen_funcs`` that perform similar functionality.
-   ``/gen_funcs`` - generator functions. Modules for producing points for simulations.
-   ``/resources`` - classes and functions for managing compute resources for MPI tasks, libensemble workers.
-   ``/sim_funcs`` - simulator functions. Modules for running simulations or performing experiments.
-   ``/tests`` - tests.
    - ``/functionality_tests`` primarily tests libEnsemble code only.
    - ``/regression_tests`` tests libEnsemble code with external code. Often more closely resembles actual use-cases.
    - ``/unit_tests`` tests for individual modules.
-   ``/tools`` - tools. misc functions and classes to ease development.
-   ``/utils`` - utilities. misc functions and classes used internally by multiple modules.
-   ``ensemble.py`` - The primary interface for parameterizing and running libEnsemble.
-   ``generators.py`` - base classes for generators that adhere to the `gest-api` standard.
-   ``history.py`` - module for recording points that have been generated and simulation results. NumPy array.
-   ``libE.py`` - libE main file. Previous primary interface for parameterizing and running libEnsemble.
-   ``logger.py`` - logging configuration
-   ``manager.py`` - module for maintaining the history array and passing points between the workers.
-   ``message_numbers.py`` - constants that represent states of the ensemble.
-   ``specs.py`` - Dataclasses for parameterizing the ensemble.
-   ``worker.py`` - module for running generators and simulators. Communicates with the manager.
-   ``version.py`` - version file.

- ``.github/`` - GitHub actions. See ``.github/workflows/`` for the CI.
- ``.docs/`` - Documentation. Check here first for information before reading the source code.
- ``examples/`` - Symlinks to examples further inside the ``libensemble/`` directory.

Other files in the root directory should be self-documenting.

Familiarize yourself with ``libensemble/tests/regression_tests/test_1d_sampling.py`` for a simple example of the libEnsemble interface.

General Guidelines
------------------

- If using a generator that adheres to the `gest-api` standard, use the ``start_only_persistent`` allocation function.
- An MPI distribution is not required for libEnsemble to run, but is required to use the ``MPIExecutor``.  ``mpich`` is recommended.
- New tests are heavily encouraged for new features, bug fixes, or integrations. See ``libensemble/tests/regression_tests`` for examples.
- Never use destructive git commands unless explicitly requested.
- Code is in the ``black`` style. This should be enforced by ``pre-commit``.
- Read ``CONTRIBUTING.md`` for more information.

Development Environment
-----------------------

- ``pixi`` is the recommended environment manager for libEnsemble development.  See ``pyproject.toml`` for the list
of dependencies and the available testing environments.

- Enter the development environment with ``pixi shell -e dev``. This environment contains the most common dependencies for development and testing.

- If ``pixi`` is not available or not preferred by the user, ``pip`` can be used instead, but the user will need to manually install other dependencies.

- If committing, use ``pre-commit`` to ensure that code style and formatting are consistent.  See ``.pre-commit-config.yaml`` for
the configuration and ``pyproject.toml`` for other configuration.

Testing
-------

- Run tests with the ``run-tests.py`` script.  See ``libensemble/tests/run-tests.py`` for usage information.

- Some tests require third party software to be installed. When developing a feature or fixing a bug the entire test suite may not necessarily need to be run,
since it will be on Github Actions.

- Individual unit tests can be run with ``pytest``.
