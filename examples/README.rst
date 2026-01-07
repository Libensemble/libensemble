For a simple tutorial with a complete libEnsemble workflow, see the simple sine tutorial::

    cd tutorials/simple_sine
    python tutorial_calling.py

Many example scripts are available from the repository in ``libensemble/tests/regression_tests/``.

If you wish to clone libEnsemble to try the examples instead of installing from a remote::

    git clone git@github.com:Libensemble/libensemble.git
    cd libensemble
    pip install -e .
    cd libensemble/tests/regression_tests

Any of the tests can be run similarly to the following (-n is also short for --nworkers)::

    python test_uniform_sampling.py --nworkers 3

The command line arguments are parsed by a ``parse_args`` module within each of the scripts. If you
have ``mpi4py`` installed you can alternatively run with::

    mpirun -np 4 python test_uniform_sampling.py

You will find many more examples here.
