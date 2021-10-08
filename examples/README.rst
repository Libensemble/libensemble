For a simple tutorial with a complete libEnsemble workflow, see the simple sin tutorial::

    cd cd tutorials/simple_sine
    python tutorial_calling.py

Many example scripts are available from the repository at ``libensemble/tests/regression_tests/``.

If you wish to clone libEnsemble to try the examples instead of installing from a remote::

    git clone git@github.com:Libensemble/libensemble.git
    cd libensemble
    pip install -e .
    cd libensemble/tests/regression_tests

Now can run using::

    python test_uniform_sampling.py --comms local --nworkers 3

The command line arguments parsed by a parse_args module you can find in the scripts. If you
have ``mpi4py`` installed you can alternatively run with::

    mpirun -np 4 python test_uniform_sampling.py

You will find many more examples here.
