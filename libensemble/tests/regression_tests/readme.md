## Running regression tests

To run the test test_1d_sampling.py with 3 workers:

    python test_1d_sampling.py -n 3

To run with `mpi4py` comms use:

    mpirun -np 4 python test_1d_sampling.py

Note the extra process for the manager.

Some regression tests require external dependencies.
