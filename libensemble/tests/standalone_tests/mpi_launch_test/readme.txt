This test creates an mpi4py job where each processor launches MPI tasks on a
node. This is an essential capability for libEnsemble with mpi4py.

You specify the size of the outer tasks and inner tasks as follows:

    mpirun -np 16 python create_mpi_tasks.py 4

runs a 16 way job, each launching a 4 processor python hello_world with a short
sleep.

This is a good test to run to ensure basic functioning on a system, including
nested launching.

The test should create an output file (job_N.out) for each outer rank. A
hello_world line is printed in these for each inner rank.

Additionally it will print the total number of processes being launched
(including outer and inner). This can help test, for example, if
oversubscription is supported.
