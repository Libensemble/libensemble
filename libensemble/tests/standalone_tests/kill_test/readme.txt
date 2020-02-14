Kill tests
==========

This is to test the killing of MPI tasks. There are two tests and different kill
methods which can be selected on the command line. The aim is that kills in
libEnsemble will be automatically configured to work correctly based on system
detection. However, this test will show what type of kill works for a given
system, and this may be configurable in libEnsemble in the future.

Test: sleep_and_print

The default test (sleep_and_print), should automatically test if all processes
of an MPI job are correctly killed and also that a second job can be launched
and killed.

Launching MPI tasks which write from each MPI task, at regular intervals, to an
output file (see sleep_and_print.c).

This test launches a job, then kills after a few seconds, and then monitors
output file to see if output continues. If the first job is successfully killed,
a second is launched and the test repeated.

Instructions
------------

Build the C program:
    mpicc -g -o sleep_and_print.x sleep_and_print.c
    OR (e.g. on Theta/Cori):
    cc -g -o sleep_and_print.x sleep_and_print.c

Either run on local node - or create an allocation of nodes and run::

    python killtest.py <kill_type> <num_nodes> <num_procs_per_node>

where kill_type currently is 1 or 2. [1 is the original kill - 2 is using
group ID approach]

E.g Single node with 4 processes:
--------------------------------
kill 1: python killtest.py 1 1 4
kill 2: python killtest.py 2 1 4
--------------------------------

E.g Two nodes with 2 processes each:
--------------------------------
kill 1: python killtest.py 1 2 2
kill 2: python killtest.py 2 2 2
--------------------------------

If the test fails, an assertion error will occur.

Output files are produced out_0.txt and out_1.txt (one for each job). These can
be deleted between runs.
