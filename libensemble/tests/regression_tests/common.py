"""
Common plumbing for regression tests
"""

import os
import os.path


def create_node_file(num_nodes, name="node_list"):
    """Create a nodelist file"""
    if os.path.exists(name):
        os.remove(name)
    with open(name, "w") as f:
        for i in range(1, num_nodes + 1):
            f.write("node-" + str(i) + "\n")
        f.flush()
        os.fsync(f)


def mpi_comm_excl(exc=[0], comm=None):
    "Exclude ranks from a communicator for MPI comms."
    from mpi4py import MPI, rc

    if rc.initialize is False and not MPI.Is_initialized():
        MPI.Init()  # For unit tests, since auto-init disabled to prevent test_executor issues

    parent_comm = comm or MPI.COMM_WORLD
    parent_group = parent_comm.Get_group()
    new_group = parent_group.Excl(exc)
    mpi_comm = parent_comm.Create(new_group)
    return mpi_comm, MPI.COMM_NULL


def mpi_comm_split(num_parts, comm=None):
    "Split COMM_WORLD into sub-communicators for MPI comms."
    from mpi4py import MPI, rc

    if rc.initialize is False and not MPI.Is_initialized():
        MPI.Init()

    parent_comm = comm or MPI.COMM_WORLD
    parent_size = parent_comm.Get_size()
    key = parent_comm.Get_rank()
    row_size = parent_size // num_parts
    sub_comm_number = key // row_size
    sub_comm = parent_comm.Split(sub_comm_number, key)
    return sub_comm, sub_comm_number


def build_simfunc():
    import subprocess

    # Build simfunc
    # buildstring='mpif90 -o my_simtask.x my_simtask.f90' # On cray need to use ftn
    buildstring = "mpicc -o my_simtask.x ../unit_tests/simdir/my_simtask.c"
    # subprocess.run(buildstring.split(),check=True) #Python3.5+
    subprocess.check_call(buildstring.split())


def build_borehole():
    import subprocess

    buildstring = "gcc -o borehole.x ../unit_tests/simdir/borehole.c -lm"
    subprocess.check_call(buildstring.split())
