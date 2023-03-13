#!/usr/bin/env python
"""
Parallel Hello World
"""

if __name__ == "__main__":
    import sys

    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()

    sys.stdout.write("Hello, World! I am process %d of %d on %s.\n" % (rank, size, name))
