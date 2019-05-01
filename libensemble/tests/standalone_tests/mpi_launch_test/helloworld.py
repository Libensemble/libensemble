#!/usr/bin/env python
"""
Parallel Hello World
"""

from mpi4py import MPI
import sys
import time

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

time.sleep(2)
sys.stdout.write("Hello, World after sleep! I am process %d of %d on %s.\n" % (rank, size, name))
