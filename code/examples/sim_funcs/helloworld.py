#!/usr/bin/env python
"""
Parallel Hello World
"""
from __future__ import print_function

from mpi4py import MPI
import sys
import time

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
if len(sys.argv) >= 2:
    try:
        sleep_time=float(sys.argv[1])
    except:
        sys.stdout.write("WARNING: Helloworld sleep time arg is not compatible with float")
        sleep_time = 1.0
else:
    sleep_time = 1.0
    
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

sys.stdout.write("Hello World rank %d of %d going to sleep for %.2f seconds on %s\n" % (rank, size, sleep_time, name))
sys.stdout.flush()
#eprint("Hello World rank %d of %d going to sleep for %.2f seconds on %s\n" % (rank, size, sleep_time, name))

time.sleep(sleep_time)

sys.stdout.write("Hello, World rank %d of %d waking up on %s.\n" % (rank, size, name))
sys.stdout.flush()
#eprint("Hello, World rank %d of %d waking up on %s.\n" % (rank, size, name))
