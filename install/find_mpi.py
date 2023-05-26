import os

import mpi4py
from mpi4py import MPI

path = mpi4py.__path__[0]
print("\nmpi4py path found is:", path)

configfile = os.path.join(path, "mpi.cfg")
print("\nShowing config file: ", configfile, "\n")

with open(configfile, "r") as confile_handle:
    print(confile_handle.read())

with open(configfile, "r") as infile:
    for line in infile:
        if line.startswith("mpicc ="):
            mpi4py_mpicc = line[8:-1]
            cmd_line = str(mpi4py_mpicc) + " -v"
            print(cmd_line, ":\n")
            os.system(cmd_line)
            break

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

assert size == 1
assert rank == 0
assert len(name)

print("Passed")
