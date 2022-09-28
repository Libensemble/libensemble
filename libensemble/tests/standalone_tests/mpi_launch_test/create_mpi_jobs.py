import mpi4py

mpi4py.rc.recv_mprobe = False

from mpi4py import MPI  # for libE communicator
import sys
import subprocess

task_nprocs = sys.argv[1]

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
runline = "mpirun -np ".split()
runline.append(task_nprocs)
runline.append("python")
runline.append("helloworld.py")

if rank == 0:
    print(f"Total sub-task procs: {size * int(task_nprocs)}")
    print(f"Total procs (parent + sub-tasks): {size * (int(task_nprocs) + 1)}")

# print("Rank {}: {}".format(rank, " ".join(runline)))
output = "task_" + str(rank) + ".out"
p = subprocess.Popen(runline, stdout=open(output, "w"), shell=False)
p.wait()
