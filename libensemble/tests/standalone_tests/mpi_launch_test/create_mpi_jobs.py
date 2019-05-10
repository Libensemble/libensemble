import mpi4py
mpi4py.rc.recv_mprobe = False

from mpi4py import MPI  # for libE communicator
import sys
import subprocess

job_nprocs = sys.argv[1]

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
runline = "mpirun -np ".split()
runline.append(job_nprocs)
runline.append('python')
runline.append('helloworld.py')

if rank == 0:
    print("Total sub-job procs: {}".format(size*int(job_nprocs)))
    print("Total procs (parent + sub-jobs): {}".format(size*(int(job_nprocs)+1)))

# print("Rank {}: {}".format(rank, " ".join(runline)))
output = 'job_' + str(rank) + '.out'
p = subprocess.Popen(runline, stdout=open(output, 'w'), shell=False)
p.wait()
