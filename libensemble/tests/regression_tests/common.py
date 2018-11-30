"""
Common plumbing for regression tests
"""

import sys

def parse_args():
    if len(sys.argv) > 1 and sys.argv[1] == "--processes":
        is_master = True
        nworkers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        libE_specs = {'nprocesses': nworkers,
                      'comms': 'local'}
    elif len(sys.argv) > 1 and sys.argv[1] == "--tcp":
        is_master = True
        nworkers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        cmd = [sys.executable, sys.argv[0], "client",
               "{manager_ip}", "{manager_port}", "{authkey}",
               "{workerID}", str(nworkers)]
        libE_specs = {'nprocesses': nworkers,
                      'worker_cmd': cmd,
                      'comms': 'tcp'}
    elif len(sys.argv) > 1 and sys.argv[1] == "client":
        is_master = False
        nworkers = int(sys.argv[6])
        libE_specs = {'ip': sys.argv[2],
                      'port': int(sys.argv[3]),
                      'authkey': sys.argv[4].encode('utf-8'),
                      'workerID': int(sys.argv[5]),
                      'nprocesses': nworkers,
                      'comms': 'tcp'}
    else:
        from mpi4py import MPI
        nworkers = MPI.COMM_WORLD.Get_size()-1
        is_master = MPI.COMM_WORLD.Get_rank() == 0
        libE_specs = {'comm': MPI.COMM_WORLD, 'color': 0, 'comms': 'mpi'}

    return nworkers, is_master, libE_specs
