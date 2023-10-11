"""
Runs libEnsemble with Latin hypercube sampling on a simple 1D problem

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_1d_sampling.py
   python test_1d_sampling.py --nworkers 3 --comms local
   python test_1d_sampling.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4

import time

import numpy as np

from libensemble import Ensemble
from libensemble.sim_funcs.borehole import borehole, gen_borehole_input


def create_input_work():
    n_samp = 1000
    H0 = np.zeros(n_samp, dtype=[("x", float, 8), ("sim_id", int), ("sim_started", bool)])
    np.random.seed(0)
    H0["x"] = gen_borehole_input(n_samp)
    H0["sim_id"] = range(n_samp)
    H0["sim_started"] = False
    return H0


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    input_array = create_input_work()

    with Ensemble(parse_args=True) as runner:
        future = runner.submit(borehole, input_array, ["x"], [("f", float)])
        while not future.done():
            print("waiting")
            time.sleep(0.1)

        print(future.result()[0][:10])
