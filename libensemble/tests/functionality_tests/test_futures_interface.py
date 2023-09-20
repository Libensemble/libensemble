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
from libensemble.gen_funcs.sampling import latin_hypercube_sample as gen_f

# Import libEnsemble items for this test
from libensemble.sim_funcs.one_d_func import one_d_example
from libensemble.specs import ExitCriteria, GenSpecs
from libensemble.tools import add_unique_random_streams

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    gen_specs = GenSpecs(
        gen_f=gen_f,
        outputs=[("x", float, (1,))],
        user={
            "gen_batch_size": 500,
            "lb": np.array([-3]),
            "ub": np.array([3]),
        },
    )

    with Ensemble(parse_args=True, gen_specs=gen_specs, exit_criteria=ExitCriteria(sim_max=1001)) as sampling:

        sampling.persis_info = add_unique_random_streams({}, sampling.nworkers + 1)

        future = sampling.submit(one_d_example, [("f", float)], "x")
        while not future.done():
            print("waiting")
            time.sleep(0.1)

        print(future.result()[0][:10])
