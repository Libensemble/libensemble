"""
Runs libEnsemble with Latin hypercube sampling on a simple 1D problem using
the libEnsemble yaml interface

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_1d_sampling_from_yaml.py
   python test_1d_sampling_from_yaml.py --nworkers 3
   python test_1d_sampling_from_yaml.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4
# TESTSUITE_EXTRA: true

import numpy as np

from libensemble import Ensemble

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    sampling = Ensemble(parse_args=True)
    sampling.from_json("1d_sampling.json")
    sampling.from_toml("1d_sampling.toml")
    sampling.from_yaml("1d_sampling.yaml")

    sampling.gen_specs.user.update(
        {
            "lb": np.array([-3]),
            "ub": np.array([3]),
        }
    )

    sampling.add_random_streams()

    # Perform the run
    sampling.run()

    if sampling.is_manager:
        assert len(sampling.H) >= 501
        print("\nlibEnsemble with random sampling has generated enough points")
    sampling.save_output(__file__)
