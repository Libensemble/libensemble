"""
Tests libEnsemble's inverse_bayes generator function

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_inverse_bayes_example.py
   python test_inverse_bayes_example.py --nworkers 3
   python test_inverse_bayes_example.py --nworkers 3 --comms tcp

Debugging:
   mpiexec -np 4 xterm -e "python inverse_bayes_example.py"

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 3 4

import numpy as np

from libensemble import Ensemble
from libensemble.alloc_funcs.inverse_bayes_allocf import only_persistent_gens_for_inverse_bayes as alloc_f
from libensemble.gen_funcs.persistent_inverse_bayes import persistent_updater_after_likelihood as gen_f
from libensemble.sim_funcs.inverse_bayes import likelihood_calculator as sim_f
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, SimSpecs
from libensemble.tools import add_unique_random_streams

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    # Parse args for test code
    bayes_test = Ensemble(
        parse_args=True,
        sim_specs=SimSpecs(
            sim_f=sim_f,
            inputs=["x"],
            out=[("like", float)],
        ),
        gen_specs=GenSpecs(
            gen_f=gen_f,
            out=[
                ("x", float, 2),
                ("batch", int),
                ("subbatch", int),
                ("prior", float),
                ("prop", float),
                ("weight", float),
            ],
            user={
                "lb": np.array([-3, -2]),
                "ub": np.array([3, 2]),
                "subbatch_size": 3,
                "num_subbatches": 2,
                "num_batches": 10,
            },
        ),
        alloc_specs=AllocSpecs(alloc_f=alloc_f),
    )

    bayes_test.persis_info = add_unique_random_streams({}, bayes_test.nworkers + 1)
    gen_user = bayes_test.gen_specs.user
    val = gen_user["subbatch_size"] * gen_user["num_subbatches"] * gen_user["num_batches"]
    bayes_test.exit_criteria = ExitCriteria(sim_max=val, wallclock_max=300)

    # Perform the run
    H, _, flag = bayes_test.run()

    if bayes_test.is_manager:
        assert flag == 0
        # Change the last weights to correct values (H is a list on other cores and only array on manager)
        ind = 2 * gen_user["subbatch_size"] * gen_user["num_subbatches"]
        H[-ind:] = H["prior"][-ind:] + H["like"][-ind:] - H["prop"][-ind:]
        assert len(H) == 60, "Failed"
