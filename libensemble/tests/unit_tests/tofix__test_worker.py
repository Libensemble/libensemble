import libensemble.tests.unit_tests.setup as setup
from libensemble.tools.fields_keys import libE_fields
from libensemble.worker import Worker
from libensemble.message_numbers import EVAL_SIM_TAG
import numpy as np


def simfunc():
    pass


def test_worker_init_run():

    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()

    L = exit_criteria["sim_max"]
    # H = np.zeros(L + len(H0), dtype=list(set(libE_fields + sim_specs['out'] + gen_specs['out'] + alloc_specs['out'])))
    H = np.zeros(L, dtype=list(set(libE_fields + sim_specs["out"] + gen_specs["out"])))

    # check
    H["sim_id"][-L:] = -1
    H["sim_started_time"][-L:] = np.inf

    # Create work
    sim_ids = np.zeros(1, dtype=int)

    # For loop - increment sim_ids here

    Work = {"tag": EVAL_SIM_TAG, "persis_info": {}, "libE_info": {"H_rows": sim_ids}, "H_fields": sim_specs["in"]}
    calc_in = H[Work["H_fields"]][Work["libE_info"]["H_rows"]]

    Worker.init_workers(sim_specs, gen_specs)

    workerID = 1

    # Testing two worker routines: init and run
    worker = Worker(workerID)
    worker.run(Work, calc_in)

    print(worker.data)


if __name__ == "__main__":
    test_worker_init_run()
