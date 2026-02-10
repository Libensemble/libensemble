"""
Tests libEnsemble with Optimas Multitask Ax Generator

Runs an initial ensemble, followed by another using the first as an H0.

*****currently fixing nworkers to batch_size*****

Execute via one of the following commands (e.g. 4 workers):
   mpiexec -np 5 python test_optimas_ax_multitask.py
   python test_optimas_ax_multitask.py -n 4

When running with the above commands, the number of concurrent evaluations of
the objective function will be 4 as the generator is on the manager.

"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true

import numpy as np
from gest_api.vocs import VOCS

from optimas.core import Task
from optimas.generators import AxMultitaskGenerator

from libensemble import Ensemble
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs


def eval_func_multitask(input_params):
    """Evaluation function for task1 or task2 in multitask test"""
    print(f'input_params: {input_params}')
    x0 = input_params["x0"]
    x1 = input_params["x1"]
    trial_type = input_params["trial_type"]

    if trial_type == "task_1":
        result = -(x0 + 10 * np.cos(x0)) * (x1 + 5 * np.cos(x1))
    else:
        result = -0.5 * (x0 + 10 * np.cos(x0)) * (x1 + 5 * np.cos(x1))

    output_params = {"f": result}
    return output_params


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    n = 2
    batch_size = 2

    libE_specs = LibeSpecs(gen_on_manager=True, nworkers=batch_size)

    vocs = VOCS(
        variables={
            "x0": [-50.0, 5.0],
            "x1": [-5.0, 15.0],
            "trial_type": {"task_1", "task_2"},
        },
        objectives={"f": "MAXIMIZE"},
    )

    sim_specs = SimSpecs(
        simulator=eval_func_multitask,
        vocs=vocs,
    )

    alloc_specs = AllocSpecs(alloc_f=alloc_f)
    exit_criteria = ExitCriteria(sim_max=15)

    H0 = None  # or np.load("multitask_first_pass.npy")
    for run_num in range(2):
        print(f"\nRun number: {run_num}")
        task1 = Task("task_1", n_init=2, n_opt=1)
        task2 = Task("task_2", n_init=5, n_opt=3)
        gen = AxMultitaskGenerator(vocs=vocs, hifi_task=task1, lofi_task=task2)

        gen_specs = GenSpecs(
            generator=gen,
            batch_size=batch_size,
            vocs=vocs,
        )

        workflow = Ensemble(
            libE_specs=libE_specs,
            sim_specs=sim_specs,
            alloc_specs=alloc_specs,
            gen_specs=gen_specs,
            exit_criteria=exit_criteria,
            H0=H0,
        )

        H, _, _ = workflow.run()

        if run_num == 0:
            H0 = H
            workflow.save_output("multitask_first_pass", append_attrs=False)  # Allows restart only run

        if workflow.is_manager:
            if run_num == 1:
                workflow.save_output("multitask_with_H0")
                print(f"Second run completed: {len(H)} simulations")
