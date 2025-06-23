import os
import sys

import numpy as np
from simf import run_t3d_and_gx

from libensemble import Ensemble
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.executors import MPIExecutor
from libensemble.gen_funcs.persistent_sampling import persistent_uniform as gen_f
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

if __name__ == "__main__":
    exctr = MPIExecutor()

    sim_app = "/global/u2/j/jmlarson/jai/t3d/.venv/bin/t3d"

    if not os.path.isfile(sim_app):
        sys.exit(f"Application not found: {sim_app}")

    exctr.register_app(full_path=sim_app, app_name="t3d")

    num_workers = 1


    sim_input_dir = "test-w7x-gx_simple"

    libE_specs = LibeSpecs(
        nworkers=num_workers,
        gen_on_manager=True,
        sim_dirs_make=True,
        sim_input_dir=sim_input_dir,
    )

    sim_specs = SimSpecs(
        sim_f=run_t3d_and_gx,
        inputs=["x"],
        outputs=[("f", float)],
        user={"input_filename": "test-w7x-gx.in", "input_names": ["H_height", "H_width"]},
    )

    n = 2
    gen_specs = GenSpecs(
        gen_f=gen_f,
        inputs=[],
        persis_in=["sim_id", "f"],
        outputs=[("x", float, (2,))],
        user={
            "initial_batch_size": num_workers,
            "lb": np.array([0, 0]),
            "ub": np.array([3, 3]),
        },
    )

    alloc_specs = AllocSpecs(
        alloc_f=alloc_f,
        user={"async_return": True},
    )

    exit_criteria = ExitCriteria(sim_max=2)

    ensemble = Ensemble(
        libE_specs=libE_specs,
        gen_specs=gen_specs,
        sim_specs=sim_specs,
        alloc_specs=alloc_specs,
        exit_criteria=exit_criteria,
        executor=exctr
    )

    ensemble.add_random_streams()
    H, persis_info, flag = ensemble.run()

    if ensemble.is_manager:
        print("First 3:", H[["sim_id", "x", "f"]][:3])
        print("Last 3:", H[["sim_id", "x", "f"]][-3:])
        ensemble.save_output(__file__)

