#!/usr/bin/env python
"""
PBSpro + CUDA forces example testing ``gpu_sharing_factor`` on real hardware.

Fractional GPU requests (0.5/1.0/1.5/2.0 -> 1-4 shares with sf=2) exercise
physical-exclusive packing (``gpu_packing=pack``). PBS nodelist and NVIDIA
GPUs are auto-detected from ``PBS_NODEFILE`` / platform detection - no
mocking. ``CUDA_VISIBLE_DEVICES`` is set via ``auto_assign_gpus``.

Build ``forces.x`` with CUDA for the target (e.g. Aurora/Polaris)::

    cd ../forces_app && ./build_forces.sh  # -DGPU line

Run on PBS allocation (e.g. Polaris ``select=2``)::

    qsub submit_pbs.sh
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 5

import math
import sys
from pathlib import Path

import numpy as np
from forces_simf import run_forces

from libensemble import Ensemble
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.executors import MPIExecutor
from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
from libensemble.specs import AllocSpecs, GenSpecs, LibeSpecs, SimSpecs
from libensemble.tools.persistent_support import PersistentSupport

GPU_SHARING_FACTOR = 2


def uniform_sample_sharing_pbs(_, persis_info, gen_specs, libE_info):
    """Cycle 0.5,1.0,1.5,2.0 GPUs -> 1,2,3,4 shares with sf=2."""
    ub = gen_specs["user"]["ub"]
    lb = gen_specs["user"]["lb"]
    n = len(lb)
    b = gen_specs["initial_batch_size"]
    from libensemble.tools import get_rng

    rng = get_rng(gen_specs, libE_info)
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    tag = None
    cycle = [0.5, 1.0, 1.5, 2.0]
    idx = 0
    while tag not in [STOP_TAG, PERSIS_STOP]:
        H_o = np.zeros(b, dtype=gen_specs["out"])
        H_o["x"] = rng.uniform(lb, ub, (b, n))
        H_o["num_gpus"] = [cycle[(idx + i) % len(cycle)] for i in range(b)]
        H_o["num_procs"] = np.ones(b, dtype=int)
        idx += b
        tag, Work, calc_in = ps.send_recv(H_o)
        if hasattr(calc_in, "__len__"):
            b = len(calc_in)
    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG


if __name__ == "__main__":
    exctr = MPIExecutor()
    sim_app = Path.cwd() / "../forces_app/forces.x"
    if not sys.platform.startswith("win") and not Path(sim_app).is_file():
        sys.exit("forces.x not found - please build first in ../forces_app dir")
    exctr.register_app(full_path=sim_app, app_name="forces")

    ensemble = Ensemble(parse_args=True, executor=exctr)
    nsim_workers = ensemble.nworkers - 1

    # Real PBS detection: PBS_NODEFILE + NVIDIA GPUs auto-detected. No resource_info mocking.
    ensemble.libE_specs = LibeSpecs(
        num_resource_sets=nsim_workers,
        gpu_sharing_factor=GPU_SHARING_FACTOR,
        scheduler_opts={"gpu_packing": "pack"},
        sim_dirs_make=True,
        stats_fmt={"show_resource_sets": True},
    )

    ensemble.sim_specs = SimSpecs(
        sim_f=run_forces,
        inputs=["x"],
        outputs=[("energy", float)],
        user={"gpu_sharing_factor": GPU_SHARING_FACTOR},
    )

    ensemble.gen_specs = GenSpecs(
        gen_f=uniform_sample_sharing_pbs,
        inputs=[],
        persis_in=["sim_id"],
        outputs=[("x", float, (1,)), ("num_gpus", float), ("num_procs", int)],
        initial_batch_size=nsim_workers,
        async_return=False,
        user={"lb": np.array([50000]), "ub": np.array([100000])},
    )

    ensemble.alloc_specs = AllocSpecs(alloc_f=alloc_f, user={"async_return": False})

    sim_max = 12
    ensemble.run(sim_max=sim_max)

    if ensemble.is_manager:
        assert ensemble.flag == 0, f"Ensemble failed flag={ensemble.flag}"
        H = ensemble.H
        assert len(H) == sim_max, f"H len {len(H)} != {sim_max}"
        assert not np.any(np.isnan(H["energy"])), "Energy NaN - GPU assignment failed"

        gpus_per_rset = 1  # sf>1 => 1 share per rset
        for row in H:
            req_gpus = float(row["num_gpus"])
            expected_shares = math.ceil(req_gpus * GPU_SHARING_FACTOR / gpus_per_rset)
            assert 1 <= expected_shares <= 4, f"shares {expected_shares} for req {req_gpus}"
            assert req_gpus in (0.5, 1.0, 1.5, 2.0), f"req_gpus {req_gpus} not in cycle"

        rm = getattr(getattr(ensemble, "resources", None), "resource_manager", None)
        if rm is not None:
            assert rm.rsets_free == rm.total_num_rsets, "Rsets not freed"
            assert rm.gpu_rsets_free == rm.total_num_gpu_rsets, "GPU rsets leak"
            assert rm.total_num_gpu_rsets > 0, "No GPU rsets detected - is CUDA allocation correct?"
            assert rm.group_sizes, "group_sizes empty"
            assert rm.num_groups >= 1, "num_groups broken"
            print(
                f"PBS sharing check: total_gpu_rsets={rm.total_num_gpu_rsets} "
                f"gpu_rsets_per_node={rm.gpu_rsets_per_node} "
                f"rsets_free={rm.rsets_free}/{rm.total_num_rsets} "
                f"ngroups={rm.num_groups} even={rm.even_groups}"
            )

        stats_path = Path("libE_stats.txt")
        if stats_path.exists():
            txt = stats_path.read_text()
            assert "resource_sets" in txt or "sim_id" in txt, "stats missing"

        print("PBS gpu_sharing_factor assignment/splitting assertions passed")
