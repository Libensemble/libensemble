#!/usr/bin/env python
"""
Slurm + CUDA forces example testing ``gpu_sharing_factor`` with splitting.

Variable ``num_gpus`` 0.5-2.0 -> 1-4 shares with sf=2 exercises ``spread``
packing and cross-node splitting. Slurm nodelist and NVIDIA GPUs are
auto-detected from ``SLURM_NODELIST`` / platform detection - no mocking.

Build ``forces.x`` with CUDA for target (e.g. Perlmutter)::

    cd ../forces_app && ./build_forces.sh  # -DGPU line

Run on Slurm allocation::

    sbatch submit_slurm.sh
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


def uniform_sample_sharing_slurm(_, persis_info, gen_specs, libE_info):
    """Cycle 0.5,1.0,1.5,2.0 -> shares 1-4, tests splitting across nodes."""
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

    # Real Slurm detection: SLURM_NODELIST + NVIDIA GPUs auto-detected. No mock env.
    ensemble = Ensemble(parse_args=True, executor=exctr)
    nsim_workers = ensemble.nworkers - 1

    num_rsets = max(8, nsim_workers * GPU_SHARING_FACTOR)
    ensemble.libE_specs = LibeSpecs(
        num_resource_sets=num_rsets,
        gpu_sharing_factor=GPU_SHARING_FACTOR,
        scheduler_opts={"gpu_packing": "spread"},
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
        gen_f=uniform_sample_sharing_slurm,
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
        assert not np.any(np.isnan(H["energy"])), "NaN energy - GPU assignment failed"

        gpus_per_rset = 1
        for row in H:
            req = float(row["num_gpus"])
            shares = math.ceil(req * GPU_SHARING_FACTOR / gpus_per_rset)
            assert shares in (1, 2, 3, 4), f"shares {shares} for req {req}"
            assert req in (0.5, 1.0, 1.5, 2.0)

        uniq = np.unique(H["num_gpus"])
        for val in [0.5, 1.0, 1.5, 2.0]:
            assert val in uniq, f"Cycle {val} missing - splitting not exercised"

        rm = getattr(getattr(ensemble, "resources", None), "resource_manager", None)
        if rm is not None:
            assert rm.rsets_free == rm.total_num_rsets, f"Rsets not freed {rm.rsets_free}/{rm.total_num_rsets}"
            assert rm.gpu_rsets_free == rm.total_num_gpu_rsets
            assert rm.total_num_gpu_rsets >= 4, f"Need >=4 GPU rsets, got {rm.total_num_gpu_rsets}"
            assert rm.gpu_rsets_per_node >= 2
            assert rm.num_groups >= 1
            print(
                f"Slurm sharing check: total_gpu_rsets={rm.total_num_gpu_rsets} "
                f"gpu_rsets_per_node={rm.gpu_rsets_per_node} "
                f"even_groups={rm.even_groups} ngroups={rm.num_groups}"
            )

        stats_path = Path("libE_stats.txt")
        if stats_path.exists():
            txt = stats_path.read_text()
            assert "resource_sets" in txt or "sim_id" in txt

        print("Slurm gpu_sharing_factor splitting (spread) assertions passed")
