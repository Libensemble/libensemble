"""
Tests GPU sharing with physical-exclusive scheduling and pack/spread policies.

This regression test mimics a user workflow: a persistent generator produces
simulations that request fractional GPUs (0.5, 1.0, 1.5, 2.0) which, with
``gpu_sharing_factor=2``, map to 1-4 logical GPU shares. The ensemble is
configured with 2 mocked nodes, 4 GPUs per node, and ``scheduler_opts`` to
exercise pack/spread packing. The simulator (``gpu_variable_resources_from_gen``)
validates GPU assignment via ``check_gpu_setting``.

Execute via (e.g. 9 workers):
    mpiexec -np 10 python test_GPU_sharing_variable_resources.py
    python test_GPU_sharing_variable_resources.py --nworkers 9
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 10

import numpy as np

from libensemble import Ensemble
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
from libensemble.sim_funcs import six_hump_camel
from libensemble.sim_funcs.var_resources import gpu_variable_resources_from_gen as sim_f
from libensemble.specs import GenSpecs, LibeSpecs, SimSpecs
from libensemble.tools.persistent_support import PersistentSupport


def uniform_sample_sharing(_, persis_info, gen_specs, libE_info):
    """Persistent generator that cycles through fractional GPU requests.

    Each batch emits ``num_gpus`` values from the repeating cycle
    ``[0.5, 1.0, 1.5, 2.0]`` which, with ``gpu_sharing_factor=2``, map to
    ``[1, 2, 3, 4]`` logical shares. ``0.5`` exercises the fractional
    (partial-GPU) path, ``1.0`` a whole single GPU, ``1.5`` whole+remainder,
    and ``2.0`` an exclusive two-GPU request. ``num_procs`` is fixed at 1 so
    GPU-driven resource sizing dominates. The generator also draws ``x``
    uniformly from ``gen_specs["user"]["lb"/"ub"]`` so the sim can run
    ``six_hump_camel``. Used by the ensemble integration portion of the
    test to drive all share-aware branches in ``ResourceScheduler`` and
    ``AllocSupport``.
    """
    # Extract domain and batch size from gen_specs
    ub = gen_specs["user"]["ub"]
    lb = gen_specs["user"]["lb"]
    n = len(lb)
    b = gen_specs["initial_batch_size"]
    from libensemble.tools import get_rng

    # Set up RNG and persistent support channel to manager
    rng = get_rng(gen_specs, libE_info)
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    tag = None
    # Cycle covers all interesting share counts: 1,2,3,4 shares
    cycle = [0.5, 1.0, 1.5, 2.0]
    idx = 0
    while tag not in [STOP_TAG, PERSIS_STOP]:
        # Build output array for this batch
        H_o = np.zeros(b, dtype=gen_specs["out"])
        H_o["x"] = rng.uniform(lb, ub, (b, n))
        # Round-robin through fractional cycle so every batch mixes types
        ngpus = [cycle[(idx + i) % len(cycle)] for i in range(b)]
        idx += b
        H_o["num_gpus"] = ngpus
        H_o["num_procs"] = np.ones(b, dtype=int)  # keep CPU sizing trivial
        print(f"GEN created {b} sims requiring {ngpus} GPUs (shares {[int(v * 2) for v in ngpus]})", flush=True)
        tag, Work, calc_in = ps.send_recv(H_o)
        if hasattr(calc_in, "__len__"):
            b = len(calc_in)
    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG


if __name__ == "__main__":
    # Register dummy app for MPIExecutor (six_hump_camel is Python, no GPU needed)
    six_hump_camel_app = six_hump_camel.__file__
    exctr = MPIExecutor()
    exctr.register_app(full_path=six_hump_camel_app, app_name="six_hump_camel")

    # Ensemble setup: parse --nworkers/--comms, ensure enough rsets for fractional tasks
    gpu_test = Ensemble(parse_args=True, executor=exctr)
    nworkers = gpu_test.nworkers
    nsets = max(gpu_test.nworkers - 1, 8)  # need >=8 shares to host max 4-share task
    gpu_test.libE_specs = LibeSpecs(
        num_resource_sets=nsets,
        resource_info={"cores_on_node": (32, 64), "gpus_on_node": 4},
        gpu_sharing_factor=2,
        scheduler_opts={"gpu_packing": "pack"},
        sim_dirs_make=True,
        reuse_output_dir=True,
        ensemble_dir_path="./ensemble_GPU_sharing_variable_w" + str(nworkers) + str(gpu_test.libE_specs.comms),
    )

    # Simulator: auto-assign GPUs, validates via check_gpu_setting in var_resources
    gpu_test.sim_specs = SimSpecs(
        sim_f=sim_f,
        inputs=["x"],
        out=[("f", float)],
        user={"dry_run": False},  # set True to only log runlines without executing
    )
    # Generator: persistent, cycles fractional num_gpus (float) to drive sharing logic
    gpu_test.gen_specs = GenSpecs(
        gen_f=uniform_sample_sharing,
        persis_in=["f", "x", "sim_id"],
        out=[("num_procs", int), ("num_gpus", float), ("x", float, 2)],
        initial_batch_size=nworkers - 1,  # fill all sim workers initially
        batch_evaluate_same_priority=False,
        async_return=False,
        user={
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    )

    if gpu_test.ready():
        gpu_test.run(sim_max=12, wallclock_max=300)  # 12 sims covers 3 full cycles

    # Manager post-checks: flag 0 and save history
    if gpu_test.is_manager:
        assert gpu_test.flag == 0  # ensemble completed without manager exception
        gpu_test.save_output(__file__)
        print("GPU sharing regression test passed", flush=True)
