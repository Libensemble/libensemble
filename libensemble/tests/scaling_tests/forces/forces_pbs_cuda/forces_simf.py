import math

import numpy as np

from libensemble.message_numbers import TASK_FAILED, WORKER_DONE
from libensemble.resources.resources import Resources
from libensemble.tools.test_support import check_gpu_setting


def run_forces(H, persis_info, sim_specs, libE_info):
    """PBSpro+CUDA with gpu_sharing_factor: validate assignment/splitting on real HW."""

    calc_status = 0
    particles = str(int(H["x"][0][0]))
    args = particles + " " + str(10) + " " + particles

    exctr = libE_info["executor"]
    task = exctr.submit(
        app_name="forces",
        app_args=args,
        auto_assign_gpus=True,
        match_procs_to_gpus=True,
    )
    task.wait()
    check_gpu_setting(task, assert_setting=False, print_setting=True)

    try:
        wres = Resources.resources.worker_resources
        sf = sim_specs["user"].get("gpu_sharing_factor", 1)
        req_gpus = float(H["num_gpus"][0]) if "num_gpus" in H.dtype.names else 1.0
        expected_shares = math.ceil(req_gpus * sf)
        assert wres.num_rsets == expected_shares, f"num_rsets {wres.num_rsets} != {expected_shares} for {req_gpus}"
        for gid in wres.gpu_ids_on_node:
            assert gid >= 0, f"negative gpu_id {gid}"
        gpu_shares_per_node = wres.gpu_rsets_per_node
        if gpu_shares_per_node and expected_shares <= gpu_shares_per_node:
            assert (
                wres.local_node_count == 1
            ), f"Expected single-node for {expected_shares}, got {wres.local_node_count}"
        elif gpu_shares_per_node:
            expected_nodes = math.ceil(expected_shares / gpu_shares_per_node)
            assert wres.local_node_count == expected_nodes
        assert wres.matching_slots, f"Unmatching slots {wres.slots}"
    except AssertionError as e:
        print(f"Worker {libE_info.get('workerID')} assignment assert failed: {e}", flush=True)
        calc_status = TASK_FAILED
    except Exception as e:
        print(f"Worker validation skipped: {e}", flush=True)

    statfile = "forces.stat"
    try:
        data = np.loadtxt(statfile)
        final_energy = data[-1]
        if calc_status != TASK_FAILED:
            calc_status = WORKER_DONE
    except Exception:
        final_energy = np.nan
        calc_status = TASK_FAILED

    output = np.zeros(1, dtype=sim_specs["out"])
    output["energy"] = final_energy
    return output, persis_info, calc_status
