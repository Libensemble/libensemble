import os
import time
import pytest
import platform
import numpy as np
from libensemble.libE import libE
import libensemble.tests.unit_tests.setup as setup
from libensemble.tools import add_unique_random_streams
from libensemble.manager import LoggedException
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first

def six_hump_camel_err(H, persis_info, sim_specs, _):
    raise Exception("Deliberate error")

@pytest.mark.extra
def test_profiling():
    
    from libensemble.sim_funcs.one_d_func import one_d_example as sim_f
    from libensemble.gen_funcs.sampling import latin_hypercube_sample as gen_f
    
    nworkers = 4
    libE_specs = {"comms": "local", "nworkers": nworkers}

    libE_specs["profile"] = True

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
    }

    gen_specs = {
        "gen_f": gen_f,
        "out": [("x", float, (1,))],
        "user": {
            "gen_batch_size": 200,
            "lb": np.array([-3]),
            "ub": np.array([3]),
        },
    }
    persis_info = add_unique_random_streams({}, nworkers + 1)
    exit_criteria = {"gen_max": 201}
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

    assert len(H) >= 201, "libEnsemble didn't generate enough points"
    assert "manager.prof" in os.listdir(), "Expected manager profile not found after run"
    os.remove("manager.prof")
    prof_files = [f"worker_{i+1}.prof" for i in range(nworkers)]
    # Ensure profile writes complete before checking
    time.sleep(0.5)
    for file in prof_files:
        assert file in os.listdir(), "Expected profile {file} not found after run"
        with open(file, "r") as f:
            data = f.read().split()
            num_worker_funcs_profiled = sum(["worker" in i for i in data])
        assert num_worker_funcs_profiled >= 8, (
            "Insufficient number of " + "worker functions profiled: " + str(num_worker_funcs_profiled)
        )
        os.remove(file)

@pytest.mark.extra
def test_calc_exception():
    from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f

    nworkers = 4
    libE_specs = {"comms": "local", "nworkers": nworkers}

    sim_specs = {
        "sim_f": six_hump_camel_err,
        "in": ["x"],
        "out": [("f", float)],
    }

    gen_specs = {
        "gen_f": gen_f,
        "in": ["sim_id"],
        "out": [("x", float, 2)],
        "user": {
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
            "gen_batch_size": 10,
        },
    }

    persis_info = add_unique_random_streams({}, nworkers + 1)
    exit_criteria = {"wallclock_max": 10}
    libE_specs["abort_on_exception"] = False

    return_flag = 1
    try:
        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)
    except LoggedException as e:
        print(f"Caught deliberate exception: {e}")
        return_flag = 0
    assert return_flag == 0


@pytest.mark.extra
def test_worker_exception():
    from libensemble.tests.regression_tests.support import nan_func as sim_f
    from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f

    nworkers = 4
    libE_specs = {"comms": "local", "nworkers": nworkers}
    n = 2

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
    }

    gen_specs = {
        "gen_f": gen_f,
        "in": [],
        "out": [("x", float, 2)],
        "user": {
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
            "initial_sample": 100,
        },
    }

    persis_info = add_unique_random_streams({}, nworkers + 1)

    libE_specs["abort_on_exception"] = False
    libE_specs["save_H_and_persis_on_abort"] = False
    exit_criteria = {"wallclock_max": 10}

    return_flag = 1
    try:
        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)
    except LoggedException as e:
        print(f"Caught deliberate exception: {e}")
        return_flag = 0
    assert return_flag == 0


@pytest.mark.extra
def test_elapsed_time_abort():
    from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f
    from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f
    from libensemble.tools import save_libE_output, add_unique_random_streams, eprint

    nworkers = 4
    libE_specs = {"comms": "local", "nworkers": nworkers}

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
        "user": {"pause_time": 2},
    }

    gen_specs = {
        "gen_f": gen_f,
        "in": ["sim_id"],
        "out": [("x", float, (2,))],
        "user": {
            "gen_batch_size": 5,
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }

    alloc_specs = {
        "alloc_f": give_sim_work_first,
        "user": {
            "batch_mode": False,
            "num_active_gens": 2,
        },
    }

    persis_info = add_unique_random_streams({}, nworkers + 1)
    exit_criteria = {"wallclock_max": 1}

    # Perform the run
    H, persis_info, flag = libE(
        sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs, alloc_specs=alloc_specs
    )

    eprint(flag)
    eprint(H)
    assert flag == 2
    save_libE_output(H, persis_info, __file__, nworkers)


if __name__ == "__main__":
    test_profiling()
    test_calc_exception()
    test_worker_exception()
    test_elapsed_time_abort()
