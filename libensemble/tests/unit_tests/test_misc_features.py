import os
import shutil
import tempfile
import time

import numpy as np
import pytest

from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from libensemble.libE import libE
from libensemble.manager import LoggedException
from libensemble.tools import add_unique_random_streams
from libensemble.utils.loc_stack import LocationStack
from libensemble.utils.timer import TaskTimer, Timer


def six_hump_camel_err(H, persis_info, sim_specs, _):
    raise Exception("Deliberate error")


@pytest.mark.extra
def test_profiling():

    from libensemble.gen_funcs.sampling import latin_hypercube_sample as gen_f
    from libensemble.sim_funcs.one_d_func import one_d_example as sim_f

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
    from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f
    from libensemble.tests.regression_tests.support import nan_func as sim_f

    nworkers = 4
    libE_specs = {"comms": "local", "nworkers": nworkers}

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
    from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f
    from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f
    from libensemble.tools import add_unique_random_streams, eprint, save_libE_output

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


def test_location_stack():
    "Test correctness of location stack (all in a temp dir)."

    tmp_dirname = tempfile.mkdtemp()
    assert os.path.isdir(tmp_dirname), f"Failed to create temporary directory {tmp_dirname}."

    try:
        # Record where we started
        start_dir = os.getcwd()

        # Set up directory for clone
        clone_dirname = os.path.join(tmp_dirname, "basedir")
        os.mkdir(clone_dirname)
        test_fname = os.path.join(clone_dirname, "test.txt")
        with open(test_fname, "w+") as f:
            f.write("This is a test file\n")

        s = LocationStack()

        # Register a valid location
        tname = s.register_loc(0, "testdir", prefix=tmp_dirname, copy_files=[test_fname])
        assert os.path.isdir(tname), f"New directory {tname} was not created."
        assert os.path.isfile(
            os.path.join(tname, "test.txt")
        ), f"New directory {tname} failed to copy test.txt from {clone_dirname}."

        # Register an empty location
        d = s.register_loc(1, None)
        assert d is None, "Dir stack not correctly register None at location 1."

        # Register a dummy location (del should not work)
        d = s.register_loc(2, os.path.join(tmp_dirname, "dummy"))
        assert ~os.path.isdir(d), "Directory stack registration of dummy should not create dir."

        # Push unregistered location (we should not move)
        s.push_loc(3)
        assert s.stack == [None], "Directory stack push_loc(missing) failed to put None on stack."
        assert os.path.samefile(
            os.getcwd(), start_dir
        ), "Directory stack push_loc failed to stay put with input None." "Wanted {}, at {}".format(
            start_dir, os.getcwd()
        )

        # Push registered location (we should move
        s.push_loc(0)
        assert s.stack == [None, start_dir], "Directory stack is incorrect." "Wanted [None, {}], got {}.".format(
            start_dir, s.stack
        )
        assert os.path.samefile(
            os.getcwd(), tname
        ), f"Directory stack push_loc failed to end up at desired dir.Wanted {tname}, at {os.getcwd()}"

        # Pop the registered location
        s.pop()
        assert s.stack == [None], f"Directory stack is incorrect after pop.Wanted [None], got {s.stack}."
        assert os.path.samefile(
            os.getcwd(), start_dir
        ), "Directory stack push_loc failed to stay put with input None." "Wanted {}, at {}".format(
            start_dir, os.getcwd()
        )

        # Context for moving again
        with s.loc(0):
            assert s.stack == [None, start_dir], "Directory stack is incorrect." "Wanted [None, {}], got {}.".format(
                start_dir, s.stack
            )
            assert os.path.samefile(
                os.getcwd(), tname
            ), f"Directory stack push_loc failed to end up at desired dir.Wanted {tname}, at {os.getcwd()}"

        # Check directory after context
        assert s.stack == [None], f"Directory stack is incorrect after ctx.Wanted [None], got {s.stack}."
        assert os.path.samefile(os.getcwd(), start_dir), "Directory looks wrong after ctx." "Wanted {}, at {}".format(
            start_dir, os.getcwd()
        )

        with s.dir(None):
            assert s.stack == [None, None], "Directory stack is incorrect in ctx."
        assert s.stack == [None], "Directory stack is incorrect after ctx."

        # Pop the unregistered location
        s.pop()
        assert not s.stack, f"Directory stack should be empty, actually {s.stack}."
        assert os.path.samefile(
            os.getcwd(), start_dir
        ), "Directory stack push_loc failed to stay put with input None." "Wanted {}, at {}".format(
            start_dir, os.getcwd()
        )

        # Clean up
        s.clean_locs()
        assert not os.path.isdir(tname), f"Directory {tname} should have been removed on cleanup."

    finally:
        shutil.rmtree(tmp_dirname)


@pytest.mark.extra
def test_cdist_issue():
    try:
        from scipy.spatial.distance import cdist
    except ModuleNotFoundError:
        pytest.skip("scipy or its dependencies not importable. Skipping.")

    """There is an issue (at least in scipy 1.1.0) with cdist segfaulting."""
    H = np.zeros(
        20,
        dtype=[
            ("x", "<f8", (2,)),
            ("m", "<i8"),
            ("a", "<f8"),
            ("b", "?"),
            ("c", "?"),
            ("d", "<f8"),
            ("e", "<f8"),
            ("fa", "<f8"),
            ("g", "<i8"),
            ("h", "<i8"),
            ("i", "?"),
            ("j", "<i8"),
            ("k", "?"),
            ("f", "<f8"),
            ("l", "?"),
        ],
    )
    np.random.seed(1)
    H["x"] = np.random.uniform(0, 1, (20, 2))
    dist_1 = cdist(np.atleast_2d(H["x"][3]), H["x"], "euclidean")
    assert len(dist_1), "We didn't segfault"


@pytest.mark.extra
def test_save():
    """Seeing if I can save parts of the H array."""
    from libensemble.tests.regression_tests.support import uniform_or_localopt_gen_out as gen_out

    n = 2
    gen_out += [("x", float, n), ("x_on_cube", float, n)]
    H = np.zeros(20, dtype=gen_out + [("f", float), ("grad", float, n)])
    np.random.seed(1)
    H["x"] = np.random.uniform(0, 1, (20, 2))
    np.save("H_test", H[["x", "f", "grad"]])

    assert 1, "We saved correctly"


def test_timer():
    "Test timer."

    time_start = time.time()

    timer = Timer()

    with timer:
        time.sleep(0.5)
        e1 = timer.elapsed

    e2 = timer.elapsed
    time.sleep(0.1)
    e3 = timer.elapsed
    time_mid = time.time() - time_start

    # Use external wall-clock time for upper limit to allow for system overhead
    # (e.g. virtualization, or sharing machine with other tasks)
    # assert (e1 >= 0.5) and (e1 <= 0.6), "Check timed sleep seems correct"
    assert (e1 >= 0.5) and (e1 < time_mid), "Check timed sleep within boundaries"
    assert e2 >= e1, "Check timer order."
    assert e2 == e3, "Check elapsed time stable when timer inactive."

    s1 = timer.date_start
    s2 = timer.date_end
    assert s1[0:2] == "20", "Start year is 20xx"
    assert s2[0:2] == "20", "End year is 20xx"

    s3 = f"{timer}"
    assert s3 == f"Time: {e3:.3f} Start: {s1} End: {s2}", "Check string formatting."

    time.sleep(0.2)
    time_start = time.time()
    with timer:
        time.sleep(0.5)
        total1 = timer.total

    time_end = time.time() - time_start + time_mid

    assert total1 >= 1 and total1 <= time_end, "Check cumulative timing (active)."
    assert timer.total >= 1 and timer.total <= time_end, "Check cumulative timing (not active)."


def test_TaskTimer():
    print(TaskTimer())


if __name__ == "__main__":
    test_profiling()
    test_calc_exception()
    test_worker_exception()
    test_elapsed_time_abort()
    test_location_stack()
    test_cdist_issue()
    test_save()
    test_timer()
    test_TaskTimer()
