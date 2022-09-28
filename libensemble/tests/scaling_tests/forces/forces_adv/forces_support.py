import os

outfiles = ["err.txt", "forces.stat", "out.txt"]


def check_log_exception():
    with open("ensemble.log", "r") as el:
        out = el.readlines()
    assert "forces_simf.ForcesException\n" in out, "ForcesException not received by manager or logged."


def test_libe_stats(status):
    with open("libE_stats.txt", "r") as ls:
        out = ls.readlines()
    assert all(
        [line.endswith(status) for line in out if "sim" in line]
    ), "Deliberate error status not logged or raised for all sim instances."


def test_ensemble_dir(libE_specs, dir, nworkers, sim_max):
    if not os.path.isdir(dir):
        print(f"Specified ensemble directory {dir} not found.")
        return

    if not libE_specs.get("sim_dirs_make"):
        print("Typical tests of ensemble directories dont apply without sim_dirs.")
        return

    if libE_specs.get("use_worker_dirs"):
        assert len(os.listdir(dir)) == nworkers, "Number of worker directories ({}) doesn't match nworkers ({})".format(
            len(os.listdir(dir)), nworkers
        )

        num_sim_dirs = 0
        files_found = []

        worker_dirs = [i for i in os.listdir(dir) if i.startswith("worker")]
        for worker_dir in worker_dirs:
            sim_dirs = [i for i in os.listdir(os.path.join(dir, worker_dir)) if i.startswith("sim")]
            num_sim_dirs += len(sim_dirs)

            for sim_dir in sim_dirs:
                files_found.append(all([i in os.listdir(os.path.join(dir, worker_dir, sim_dir)) for i in outfiles]))

        assert (
            num_sim_dirs == sim_max
        ), f"Number of simulation specific-directories ({num_sim_dirs}) doesn't match sim_max ({sim_max})"

        assert all(
            files_found
        ), "Set of expected files ['err.txt', 'forces.stat', 'out.txt'] not found in each sim_dir."

    else:
        sim_dirs = os.listdir(dir)
        assert all(
            [i.startswith("sim") for i in sim_dirs]
        ), "All directories within ensemble dir not labeled as (or aren't) sim_dirs."

        assert (
            len(sim_dirs) == sim_max
        ), f"Number of simulation specific-directories ({len(sim_dirs)}) doesn't match sim_max ({sim_max})"

        files_found = []
        for sim_dir in sim_dirs:
            files_found.append(all([i in os.listdir(os.path.join(dir, sim_dir)) for i in outfiles]))

        assert all(
            files_found
        ), "Set of expected files ['err.txt', 'forces.stat', 'out.txt'] not found in each sim_dir."

    print(f"Output directory {dir} passed tests.")
