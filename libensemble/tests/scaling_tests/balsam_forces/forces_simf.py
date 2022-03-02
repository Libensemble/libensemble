def run_forces_balsam(H, persis_info, sim_specs, libE_info):

    import os
    import time
    import secrets
    import numpy as np

    from libensemble.executors.executor import Executor
    from libensemble.message_numbers import WORKER_DONE, TASK_FAILED

    def perturb(particles, seed, max_fraction):
        MAX_SEED = 32767
        """Modify particle count"""
        seed_fraction = seed / MAX_SEED
        max_delta = particles * max_fraction
        delta = seed_fraction * max_delta
        delta = delta - max_delta / 2  # translate so -/+
        new_particles = particles + delta
        return int(new_particles)

    def read_last_line(filepath):
        """Read last line of statfile"""
        try:
            with open(filepath, "rb") as fh:
                line = fh.readlines()[-1].decode().rstrip()
        except Exception:
            line = ""  # In case file is empty or not yet created
        return line

    calc_status = 0  # Returns to worker

    exctr = Executor.executor

    x = H["x"]
    sim_particles = sim_specs["user"]["sim_particles"]
    sim_timesteps = sim_specs["user"]["sim_timesteps"]
    TRANSFER_STATFILES = sim_specs["user"]["transfer"]
    globus_endpoint = sim_specs["user"]["globus_endpoint"]

    # Get from dictionary if key exists, else return default (e.g. 0)
    kill_rate = sim_specs["user"].get("kill_rate", 0)
    particle_variance = sim_specs["user"].get("particle_variance", 0)

    # Composing variable names and x values to set up simulation
    seed = int(np.rint(x[0][0]))

    # This is to give a random variance of work-load
    sim_particles = perturb(sim_particles, seed, particle_variance)
    print("seed: {}   particles: {}".format(seed, sim_particles))

    args = {
        "sim_particles": sim_particles,
        "sim_timesteps": sim_timesteps,
        "seed": seed,
        "kill_rate": kill_rate,
    }
    workdir = "worker" + str(libE_info["workerID"]) + "_" + secrets.token_hex(nbytes=3)

    file_dest = os.getcwd() + "/forces_" + secrets.token_hex(nbytes=3) + ".stat"
    if TRANSFER_STATFILES:
        transfer = {"result": globus_endpoint + ":" + file_dest}
    else:
        transfer = {}

    task = exctr.submit(
        app_name="forces",
        app_args=args,
        num_procs=4,
        num_nodes=1,
        procs_per_node=4,
        max_tasks_per_node=1,
        transfers=transfer,
        workdir=workdir,
    )

    poll_interval = 2  # secs
    print("Beginning to poll Task {}".format(task.name))
    while not task.finished:
        time.sleep(poll_interval)
        task.poll()
        if task.state == "FAILED":
            break

    if task.state in ["FINISHED", "FAILED"]:
        print("Task {} exited with state {}.".format(task.name, task.state))
        if TRANSFER_STATFILES:
            print("Waiting for Task {} statfile.".format(task.name))
            while file_dest not in [os.path.join(os.getcwd(), i) for i in os.listdir(".")]:
                time.sleep(1)
            if read_last_line(file_dest) == "kill":
                print("Warning: Task completed although marked as a bad run (kill flag set in retrieved forces.stat)")
                calc_status = TASK_FAILED
            else:
                calc_status = WORKER_DONE
                print("Task completed successfully. forces.stat retrieved.")
        else:
            calc_status = WORKER_DONE
            print("Task completed.")
    else:
        print(task.state)

    time.sleep(0.2)
    try:
        data = np.loadtxt(file_dest)
        final_energy = data[-1]
    except Exception:
        final_energy = np.nan

    outspecs = sim_specs["out"]
    output = np.zeros(1, dtype=outspecs)
    output["energy"][0] = final_energy

    return output, persis_info, calc_status
