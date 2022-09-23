def run_forces_funcx(H, persis_info, sim_specs, libE_info):

    import os
    import time
    import secrets
    import numpy as np

    from libensemble.executors.mpi_executor import MPIExecutor
    from libensemble.message_numbers import WORKER_DONE, WORKER_KILL, TASK_FAILED

    class ForcesException(Exception):
        """Raised on some issue with Forces"""

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

    if sim_specs["user"]["fail_on_sim"]:
        raise ForcesException(Exception)

    calc_status = 0  # Returns to worker

    x = H["x"]
    sim_particles = sim_specs["user"]["sim_particles"]
    sim_timesteps = sim_specs["user"]["sim_timesteps"]
    time_limit = sim_specs["user"]["sim_kill_minutes"] * 60.0
    sim_app = sim_specs["user"]["sim_app"]

    exctr = MPIExecutor()
    exctr.register_app(full_path=sim_app, app_name="forces")

    calc_dir = os.path.join(sim_specs["user"]["remote_ensemble_dir"], secrets.token_hex(nbytes=4))
    os.makedirs(calc_dir, exist_ok=True)
    os.chdir(calc_dir)

    # Get from dictionary if key exists, else return default (e.g. 0)
    cores = sim_specs["user"].get("cores", None)
    kill_rate = sim_specs["user"].get("kill_rate", 0)
    particle_variance = sim_specs["user"].get("particle_variance", 0)

    # Composing variable names and x values to set up simulation
    seed = int(np.rint(x[0][0]))

    # This is to give a random variance of work-load
    sim_particles = perturb(sim_particles, seed, particle_variance)
    print(f"seed: {seed}   particles: {sim_particles}")

    args = str(int(sim_particles)) + " " + str(sim_timesteps) + " " + str(seed) + " " + str(kill_rate)

    machinefile = None
    if sim_specs["user"]["fail_on_submit"]:
        machinefile = "fail"

    # Machinefile only used here for exception testing
    if cores:
        task = exctr.submit(
            app_name="forces",
            num_procs=cores,
            app_args=args,
            stdout="out.txt",
            stderr="err.txt",
            wait_on_start=True,
            machinefile=machinefile,
        )
    else:
        task = exctr.submit(
            app_name="forces",
            app_args=args,
            stdout="out.txt",
            stderr="err.txt",
            wait_on_start=True,
            hyperthreads=True,
            machinefile=machinefile,
        )  # Auto-partition

    # Stat file to check for bad runs
    statfile = "forces.stat"
    filepath = os.path.join(task.workdir, statfile)
    line = None

    poll_interval = 1  # secs
    while not task.finished:
        # Read last line of statfile
        line = read_last_line(filepath)
        if line == "kill":
            task.kill()  # Bad run
        elif task.runtime > time_limit:
            task.kill()  # Timeout
        else:
            time.sleep(poll_interval)
            task.poll()

    if task.finished:
        if task.state == "FINISHED":
            print(f"Task {task.name} completed")
            calc_status = WORKER_DONE
            if read_last_line(filepath) == "kill":
                # Generally mark as complete if want results (completed after poll - before readline)
                print("Warning: Task completed although marked as a bad run (kill flag set in forces.stat)")
        elif task.state == "FAILED":
            print(f"Warning: Task {task.name} failed: Error code {task.errcode}")
            calc_status = TASK_FAILED
        elif task.state == "USER_KILLED":
            print(f"Warning: Task {task.name} has been killed")
            calc_status = WORKER_KILL
        else:
            print(f"Warning: Task {task.name} in unknown state {task.state}. Error code {task.errcode}")

    time.sleep(0.2)
    try:
        data = np.loadtxt(filepath)
        # task.read_file_in_workdir(statfile)
        final_energy = data[-1]
    except Exception:
        final_energy = np.nan
        # print('Warning - Energy Nan')

    outspecs = sim_specs["out"]
    output = np.zeros(1, dtype=outspecs)
    output["energy"][0] = final_energy

    return output, persis_info, calc_status
