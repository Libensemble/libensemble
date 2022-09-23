import os
import time
import numpy as np

from libensemble.executors.executor import Executor
from libensemble.message_numbers import WORKER_DONE, WORKER_KILL, TASK_FAILED

MAX_SEED = 32767


class ForcesException(Exception):
    """Custom forces exception"""


def perturb(particles, seed, max_fraction):
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


def make_unique_simdir(simdir, count=0):
    """As some dir names could recur, make sure unique"""
    if not os.path.isdir(simdir):
        return simdir
    else:
        count += 1
        return make_unique_simdir(".".join([simdir.split(".")[0], str(count)]), count)


def run_forces(H, persis_info, sim_specs, libE_info):
    # Setting up variables needed for input and output
    # keys              = variable names
    # x                 = variable values
    # output            = what will be returned to libE
    if sim_specs["user"]["fail_on_sim"]:
        raise ForcesException

    calc_status = 0  # Returns to worker

    x = H["x"]
    # keys = sim_specs['user']['keys']
    sim_particles = sim_specs["user"]["sim_particles"]
    sim_timesteps = sim_specs["user"]["sim_timesteps"]
    time_limit = sim_specs["user"]["sim_kill_minutes"] * 60.0

    # Get from dictionary if key exists, else return default (e.g. 0)
    cores = sim_specs["user"].get("cores", None)
    kill_rate = sim_specs["user"].get("kill_rate", 0)
    particle_variance = sim_specs["user"].get("particle_variance", 0)

    # Composing variable names and x values to set up simulation
    seed = int(np.rint(x[0][0]))

    # This is to give a random variance of work-load
    sim_particles = perturb(sim_particles, seed, particle_variance)
    print(f"seed: {seed}   particles: {sim_particles}")

    exctr = Executor.executor  # Get Executor

    args = str(int(sim_particles)) + " " + str(sim_timesteps) + " " + str(seed) + " " + str(kill_rate)
    # task = exctr.submit( app_name='forces', num_procs=cores, app_args=args, stdout='out.txt', stderr='err.txt')

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
            machinefile=machinefile,
        )
    else:
        task = exctr.submit(
            app_name="forces",
            app_args=args,
            stdout="out.txt",
            stderr="err.txt",
            hyperthreads=True,
            machinefile=machinefile,
        )  # Auto-partition

    # Stat file to check for bad runs
    statfile = "forces.stat"
    filepath = os.path.join(task.workdir, statfile)
    line = None

    poll_interval = 0.1  # secs
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
