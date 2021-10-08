import os
import time
import numpy as np

from libensemble.executors.executor import Executor
from libensemble.message_numbers import WORKER_DONE, WORKER_KILL, TASK_FAILED

MAX_SEED = 32767


def perturb(particles, seed, max_fraction):
    """Modify particle count"""
    seed_fraction = seed/MAX_SEED
    max_delta = particles * max_fraction
    delta = seed_fraction * max_delta
    delta = delta - max_delta/2  # translate so -/+
    new_particles = particles + delta
    return int(new_particles)


def read_last_line(filepath):
    """Read last line of statfile"""
    try:
        with open(filepath, 'rb') as fh:
            line = fh.readlines()[-1].decode().rstrip()
    except Exception:
        line = ""  # In case file is empty or not yet created
    return line


def run_forces(H, persis_info, sim_specs, libE_info):
    calc_status = 0

    x = H['x']
    sim_particles = sim_specs['user']['sim_particles']
    sim_timesteps = sim_specs['user']['sim_timesteps']
    time_limit = sim_specs['user']['sim_kill_minutes'] * 60.0

    # Get from dictionary if key exists, else return default (e.g. 0)
    cores = sim_specs['user'].get('cores', None)
    kill_rate = sim_specs['user'].get('kill_rate', 0)
    particle_variance = sim_specs['user'].get('particle_variance', 0)

    # Composing variable names and x values to set up simulation
    seed = int(np.rint(x[0][0]))

    # This is to give a random variance of work-load
    sim_particles = perturb(sim_particles, seed, particle_variance)

    exctr = Executor.executor

    args = str(int(sim_particles)) + ' ' + str(sim_timesteps) + ' ' + str(seed) + ' ' + str(kill_rate)
    if cores:
        task = exctr.submit(app_name='forces', app_args=args, num_procs=cores,
                            stdout='out.txt', stderr='err.txt', wait_on_start=True)
    else:
        task = exctr.submit(app_name='forces', app_args=args,
                            stdout='out.txt', stderr='err.txt', wait_on_start=True)

    # Stat file to check for bad runs
    statfile = 'forces.stat'
    filepath = os.path.join(task.workdir, statfile)
    line = None

    poll_interval = 1
    while not task.finished:
        line = read_last_line(filepath)
        if line == "kill":
            task.kill()
        elif task.runtime > time_limit:
            task.kill()
        else:
            time.sleep(poll_interval)
            task.poll()

    if task.finished:
        if task.state == 'FINISHED':
            print("Task {} completed".format(task.name))
            calc_status = WORKER_DONE
            if read_last_line(filepath) == "kill":
                print("Warning: Task complete but marked bad (kill flag in forces.stat)")
        elif task.state == 'FAILED':
            print("Warning: Task {} failed: Error code {}".format(task.name, task.errcode))
            calc_status = TASK_FAILED
        elif task.state == 'USER_KILLED':
            print("Warning: Task {} has been killed".format(task.name))
            calc_status = WORKER_KILL
        else:
            print("Warning: Task {} in unknown state {}. Error code {}".format(task.name, task.state, task.errcode))

    time.sleep(0.2)
    try:
        data = np.loadtxt(filepath)
        final_energy = data[-1]
    except Exception:
        final_energy = np.nan

    outspecs = sim_specs['out']
    output = np.zeros(1, dtype=outspecs)
    output['energy'][0] = final_energy

    return output, persis_info, calc_status
