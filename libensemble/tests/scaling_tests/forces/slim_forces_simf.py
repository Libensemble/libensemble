import os
import time
import numpy as np

from libensemble.controller import JobController
from libensemble.message_numbers import WORKER_DONE, WORKER_KILL, JOB_FAILED

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

    jobctl = JobController.controller

    args = str(int(sim_particles)) + ' ' + str(sim_timesteps) + ' ' + str(seed) + ' ' + str(kill_rate)
    if cores:
        job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args, stdout='out.txt', stderr='err.txt', wait_on_run=True)
    else:
        job = jobctl.launch(calc_type='sim', app_args=args, stdout='out.txt', stderr='err.txt', wait_on_run=True)

    # Stat file to check for bad runs
    statfile = 'forces.stat'
    filepath = os.path.join(job.workdir, statfile)
    line = None

    poll_interval = 1
    while not job.finished :
        line = read_last_line(filepath)
        if line == "kill":
            job.kill()
        elif job.runtime > time_limit:
            job.kill()
        else:
            time.sleep(poll_interval)
            job.poll()

    if job.finished:
        if job.state == 'FINISHED':
            print("Job {} completed".format(job.name))
            calc_status = WORKER_DONE
            if read_last_line(filepath) == "kill":
                print("Warning: Job completed although marked as a bad run (kill flag set in forces.stat)")
        elif job.state == 'FAILED':
            print("Warning: Job {} failed: Error code {}".format(job.name, job.errcode))
            calc_status = JOB_FAILED
        elif job.state == 'USER_KILLED':
            print("Warning: Job {} has been killed".format(job.name))
            calc_status = WORKER_KILL
        else:
            print("Warning: Job {} in unknown state {}. Error code {}".format(job.name, job.state, job.errcode))

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
