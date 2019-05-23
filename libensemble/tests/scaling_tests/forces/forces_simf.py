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


def make_unique_simdir(simdir, count=0):
    """As some dir names could recur, make sure unique"""
    if not os.path.isdir(simdir):
        return simdir
    else:
        count += 1
        return make_unique_simdir(".".join([simdir.split('.')[0], str(count)]), count)


def run_forces(x, gen_specs, sim_specs, libE_info):
    # Setting up variables needed for input and output
    # keys              = variable names
    # x                 = variable values
    # output            = what will be returned to libE

    calc_status = 0  # Returns to worker

    simdir_basename = sim_specs['simdir_basename']
    # cores           = sim_specs['cores']
    keys = sim_specs['keys']
    sim_particles = sim_specs['sim_particles']
    sim_timesteps = sim_specs['sim_timesteps']
    time_limit = sim_specs['sim_kill_minutes'] * 60.0

    # Get from dictionary if key exists, else return default (e.g. 0)
    cores = sim_specs.get('cores', None)
    kill_rate = sim_specs.get('kill_rate', 0)
    particle_variance = sim_specs.get('particle_variance', 0)

    # Composing variable names and x values to set up simulation
    # arguments = []
    # sim_dir   = [simdir_basename]
    # for i,key in enumerate(keys):
    #    variable = key+'='+str(x[i])
    #    arguments.append(variable)
    #    sim_dir.append('_'+variable)
    # print(os.getcwd(), sim_dir)

    # For one key
    seed = int(np.rint(x[0][0]))

    # This is to give a random variance of work-load
    sim_particles = perturb(sim_particles, seed, particle_variance)
    print('seed: {}   particles: {}'.format(seed, sim_particles))

    # At this point you will be in the sim directory (really worker dir) for this worker (eg. sim_1).
    # The simdir below is created for each job for this worker.
    # Any input needs to be copied into this directory. Currently there is none.
    simdir = simdir_basename + '_' + keys[0] + '_' + str(seed)
    simdir = make_unique_simdir(simdir)
    os.mkdir(simdir)
    os.chdir(simdir)
    jobctl = JobController.controller  # Get JobController

    args = str(int(sim_particles)) + ' ' + str(sim_timesteps) + ' ' + str(seed) + ' ' + str(kill_rate)
    # job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args, stdout='out.txt', stderr='err.txt')

    if cores:
        job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args, stdout='out.txt', stderr='err.txt', wait_on_run=True)
    else:
        job = jobctl.launch(calc_type='sim', app_args=args, stdout='out.txt', stderr='err.txt', wait_on_run=True)  # Auto-partition

    # Stat file to check for bad runs
    statfile = simdir_basename+'.stat'
    filepath = os.path.join(job.workdir, statfile)
    line = None

    poll_interval = 1  # secs
    while(not job.finished):
        # Read last line of statfile
        line = read_last_line(filepath)
        if line == "kill":
            job.kill()  # Bad run
        elif job.runtime > time_limit:
            job.kill()  # Timeout
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

    os.chdir('../')

    time.sleep(0.2)
    try:
        data = np.loadtxt(filepath)
        # job.read_file_in_workdir(statfile)
        final_energy = data[-1]
    except Exception as e:
        print('Caught:', e)
        final_energy = np.nan
        # print('Warning - Energy Nan')

    outspecs = sim_specs['out']
    output = np.zeros(1, dtype=outspecs)
    output['energy'][0] = final_energy

    return output, gen_specs, calc_status
