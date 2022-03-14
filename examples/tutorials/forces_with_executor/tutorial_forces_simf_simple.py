import os
import time
import numpy as np

from libensemble.executors.executor import Executor
from libensemble.message_numbers import WORKER_DONE, WORKER_KILL, TASK_FAILED

MAX_SEED = 32767

def run_forces(H, persis_info, sim_specs, libE_info):
    calc_status = 0

    sim_particles_in = H['x'][0][0]
    seed = int(np.rint(sim_particles_in))

    particles = str(int(sim_particles_in))
    args = particles + ' ' + str(10) + ' ' + particles

    exctr = Executor.executor
    task = exctr.submit(app_name='forces', app_args=args, wait_on_start=True)
    task.wait(timeout=60)

    # Stat file to check for bad runs
    statfile = 'forces{}.stat'.format(particles)

    try:
        data = np.loadtxt(statfile)
        final_energy = data[-1]
        calc_status = WORKER_DONE
    except Exception:
        final_energy = np.nan
        calc_status = TASK_FAILED

    outspecs = sim_specs['out']
    output = np.zeros(1, dtype=outspecs)
    output['energy'][0] = final_energy

    return output, persis_info, calc_status
