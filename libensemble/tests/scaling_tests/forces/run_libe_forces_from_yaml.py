#!/usr/bin/env python
import os
import numpy as np

from libensemble import Ensemble
from libensemble.executors.mpi_executor import MPIExecutor

####################

sim_app = os.path.join(os.getcwd(), 'forces.x')

if not os.path.isfile('forces.x'):
    if os.path.isfile('build_forces.sh'):
        import subprocess
        subprocess.check_call(['./build_forces.sh'])

####################

api = Ensemble()
api.from_yaml('forces.yaml')

api.logger.set_level('INFO')

if api.is_manager:
    print('\nRunning with {} workers\n'.format(api.nworkers))

exctr = MPIExecutor()
exctr.register_calc(full_path=sim_app, calc_type='sim')

api.persis_info.add_random_streams()
api.libE_specs['ensemble_dir_path'] = './ensemble'
api.gen_specs['user'].update({
    'lb': np.array([0]),
    'ub': np.array([32767])
})

api.run()

if api.is_manager:
    api.save_output(__file__)
