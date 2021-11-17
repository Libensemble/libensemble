#!/usr/bin/env python
import os
import secrets
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

forces = Ensemble()
forces.from_yaml('funcx_forces.yaml')

exctr = MPIExecutor()
exctr.register_app(full_path=sim_app, app_name='forces')

forces.libE_specs['ensemble_dir_path'] = './ensemble_' + secrets.token_hex(nbytes=3)
forces.sim_specs['user'].update({
    'exctr': exctr
})
forces.gen_specs['user'].update({
    'lb': np.array([0]),
    'ub': np.array([32767])
})

forces.persis_info.add_random_streams()

forces.run()
