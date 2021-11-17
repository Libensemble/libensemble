#!/usr/bin/env python
import os
import secrets
import numpy as np

from libensemble import Ensemble
from libensemble.executors.mpi_executor import MPIExecutor

forces = Ensemble()
forces.from_yaml('funcx_forces.yaml')

sim_app = '/home/jnavarro/libensemble/libensemble/tests/scaling_tests/forces/forces.x'

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
