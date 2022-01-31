#!/usr/bin/env python
import secrets
import numpy as np

from libensemble import Ensemble
from libensemble.executors import NewBalsamMPIExecutor

from balsam.api import ApplicationDefinition

forces = Ensemble()
forces.from_yaml('balsam_forces.yaml')

class RemoteForces(ApplicationDefinition):
    site = 'libe-bebop'
    command_template = './forces.x {{sim_particles}} {{sim_timesteps}} {{seed}} {{kill_rate}}'

import ipdb; ipdb.set_trace()

exctr = NewBalsamMPIExecutor()
exctr.register_app(RemoteForces, app_name='forces')

forces.sim_specs['user']['remote_ensemble_dir'] += secrets.token_hex(nbytes=3)

forces.gen_specs['user'].update({
    'lb': np.array([0]),
    'ub': np.array([32767])
})

forces.persis_info.add_random_streams()

forces.run()
