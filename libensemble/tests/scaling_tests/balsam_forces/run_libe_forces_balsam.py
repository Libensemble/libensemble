#!/usr/bin/env python
import numpy as np

from libensemble import Ensemble
from libensemble.executors import NewBalsamMPIExecutor

from balsam.api import ApplicationDefinition

forces = Ensemble()
forces.from_yaml('balsam_forces.yaml')


class RemoteForces(ApplicationDefinition):
    site = 'three'
    command_template = \
        '/Users/jnavarro/Desktop/libensemble/' + \
        'libensemble/libensemble/tests/scaling_tests/balsam_forces/forces.x' + \
        ' {{sim_particles}} {{sim_timesteps}} {{seed}} {{kill_rate}}' + \
        ' > out.txt 2>&1'


exctr = NewBalsamMPIExecutor()
exctr.submit_allocation(site_id='three', num_nodes=1, wall_time_min=30,
                        queue='local', project='local')
exctr.register_app(RemoteForces, app_name='forces')

forces.gen_specs['user'].update({
    'lb': np.array([0]),
    'ub': np.array([32767])
})

forces.persis_info.add_random_streams()

forces.run()
