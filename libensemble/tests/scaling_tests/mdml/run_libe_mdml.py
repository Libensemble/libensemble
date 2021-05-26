#!/usr/bin/env python
import os
import datetime
import numpy as np
from deepdrivemd.sim.openmm import run_openmm
from openmm_md_simf import run_openmm_sim_f

# Import libEnsemble modules
from libensemble.libE import libE
from libensemble.tools import parse_args, add_unique_random_streams
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble import libE_logger

from libensemble.alloc_funcs.give_pregenerated_work import give_pregenerated_sim_work as alloc_f

libE_logger.set_level('INFO')  # INFO is now default

nworkers, is_manager, libE_specs, _ = parse_args()

sim_app = run_openmm.__file__

exctr = MPIExecutor()
exctr.register_calc(full_path=sim_app, app_name='run_openmm')

sim_specs = {'sim_f': run_openmm_sim_f,
             'in': ['tk'],
             'out': [('file_path', "<U70", (1,)), ('cstat', int, (1,))],
             'user': {'sim_kill_minutes': 15,
                      'poll_interval': 1,
                      'dry_run': True,
                      'config_file': 'config.yaml'}
             }

gen_specs = {}

sim_max = 4
exit_criteria = {'sim_max': sim_max}

alloc_specs = {'alloc_f': alloc_f, 'out': [('tk', float, sim_max)]}

libE_specs['sim_dirs_make'] = True
libE_specs['sim_input_dir'] = './sim'
libE_specs['sim_dir_symlink_files'] = [os.path.abspath('1FME-unfolded.pdb')]
libE_specs['ensemble_dir_path'] = './ensemble_' + str(datetime.datetime.today()).replace(' ', '_')

H0 = np.zeros(sim_max, dtype=[('tk', float, 1), ('sim_id', int), ('given', bool)])
H0['tk'] = [310.0, 309.0, 308.0, 307.0]
H0['sim_id'] = range(sim_max)
H0['given'] = False

persis_info = add_unique_random_streams({}, nworkers + 1)

H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                            persis_info=persis_info,
                            alloc_specs=alloc_specs,
                            libE_specs=libE_specs, H0=H0)
