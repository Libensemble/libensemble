#!/usr/bin/env python
import os
import datetime

from deepdrivemd.sim.openmm import run_openmm
from deepdrivemd.aggregation.basic import aggregate

from openmm_md_simf import run_openmm_sim_f
from agg_ml_genf import run_agg_ml_gen_f

from libensemble.libE import libE
from libensemble.tools import parse_args, add_unique_random_streams
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble import libE_logger

from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f

libE_logger.set_level('INFO')  # INFO is now default

nworkers, is_manager, libE_specs, _ = parse_args()

sim_app = run_openmm.__file__
agg_gen_app = aggregate.__file__

exctr = MPIExecutor()
exctr.register_calc(full_path=sim_app, app_name='run_openmm')
exctr.register_calc(full_path=agg_gen_app, app_name='run_aggregate')
experiment_directory = os.path.abspath('./ensemble_' + str(datetime.datetime.today()).replace(' ', '_'))

sim_max = 4
init_sample_parameter_name = 'temperature_kelvin'
init_sample_parameter_range = [280, 320]

sim_specs = {'sim_f': run_openmm_sim_f,
             'in': [init_sample_parameter_name],
             'out': [('file_path', "<U70", (1,)), ('cstat', int, (1,))],
             'user': {'sim_kill_minutes': 15,
                      'experiment_directory': experiment_directory,
                      'poll_interval': 1,
                      'dry_run': False,
                      'sample_parameter_name': init_sample_parameter_name,
                      'config_file': 'md_config.yaml'}
             }

gen_specs = {'gen_f': run_agg_ml_gen_f,
             'in': [],
             'out': [('sample_parameter_value', float), ('sim_id', int), ('agg_cstat', int)],
             'user': {'agg_kill_minutes': 15,
                      'ml_kill_minutes': 30,
                      'experiment_directory': experiment_directory,
                      'poll_interval': 1,
                      'agg_config_file': 'aggregate_config.yaml',
                      'agg_dry_run': False,
                      'ml_dry_run': False,
                      'ml_config_file': 'ml_config.yaml',
                      'initial_sample_size': sim_max,
                      'parameter_range': init_sample_parameter_range}
             }

exit_criteria = {'sim_max': sim_max}

alloc_specs = {'alloc_f': alloc_f, 'out': [('given_back', bool)]}

libE_specs['sim_dirs_make'] = True
libE_specs['sim_input_dir'] = './sim'
libE_specs['sim_dir_symlink_files'] = [os.path.abspath('./1FME-folded.pdb'),
                                       os.path.abspath('./1FME-unfolded.pdb')]

libE_specs['gen_dirs_make'] = True
libE_specs['gen_input_dir'] = './gen'

libE_specs['ensemble_dir_path'] = experiment_directory

persis_info = add_unique_random_streams({}, nworkers + 1)

H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                            persis_info=persis_info,
                            alloc_specs=alloc_specs,
                            libE_specs=libE_specs)
