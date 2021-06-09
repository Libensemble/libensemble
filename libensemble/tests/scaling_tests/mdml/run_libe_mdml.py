#!/usr/bin/env python
import os
import datetime

from deepdrivemd.sim.openmm import run_openmm
from deepdrivemd.aggregation.basic import aggregate
from deepdrivemd.models.keras_cvae import train
from deepdrivemd.selection.latest import select_model
from deepdrivemd.agents.lof import lof

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
ml_gen_app = train.__file__
sel_gen_app = select_model.__file__
agent_gen_app = lof.__file__

exctr = MPIExecutor()
exctr.register_calc(full_path=sim_app, app_name='run_openmm')
exctr.register_calc(full_path=agg_gen_app, app_name='run_aggregate')
exctr.register_calc(full_path=ml_gen_app, app_name='run_ml_train')
exctr.register_calc(full_path=sel_gen_app, app_name='run_model_select')
exctr.register_calc(full_path=agent_gen_app, app_name='run_outlier_agent')

experiment_directory = os.path.abspath('./ensemble_' + str(datetime.datetime.today()).replace(' ', '_'))

gen_max = 8
initial_md_runs = 4
ml_num_tasks = 1
init_sample_parameter_name = 'temperature_kelvin'
init_sample_parameter_range = [280, 320]

sim_specs = {'sim_f': run_openmm_sim_f,
             'in': [init_sample_parameter_name],
             'out': [('file_path', "<U70", (1,)), ('sim_cstat', int, (1,))],
             'user': {'sim_kill_minutes': 15,
                      'sim_length_ns': 0.01,  # 1.0
                      'experiment_directory': experiment_directory,
                      'poll_interval': 1,
                      'dry_run': False,
                      'sample_parameter_name': init_sample_parameter_name,
                      'config_file': 'md_config.yaml'}
             }

gen_specs = {'gen_f': run_agg_ml_gen_f,
             'in': [],
             'out': [(init_sample_parameter_name, float), ('sim_id', int), ('agg_cstat', int),
                     ('ml_cstat', int, ml_num_tasks)],

             'user': {'initial_sample_size': initial_md_runs,
                      'parameter_range': init_sample_parameter_range,
                      'sample_parameter_name': init_sample_parameter_name,
                      'experiment_directory': experiment_directory,
                      'poll_interval': 1,
                      'agg_kill_minutes': 15,
                      'agg_config_file': 'aggregate_config.yaml',
                      'agg_dry_run': False,
                      'ml_kill_minutes': 30,
                      'ml_config_file': 'ml_config.yaml',
                      'ml_num_tasks': ml_num_tasks,
                      'ml_dry_run': False,
                      'sel_kill_minutes': 15,
                      'sel_config_file': 'selection_config.yaml',
                      'sel_dry_run': False,
                      'agent_kill_minutes': 15,
                      'agent_config_file': 'agent_config.yaml',
                      'agent_dry_run': False}
             }

alloc_specs = {'alloc_f': alloc_f, 'out': [('given_back', bool)],
               'user': {'init_sample_size': initial_md_runs}}

exit_criteria = {'gen_max': gen_max}

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
