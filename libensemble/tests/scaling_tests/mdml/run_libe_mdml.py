#!/usr/bin/env python
import os
import datetime

from libensemble.libE import libE
from libensemble.tools import parse_args, add_unique_random_streams
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble import libE_logger
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f

from deepdrivemd.sim.openmm import run_openmm
from deepdrivemd.aggregation.basic import aggregate
from deepdrivemd.models.keras_cvae import train
from deepdrivemd.selection.latest import select_model
from deepdrivemd.agents.lof import lof

from openmm_md_simf import run_openmm_sim_f
from agg_ml_genf import run_agg_ml_gen_f

libE_logger.set_level('INFO')  # INFO is now default

nworkers, is_manager, libE_specs, _ = parse_args()

sim_app = run_openmm.__file__
agg_gen_app = aggregate.__file__
ml_gen_app = train.__file__
sel_gen_app = select_model.__file__
agent_gen_app = lof.__file__

ddmd_apps = {'molecular_dynamics': sim_app,
             'aggregation': agg_gen_app,
             'machine_learning': ml_gen_app,
             'model_selection': sel_gen_app,
             'agent': agent_gen_app}

exctr = MPIExecutor()

for app in ddmd_apps:
    exctr.register_calc(full_path=ddmd_apps[app], app_name=app)

experiment_directory = os.path.abspath('./ensemble_' + str(datetime.datetime.today()).replace(' ', '_'))

gen_max = 8
initial_md_runs = 4
init_sample_parameter_name = 'temperature_kelvin'
init_sample_parameter_range = [280, 320]

sim_specs = {'sim_f': run_openmm_sim_f,
             'in': [init_sample_parameter_name, 'do_initial'],
             'out': [('file_path', "<U70", (1,)), ('sim_cstat', int, (1,))],
             'user': {'sim_kill_minutes': 15,
                      'sim_length_ns': 0.01,  # 1.0
                      'experiment_directory': experiment_directory,
                      'poll_interval': 1,
                      'dry_run': False,
                      'sample_parameter_name': init_sample_parameter_name,
                      'config_file': 'molecular_dynamics.yaml'}
             }

gen_specs = {'gen_f': run_agg_ml_gen_f,
             'in': [],
             'out': [(init_sample_parameter_name, float), ('sim_id', int),
                     ('do_initial', bool)],
             'user': {'initial_sample_size': initial_md_runs,
                      'parameter_range': init_sample_parameter_range,
                      'sample_parameter_name': init_sample_parameter_name,
                      'poll_interval': 1,
                      'skip_aggregation': True,
                      'aggregation_kill_minutes': 15,
                      'machine_learning_kill_minutes': 30,
                      'model_selection_kill_minutes': 15,
                      'agent_kill_minutes': 15,
                      }
             }

for app in ddmd_apps:
    gen_specs['user'][app + '_config'] = app + '.yaml'
    gen_specs['out'].append((app + '_cstat', int))

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
