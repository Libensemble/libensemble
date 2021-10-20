#!/usr/bin/env python
import os
import requests
import datetime

from libensemble.libE import libE
from libensemble.tools import parse_args, add_unique_random_streams
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble import logger

# Import our simulator and generator functions
from openmm_md_simf import run_openmm_sim_f
from keras_cvae_ml_genf import run_keras_cvae_ml_genf

# Import DeepDriveMD components for registering with libEnsemble's executor
from deepdrivemd.sim.openmm import run_openmm
from deepdrivemd.aggregation.basic import aggregate
from deepdrivemd.models.keras_cvae import train
from deepdrivemd.selection.latest import select_model
from deepdrivemd.agents.lof import lof

logger.set_level('INFO')


def download_data(url, file):
    if file not in os.listdir('.'):
        print('Downloading ' + file + ' ...')
        out = requests.get(url)
        with open(file, 'wb') as f:
            f.write(out.content)


folded_url = 'https://raw.githubusercontent.com/DeepDriveMD/DeepDriveMD-pipeline/main/data/bba/1FME-folded.pdb'
unfolded_url = folded_url.replace('1FME-folded.pdb', 'system/1FME-unfolded.pdb')

for url in [folded_url, unfolded_url]:
    download_data(url, url.split('/')[-1])

# Parse comms type, number of workers, etc. from command-line
nworkers, is_manager, libE_specs, _ = parse_args()

ddmd_apps = {'molecular_dynamics': run_openmm.__file__,
             'aggregation': aggregate.__file__,
             'machine_learning': train.__file__,
             'model_selection': select_model.__file__,
             'agent': lof.__file__}

# Submit each component via a MPI runner. This Executor can be swapped with the Balsam
#  executor to potentially submit specific components to separate systems.
exctr = MPIExecutor()

for app in ddmd_apps:
    exctr.register_app(full_path=ddmd_apps[app], app_name=app)

# Specify directory structure where user functions will be called
ensemble_directory = os.path.abspath('./ensemble_' + str(datetime.datetime.today()).replace(' ', '_').split('.')[0])

MD_BATCH_SIZE = 12

# Parameterize our simulator function
sim_specs = {'sim_f': run_openmm_sim_f,
             'in': ['stage_id', 'gen_dir_loc', 'initial', 'task_id'],
             'out': [('sim_dir_loc', "<U70"), ('sim_cstat', int)],
             'user': {'sim_kill_minutes': 15,
                      'sim_length_ns': 1.0,
                      'poll_interval': 1,
                      'reference_pdb_file': os.path.abspath(folded_url.split('/')[-1]),
                      'config_file': 'molecular_dynamics.yaml'}
             }

# Parameterize our persistent generator function
gen_specs = {'gen_f': run_keras_cvae_ml_genf,
             'out': [('sim_id', int), ('stage_id', int), ('task_id', int),
                     ('initial', bool), ('gen_dir_loc', "<U70")],
             'user': {'initial_sample_size': MD_BATCH_SIZE,
                      'outliers': MD_BATCH_SIZE,
                      'last_n_h5_files': MD_BATCH_SIZE,  # ML stage
                      'n_most_recent_h5_files': MD_BATCH_SIZE,  # agent stage
                      'n_traj_frames': int(sim_specs['user']['sim_length_ns']*1000),  # agent stage
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

# Parameterize the provided allocation function
alloc_specs = {'alloc_f': alloc_f,
               'user': {'init_sample_size': MD_BATCH_SIZE}}

# Specify when libEnsemble should shut down
exit_criteria = {'sim_max': 120}

# Additional libEnsemble settings to customize our ensemble directory
libE_specs['sim_dirs_make'] = True
libE_specs['sim_input_dir'] = './sim'
libE_specs['sim_dir_symlink_files'] = [os.path.abspath(unfolded_url.split('/')[-1])]

libE_specs['gen_dirs_make'] = True
libE_specs['gen_input_dir'] = './gen'

libE_specs['ensemble_dir_path'] = ensemble_directory

persis_info = add_unique_random_streams({}, nworkers + 1)

# Call libEnsemble
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                            persis_info=persis_info,
                            alloc_specs=alloc_specs,
                            libE_specs=libE_specs)
