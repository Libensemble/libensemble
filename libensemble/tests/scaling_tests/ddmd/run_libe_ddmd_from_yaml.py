#!/usr/bin/env python
import os
import requests
import datetime

from libensemble import Ensemble
from libensemble.executors.mpi_executor import MPIExecutor

# Import DeepDriveMD components for registering with libEnsemble's executor
from deepdrivemd.sim.openmm import run_openmm
from deepdrivemd.aggregation.basic import aggregate
from deepdrivemd.models.keras_cvae import train
from deepdrivemd.selection.latest import select_model
from deepdrivemd.agents.lof import lof


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

ddmd_apps = {'molecular_dynamics': run_openmm.__file__,
             'aggregation': aggregate.__file__,
             'machine_learning': train.__file__,
             'model_selection': select_model.__file__,
             'agent': lof.__file__}

# Submit each component via a MPI runner. This Executor can be swapped with the Balsam
#  executor to potentially submit specific components to separate systems.
exctr = MPIExecutor()
api = Ensemble()

api.from_yaml('libE_ddmd.yaml')

api.sim_specs['user']['reference_pdb_file'] = os.path.abspath(folded_url.split('/')[-1])
api.gen_specs['user']['n_traj_frames'] = int(api.sim_specs['user']['sim_length_ns']*1000)

for app in ddmd_apps:
    exctr.register_calc(full_path=ddmd_apps[app], app_name=app)
    api.gen_specs['user'][app + '_config'] = app + '.yaml'
    api.gen_specs['out'].append((app + '_cstat', int))

api.libE_specs['sim_dir_symlink_files'] = [os.path.abspath(unfolded_url.split('/')[-1])]
api.libE_specs['ensemble_dir_path'] = \
    os.path.abspath('./ensemble_' + str(datetime.datetime.today()).replace(' ', '_').split('.')[0])

api.persis_info.add_unique_streams()

api.run()
