__all__ = ['run_keras_cvae_ml_genf']

import os
import glob
import yaml
import json
import numpy as np
from libensemble.executors.executor import Executor
from libensemble.message_numbers import (STOP_TAG, PERSIS_STOP,
                                         FINISHED_PERSISTENT_GEN_TAG,
                                         EVAL_GEN_TAG)
from libensemble.tools.persistent_support import PersistentSupport


def get_stage(persis_info):
    return str(persis_info['stage_count']).zfill(4)


def update_config_file(user, app_type, pinfo):
    """
    Updates configuration files for each application prior to launching, and
    produces expected output directory structure for each.
    """
    with open(user[app_type + '_config'], 'r') as f:
        config = yaml.safe_load(f)

    output_path = os.getcwd() + '/{}_runs/stage'.format(app_type) +  \
                                get_stage(pinfo) + '/task0000'

    updates = {
        'experiment_directory': os.getcwd(),
        'output_path': output_path,
        'stage_idx': pinfo['stage_count']
    }

    if app_type == 'aggregation':
        updates.update({'last_n_h5_files': user['initial_sample_size']})

    elif app_type == 'machine_learning':
        updates.update({
            'model_tag': 'keras_cvae_model' + get_stage(pinfo),
            'last_n_h5_files': user['last_n_h5_files']
        })

    elif app_type == 'model_selection':
        updates.update({'checkpoint_dir': output_path.replace(app_type, 'machine_learning') + '/checkpoint'})

    elif app_type == 'agent':
        updates.update({
            'num_intrinsic_outliers': user['outliers'],
            'num_extrinsic_outliers': user['outliers'],
            'n_most_recent_h5_files': user['n_most_recent_h5_files'],
            'n_traj_frames': user['n_traj_frames']
        })

    os.makedirs(output_path, exist_ok=True)
    task_config = os.path.join(output_path, 'stage' + get_stage(pinfo) + '_task0000.yaml')

    config.update(updates)

    with open(task_config, 'w') as f:
        yaml.dump(config, f)

    return output_path, task_config


def submit_application(exctr, user, app_type, output_path, task_config):
    """
    Switches to an expected output directory, launches an application
    via libEnsemble's executor, then polls its status until it finishes.
    """
    start = os.getcwd()
    os.chdir(output_path)

    args = '-c ' + os.path.join(os.getcwd(), task_config)
    task = exctr.submit(app_name=app_type, app_args=args, wait_on_start=True,
                        num_procs=1, num_nodes=1, procs_per_node=1)

    calc_status = exctr.polling_loop(task, timeout=user[app_type + '_kill_minutes']*60, delay=1)
    os.chdir(start)
    return calc_status


def postprocess_md_sim_dirs(calc_in, pinfo):
    """
    Symlink the Molecular Dynamics results into directories that resemble
    DeepDriveMD's output.
    """
    expected_md_dir = './molecular_dynamics_runs/stage' + get_stage(pinfo)
    os.makedirs(expected_md_dir)
    for entry in calc_in:
        base_task_dir = 'task' + str(entry['task_id']).zfill(4)
        full_task_dir = os.path.join(expected_md_dir, base_task_dir)
        sim_dir = entry['sim_dir_loc']
        os.symlink(os.path.abspath('../' + sim_dir), os.path.abspath(full_task_dir))


def generate_initial_md_runs(gen_specs, persis_info):
    """
    Generate an initial local History array, and populate with an initial set
    of parameters for an initial set of MD simulations.
    """
    persis_info['stage_count'] += 1

    sample_size = gen_specs['user']['initial_sample_size']
    local_H = np.zeros(sample_size, dtype=gen_specs['out'])

    local_H['task_id'] = np.arange(sample_size)
    local_H['initial'] = True
    local_H['gen_dir_loc'] = os.getcwd().split('/')[-1]
    local_H['sim_id'] = np.arange(sample_size)
    local_H['stage_id'] = 0

    persis_info['last_sim_id'] = local_H['sim_id'][-1]

    return local_H, persis_info


def generate_subsequent_md_runs(gen_specs, persis_info, local_H, output_path):
    """
    Generate subsequent MD simulation run parameters in the local History array,
    based on the number of outlier points detected by the Agent application.
    """
    persis_info['stage_count'] += 1

    presumed_agent_output = glob.glob(output_path + '/stage*_task*.json')[0]
    with open(os.path.join(output_path, presumed_agent_output), 'r') as f:
        sample_size = len(json.load(f))

    local_H.resize(len(local_H) + sample_size, refcheck=False)

    local_H['task_id'][-sample_size:] = np.arange(sample_size)
    local_H['initial'][-sample_size:] = False
    local_H['gen_dir_loc'][-sample_size:] = os.getcwd().split('/')[-1]

    subs_sim_id = persis_info['last_sim_id'] + 1
    local_H['sim_id'][-sample_size:] = np.arange(subs_sim_id, subs_sim_id + sample_size)
    local_H['stage_id'][-sample_size:] = persis_info['stage_count']

    persis_info['last_sim_id'] = local_H['sim_id'][-1]

    return local_H, persis_info


def skip_app(gen_specs, app):
    """
    Optionally skip certain apps, if specified in gen_specs['user']
    """
    if 'skip_' + app in gen_specs['user']:
        if gen_specs['user']['skip_' + app]:
            return True
    return False


def run_keras_cvae_ml_genf(H, persis_info, gen_specs, libE_info):
    """ Persistent Generator user function for processing simulation output and
    running via the Executor each of the remaining DeepDriveMD applications concerned
    with simulation output. This generator does not return until libEnsemble
    concludes.

    On initialization, this generator function produces an initial set of parameters
    for an initial set of simulation function calls, then sends the local History
    array containing these values directly to the Manager, which will distribute
    the work accordingly.

    After this, the persistent generator waits until all the results are available
    from the Manager, preprocesses some of the output, then configures and launches
    the other DeepDriveMD applications in a sequence. The final app's output
    determines the number of future candidate simulations. The local History array
    is updated, then sent directly to the Manager.
    """

    user = gen_specs['user']
    exctr = Executor.executor
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    persis_info['stage_count'] = -1
    os.environ["OMP_NUM_THREADS"] = '4'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(libE_info['workerID'] - 1)
    initial_complete = False
    tag = None

    while True:
        if not initial_complete:
            local_H, persis_info = generate_initial_md_runs(gen_specs, persis_info)
            # Send initial MD run parameters directly to the Manager
            ps.send(local_H)
            initial_complete = True
        else:
            # Wait for batch of MD results
            tag, Work, calc_in = ps.recv()
            if tag in [STOP_TAG, PERSIS_STOP]:  # Generator instructed to stop
                break

            # Symlink MD data into directory structure expected by future apps
            postprocess_md_sim_dirs(calc_in, persis_info)

            # Run each subsequent DeepDriveMD app
            for app in ['aggregation', 'machine_learning', 'model_selection', 'agent']:
                if skip_app(gen_specs, app):
                    continue
                output_path, task_config = update_config_file(user, app, persis_info)
                calc_status = submit_application(exctr, user, app, output_path, task_config)
                local_H[app + '_cstat'][Work['libE_info']['H_rows']] = calc_status

            # Produce subsequent set of MD runs parameters based on the final app's results
            local_H, persis_info = generate_subsequent_md_runs(gen_specs, persis_info, local_H, output_path)
            # Send subsequent MD run parameters directly to the Manager
            ps.send(local_H)

    return local_H, persis_info, FINISHED_PERSISTENT_GEN_TAG
