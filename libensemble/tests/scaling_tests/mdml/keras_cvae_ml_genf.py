

__all__ = ['run_keras_cvae_ml_genf']

import os
import glob
import time
import yaml
import json
import shutil
import numpy as np
from libensemble.executors.executor import Executor
from libensemble.message_numbers import (STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG,
                                         WORKER_DONE, WORKER_KILL, TASK_FAILED)
from libensemble.tools.gen_support import get_mgr_worker_msg, send_mgr_worker_msg


def get_stage(persis_info):
    return str(persis_info['stage_count']).zfill(4)


def polling_loop(task, poll_interval, kill_minutes):
    while(not task.finished):
        time.sleep(poll_interval)
        task.poll()
        if task.runtime > kill_minutes*60:
            task.kill()

    if task.finished:
        if task.state == 'FINISHED':
            calc_status = WORKER_DONE
        elif task.state == 'FAILED':
            print("Warning: Task {} failed: Error code {}"
                  .format(task.name, task.errcode))
            calc_status = TASK_FAILED
        elif task.state == 'USER_KILLED':
            print("Warning: Task {} has been killed"
                  .format(task.name))
            calc_status = WORKER_KILL
        else:
            print("Warning: Task {} in unknown state {}. Error code {}"
                  .format(task.name, task.state, task.errcode))

    return calc_status


def update_config_file(user, app_type, pinfo):
    with open(user[app_type + '_config'], 'r') as f:
        config = yaml.safe_load(f)

    output_path = os.getcwd() + '/{}_runs/stage'.format(app_type) + get_stage(pinfo) + '/task0000'
    config['experiment_directory'] = os.getcwd()
    config['output_path'] = output_path
    config['stage_idx'] = pinfo['stage_count']

    if app_type == 'aggregation':
        config['last_n_h5_files'] = user['initial_sample_size']
    elif app_type == 'machine_learning':
        config['model_tag'] = 'keras_cvae_model' + get_stage(pinfo)
    elif app_type == 'model_selection':
        config['checkpoint_dir'] = output_path.replace(app_type, 'machine_learning') + '/checkpoint'
    elif app_type == 'agent':
        config['num_intrinsic_outliers'] = user['outliers']
        config['num_extrinsic_outliers'] = user['outliers']

    os.makedirs(output_path, exist_ok=True)
    task_config = os.path.join(output_path, 'stage' + get_stage(pinfo) + '_task0000.yaml')

    with open(task_config, 'w') as f:
        yaml.dump(config, f)

    return output_path, task_config


def submit_application(exctr, user, app_type, output_path, task_config):
    start = os.getcwd()
    os.chdir(output_path)
    args = '-c ' + os.path.join(os.getcwd(), task_config)

    task = exctr.submit(app_name=app_type, app_args=args, wait_on_run=True,
                        num_procs=1, num_nodes=1, ranks_per_node=1)

    calc_status = polling_loop(task, user['poll_interval'], user[app_type + '_kill_minutes'])
    os.chdir(start)
    return calc_status


def preprocess_md_dirs(calc_in, pinfo):
    agg_expected_md_dir = './molecular_dynamics_runs/stage' + get_stage(pinfo)
    for entry in calc_in:
        base_task_dir = 'task' + str(entry['task_id']).zfill(4)
        agg_task_dir = os.path.join(agg_expected_md_dir, base_task_dir)
        h5file = entry['file_path']
        shutil.copytree('../' + h5file.split('/')[-2], agg_task_dir)


def generate_initial_md_runs(gen_specs, persis_info):
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
    if 'skip_' + app in gen_specs['user']:
        if gen_specs['user']['skip_' + app]:
            return True
    return False


def run_keras_cvae_ml_genf(H, persis_info, gen_specs, libE_info):
    comm = libE_info['comm']
    user = gen_specs['user']
    exctr = Executor.executor
    initial_complete = False
    apps = ['aggregation', 'machine_learning', 'model_selection', 'agent']
    persis_info['stage_count'] = -1
    tag = None

    while True:
        if not initial_complete:
            local_H, persis_info = generate_initial_md_runs(gen_specs, persis_info)
            send_mgr_worker_msg(comm, local_H)
            initial_complete = True
        else:
            tag, Work, calc_in = get_mgr_worker_msg(comm)
            if tag in [STOP_TAG, PERSIS_STOP]:
                break

            preprocess_md_dirs(calc_in, persis_info)

            for app in apps:
                if skip_app(gen_specs, app):
                    continue
                output_path, task_config = update_config_file(user, app, persis_info)
                calc_status = submit_application(exctr, user, app, output_path, task_config)
                local_H[app + '_cstat'][Work['libE_info']['H_rows']] = calc_status

            local_H, persis_info = generate_subsequent_md_runs(gen_specs, persis_info, local_H, output_path)
            send_mgr_worker_msg(comm, local_H)

    return local_H, persis_info, FINISHED_PERSISTENT_GEN_TAG
