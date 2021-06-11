

__all__ = ['run_agg_ml_gen_f']

import os
import glob
import time
import yaml
import shutil
import numpy as np
from libensemble.executors.executor import Executor
from libensemble.message_numbers import (STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG,
                                         WORKER_DONE, WORKER_KILL, TASK_FAILED)
from libensemble.tools.gen_support import get_mgr_worker_msg, send_mgr_worker_msg

agg_count = 0


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


def update_config_file(user, app_type, optional_path=None):
    with open(user[app_type + '_config'], 'r') as f:
        config = yaml.safe_load(f)

    config['experiment_directory'] = os.getcwd()
    config['output_path'] = os.getcwd() + '/{}_runs/stage'.format(app_type) + str(agg_count).zfill(4) + '/task0000'

    if app_type == 'aggregation':
        config['last_n_h5_files'] = user['initial_sample_size']
    elif app_type == 'machine_learning':
        config['model_tag'] = 'keras_cvae_model' + str(agg_count).zfill(4)
    elif app_type == 'model_selection':
        config['checkpoint_dir'] = optional_path + '/checkpoint'
    elif app_type == 'agent':
        os.makedirs(config['output_path'], exist_ok=True)

    with open(user[app_type + '_config'], 'w') as f:
        yaml.dump(config, f)

    return config['output_path']


def submit_application(exctr, user, app_type):
    args = '-c ' + os.path.join(os.getcwd(), user[app_type + '_config'])
    task = exctr.submit(app_name=app_type, app_args=args, wait_on_run=True,
                        num_procs=1, num_nodes=1, ranks_per_node=1)

    calc_status = polling_loop(task, user['poll_interval'], user[app_type + '_kill_minutes'])
    time.sleep(0.2)
    assert len(glob.glob(app_name + '_runs*')), \
        app_name + " task didn't produce detectable output"
    return calc_status


def preprocess_md_dirs(calc_in):
    agg_expected_md_dir = './molecular_dynamics_runs/stage' + str(agg_count).zfill(4)
    for sim_id in calc_in['sim_id']:
        base_task_dir = 'task' + str(sim_id).zfill(4)
        agg_task_dir = os.path.join(agg_expected_md_dir, base_task_dir)
        h5file = calc_in['file_path'][sim_id][0]
        shutil.copytree('../' + h5file.split('/')[-2], agg_task_dir)


# def postprocess_ml_dir(user, ml_output_dir):
#     sel_expected_ml_dir = './machine_learning_runs/stage' + \
#         str(agg_count).zfill(4) + '/task0000'
#     shutil.copytree(ml_output_dir, sel_expected_ml_dir)
#     shutil.copy(user['ml_config_file'], os.path.join(sel_expected_ml_dir,
#                                                      'stage' + str(agg_count).zfill(4) + '_task0000.yaml'))
#     os.makedirs('./model_selection_runs/stage' + str(agg_count).zfill(4) + '/task0000')


def produce_initial_parameter_sample(gen_specs, persis_info):
    user = gen_specs['user']
    initial_sample_size = user['initial_sample_size']
    pr = user['parameter_range']

    init_H = np.zeros(initial_sample_size, dtype=gen_specs['out'])
    sampled_points = persis_info['rand_stream'].uniform(pr[0], pr[1], initial_sample_size)
    init_H[user['sample_parameter_name']] = sampled_points
    init_H['sim_id'] = np.arange(initial_sample_size)
    init_H['do_initial'] = [True for i in range(initial_sample_size)]
    persis_info['last_sim_id'] = init_H['sim_id'][-1]
    return init_H, persis_info


def produce_subsequent_md_runs(local_H, gen_specs, persis_info):
    pass


def run_agg_ml_gen_f(H, persis_info, gen_specs, libE_info):
    comm = libE_info['comm']
    user = gen_specs['user']
    exctr = Executor.executor
    initial_complete = False
    apps = ['aggregation', 'machine_learning', 'model_selection', 'agent']
    tag = None

    while True:
        if not initial_complete:
            local_H, persis_info = produce_initial_parameter_sample(gen_specs, persis_info)
            send_mgr_worker_msg(comm, local_H)
            initial_complete = True
        else:
            while tag not in [STOP_TAG, PERSIS_STOP]:
                tag, Work, calc_in = get_mgr_worker_msg(comm)

                preprocess_md_dirs(calc_in)

                for app in apps:
                    if 'skip_' + app in gen_specs['user']:
                        if gen_specs['user']['skip_' + app]:
                            continue
                    update_config_file(user, app)
                    calc_status = submit_application(exctr, user, app)
                    local_H[app + '_cstat'][Work['libE_info']['H_rows']] = calc_status

                print('all done!')
                # local_H, persis_info = produce_subsequent_md_runs(local_H, gen_specs, persis_info)
                # send_mgr_worker_msg(comm, local_H)

    return local_H, persis_info, FINISHED_PERSISTENT_GEN_TAG
