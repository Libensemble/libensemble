

__all__ = ['run_agg_ml_gen_f']

import os
import glob
import time
import json
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


def update_agg_config_file(user):
    with open(user['agg_config_file'], 'r') as f:
        config = yaml.safe_load(f)

    config['experiment_directory'] = os.getcwd()
    config['output_path'] = os.getcwd() + '/aggregation_runs/stage' + str(agg_count).zfill(4) + '/task0000'
    config['last_n_h5_files'] = user['initial_sample_size']

    with open(user['agg_config_file'], 'w') as f:
        yaml.dump(config, f)


def update_ml_config_file(user):
    with open(user['ml_config_file'], 'r') as f:
        config = yaml.safe_load(f)

    config['experiment_directory'] = os.getcwd()
    config['output_path'] = os.getcwd() + '/keras_cvae_model' + str(agg_count).zfill(4)
    config['model_tag'] = 'keras_cvae_model' + str(agg_count).zfill(4)

    with open(user['ml_config_file'], 'w') as f:
        yaml.dump(config, f)

    return config['output_path']


def update_selection_config_file(user, ml_output_dir):
    with open(user['sel_config_file'], 'r') as f:
        config = yaml.safe_load(f)

    config['experiment_directory'] = os.getcwd()
    config['output_path'] = os.getcwd() + '/model_selection' + str(agg_count).zfill(4)
    config['checkpoint_dir'] = ml_output_dir + '/checkpoint'

    with open(user['sel_config_file'], 'w') as f:
        yaml.dump(config, f)


def update_agent_config_file(user):
    with open(user['agent_config_file'], 'r') as f:
        config = yaml.safe_load(f)

    config['experiment_directory'] = os.getcwd()
    config['output_path'] = os.getcwd() + '/agent_runs/stage' + str(agg_count).zfill(4) + '/task0000'

    os.makedirs(config['output_path'], exist_ok=True)

    with open(user['agent_config_file'], 'w') as f:
        yaml.dump(config, f)


def submit_application(exctr, exctr_app_name, app_type, output_name, user):
    args = '-c ' + os.path.join(os.getcwd(), user[app_type + '_config_file'])
    task = exctr.submit(app_name=exctr_app_name, app_args=args, wait_on_run=True,
                        num_procs=1, num_nodes=1, ranks_per_node=1)

    calc_status = polling_loop(task, user['poll_interval'], user['agg_kill_minutes'])
    time.sleep(0.2)
    assert len(glob.glob(output_name + '*')), \
        output_name + " task didn't produce detectable output"
    return calc_status


def preprocess_md_dirs(calc_in):
    agg_expected_md_dir = './molecular_dynamics_runs/stage' + str(agg_count).zfill(4)
    for sim_id in calc_in['sim_id']:
        base_task_dir = 'task' + str(sim_id).zfill(4)
        agg_task_dir = os.path.join(agg_expected_md_dir, base_task_dir)
        h5file = calc_in['file_path'][sim_id][0]
        shutil.copytree('../' + h5file.split('/')[-2], agg_task_dir)


def postprocess_ml_dir(user, ml_output_dir):
    sel_expected_ml_dir = './machine_learning_runs/stage' + \
        str(agg_count).zfill(4) + '/task0000'
    shutil.copytree(ml_output_dir, sel_expected_ml_dir)
    shutil.copy(user['ml_config_file'], os.path.join(sel_expected_ml_dir, 'stage' + str(agg_count).zfill(4) + '_task0000.yaml'))
    os.makedirs('./model_selection_runs/stage' + str(agg_count).zfill(4) + '/task0000')
    os.makedirs('./agent_runs/stage' + str(agg_count).zfill(4) + '/task0000')


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

                if not user['skip_aggregation']:
                    update_agg_config_file(user)
                    agg_cstat = submit_application(exctr, 'run_aggregate', 'agg', 'aggregation', user)
                    local_H['agg_cstat'][Work['libE_info']['H_rows']] = agg_cstat
                else:
                    local_H['agg_cstat'][Work['libE_info']['H_rows']] = 0

                ml_output_dir = update_ml_config_file(user)
                ml_cstat = submit_application(exctr, 'run_ml_train', 'ml', 'keras_cvae_model', user)
                postprocess_ml_dir(user, ml_output_dir)
                local_H['ml_cstat'][Work['libE_info']['H_rows']] = ml_cstat

                update_selection_config_file(user, ml_output_dir)
                sel_cstat = submit_application(exctr, 'run_model_select', 'sel', 'model_selection', user)
                local_H['sel_cstat'][Work['libE_info']['H_rows']] = sel_cstat

                update_agent_config_file(user)
                agent_cstat = submit_application(exctr, 'run_outlier_agent', 'agent', 'agent', user)
                local_H['agent_cstat'][Work['libE_info']['H_rows']] = agent_cstat

                print('all done!')
                # local_H, persis_info = produce_subsequent_md_runs(local_H, gen_specs, persis_info)
                # send_mgr_worker_msg(comm, local_H)

    return local_H, persis_info, FINISHED_PERSISTENT_GEN_TAG
