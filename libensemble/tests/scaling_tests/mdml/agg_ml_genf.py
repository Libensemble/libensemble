

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
            task.kill()  # Timeout

    # Set calc_status with optional prints.
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


def update_agg_config_file(user, md_dir):
    with open(user['agg_config_file'], 'r') as f:
        config = yaml.safe_load(f)

    config['experiment_directory'] = os.getcwd()
    config['output_path'] = os.getcwd() + '/aggregation' + str(agg_count).zfill(4)
    config['last_n_h5_files'] = user['initial_sample_size']

    with open(user['agg_config_file'], 'w') as f:
        yaml.dump(config, f)


def submit_aggregation_app(user, exctr):

    dry_run = user['agg_dry_run']
    args = '-c ' + os.path.join(os.getcwd(), user['agg_config_file'])
    task = exctr.submit(app_name='run_aggregate', app_args=args, wait_on_run=True,
                        dry_run=dry_run, num_procs=1, num_nodes=1, ranks_per_node=1)

    if not dry_run:
        calc_status = polling_loop(task, user['poll_interval'], user['agg_kill_minutes'])
        time.sleep(0.2)
        return glob.glob('aggregation*'), calc_status
    else:
        return 0, 0


def preprocess_md_dirs(calc_in):
    agg_expected_md_dir = './molecular_dynamics_runs/stage' + str(agg_count).zfill(4)
    for sim_id in calc_in['sim_id']:
        base_task_dir = 'task' + str(sim_id).zfill(4)
        agg_task_dir = os.path.join(agg_expected_md_dir, base_task_dir)
        h5file = calc_in['file_path'][sim_id][0]
        shutil.copytree('../' + h5file.split('/')[-2], agg_task_dir)

    return agg_expected_md_dir


def produce_initial_parameter_sample(gen_specs, persis_info):
    user = gen_specs['user']
    initial_sample_size = user['initial_sample_size']
    pr = user['parameter_range']

    init_H = np.zeros(initial_sample_size, dtype=gen_specs['out'])
    sampled_points = persis_info['rand_stream'].uniform(pr[0], pr[1], initial_sample_size)
    init_H[user['sample_parameter_name']] = sampled_points
    init_H['sim_id'] = np.arange(initial_sample_size)
    return init_H


def run_agg_ml_gen_f(H, persis_info, gen_specs, libE_info):
    comm = libE_info['comm']
    user = gen_specs['user']
    exctr = Executor.executor
    initial_complete = False
    tag = None

    while True:
        if not initial_complete:  # initial batch
            local_H = produce_initial_parameter_sample(gen_specs, persis_info)
            send_mgr_worker_msg(comm, local_H)
            initial_complete = True
        else:
            while tag not in [STOP_TAG, PERSIS_STOP]:
                tag, Work, calc_in = get_mgr_worker_msg(comm)

                md_dir = preprocess_md_dirs(calc_in)
                update_agg_config_file(user, md_dir)
                sim_agg_out, cstat = submit_aggregation_app(user, exctr)

                if not len(sim_agg_out):
                    return None, persis_info, TASK_FAILED
                else:
                    local_H['agg_cstat'][Work['libE_info']['H_rows']] = cstat

    return local_H, persis_info, FINISHED_PERSISTENT_GEN_TAG
