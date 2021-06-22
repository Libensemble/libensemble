import os
import glob
import yaml
import time
import numpy as np

from libensemble.executors.executor import Executor


def update_config_file(H, sim_specs):
    """
    Parameterize the configuration file for the run_openmm.py application. Values
    such as 'experiment_directory', 'task_idx', and 'stage_idx' were produced by
    the generator function (are parsed from 'H', the History array).
    """
    here = os.getcwd()

    config_file = sim_specs['user']['config_file']
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    config['experiment_directory'] = os.path.abspath('../' + H['gen_dir_loc'][0])
    config['output_path'] = here
    config['initial_pdb_dir'] = here
    config['reference_pdb_file'] = os.path.join(here, '1FME-folded.pdb')
    config['simulation_length_ns'] = sim_specs['user']['sim_length_ns']
    config['task_idx'] = int(H['task_id'])
    config['stage_idx'] = int(H['stage_id'])

    #  Specify an unfolded pdb file if the simulation is an "initial" one
    if H['initial']:
        config['pdb_file'] = os.path.join(here, '1FME-unfolded.pdb')
    else:
        config['pdb_file'] = None

    with open(config_file, 'w') as f:
        yaml.dump(config, f)


def run_openmm_sim_f(H, persis_info, sim_specs, libE_info):
    """ Simulation user function for running DeepDriveMD's run_openmm.py via
    the Executor. libEnsemble's worker processes call this simulation function
    with a selection of the History array that contains values produced by the
    generator function.
    """

    calc_status = 0
    dry_run = sim_specs['user']['dry_run']

    #  Specify the Executor object created in the calling script.
    exctr = Executor.executor

    # Update the config file copy with relevant values
    update_config_file(H, sim_specs)

    config_file = sim_specs['user']['config_file']
    args = '-c ' + os.path.join(os.getcwd(), config_file)
    os.environ["OMP_NUM_THREADS"] = str(sim_specs['user']['omp_num_threads'])

    # Submit the molecular_dynamics app that was registered with the Executor.
    #  Only one process needed since bulk work presumably done on GPU. If not,
    #  then OpenMM app can take advantage of cores itself.
    task = exctr.submit(app_name='molecular_dynamics', app_args=args, wait_on_run=True,
                        dry_run=dry_run, num_procs=1, num_nodes=1, ranks_per_node=1)

    # Periodically poll our running task, then ensure the task created the expected output.
    if not dry_run:
        calc_status = exctr.polling_loop(task, timeout=sim_specs['user']['sim_kill_minutes']*60,
                                         delay=sim_specs['user']['poll_interval'])
        time.sleep(0.2)
        output_file = glob.glob('*.h5')
        assert len(output_file), \
            'MD Simulation did not write final output to .h5 file.'
    else:
        output_file = ['test.txt']

    current_dir = os.getcwd().split('/')[-1]  # sim_dir

    # Create a local History array to be populated with simulation values to send
    #  to the Manager and/or the generator function.
    H_o = np.zeros(1, dtype=sim_specs['out'])
    H_o['sim_cstat'] = calc_status
    H_o['file_path'] = os.path.join(current_dir, output_file[0])

    return H_o, persis_info, calc_status
