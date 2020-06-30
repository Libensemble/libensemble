# """
# Runs libEnsemble testing the in_place gen argument.
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_inplace_gen.py
#    python3 test_inplace_gen.py --nworkers 3 --comms local
#    python3 test_inplace_gen.py --nworkers 3 --comms tcp
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

import os
import sys
import numpy as np

from libensemble.message_numbers import WORKER_DONE
from libensemble.libE import libE
from libensemble.gen_funcs.persistent_uniform_sampling import persistent_uniform as gen_f
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.tools import parse_args, add_unique_random_streams
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble import libE_logger

# libE_logger.set_level('DEBUG')  # For testing the test
libE_logger.set_level('INFO')

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 3 4

nodes_per_worker = 2


def exp_nodelist_for_worker(exp_list, workerID):
    """Modify expected node-lists based on workerID"""
    comps = exp_list.split()
    new_line = []
    for comp in comps:
        if comp.startswith('node-'):
            new_node_list = []
            node_list = comp.split(',')
            for node in node_list:
                node_name, node_num = node.split('-')
                new_num = int(node_num) + nodes_per_worker*(workerID - 2)  # For 1 persistent gen
                new_node = '-'.join([node_name, str(new_num)])
                new_node_list.append(new_node)
            new_list = ','.join(new_node_list)
            new_line.append(new_list)
        else:
            new_line.append(comp)
    return ' '.join(new_line)


def runline_check(H, persis_info, sim_specs, libE_info):
    """Check run-lines produced by executor provided by a list"""
    calc_status = 0
    x = H['x'][0][0]
    exctr = MPIExecutor.executor
    test_list = sim_specs['user']['tests']
    exp_list = sim_specs['user']['expect']

    for i, test in enumerate(test_list):
        task = exctr.submit(calc_type='sim',
                            num_procs=test.get('nprocs', None),
                            num_nodes=test.get('nnodes', None),
                            ranks_per_node=test.get('ppn', None),
                            extra_args=test.get('e_args', None),
                            app_args='--testid ' + test.get('testid', None),
                            stdout='out.txt',
                            stderr='err.txt',
                            hyperthreads=test.get('ht', None),
                            dry_run=True)

        outline = task.runline
        new_exp_list = exp_nodelist_for_worker(exp_list[i], libE_info['workerID'])

        if outline != new_exp_list:
            print('outline is: {}\nexp     is: {}'.format(outline, new_exp_list), flush=True)

        assert(outline == new_exp_list)

    calc_status = WORKER_DONE
    output = np.zeros(1, dtype=sim_specs['out'])
    output['f'][0] = np.linalg.norm(x)
    return output, persis_info, calc_status

# --------------------------------------------------------------------


nworkers, is_master, libE_specs, _ = parse_args()
rounds = 1
sim_app = '/path/to/fakeapp.x'
comms = libE_specs['comms']
libE_specs['in_place_workers'] = [1]


# To allow visual checking - log file not used in test
log_file = 'ensemble_inplace_workers_comms_' + str(comms) + '_wrks_' + str(nworkers) + '.log'
libE_logger.set_filename(log_file)

# For varying size test - relate node count to nworkers
in_place = libE_specs['in_place_workers']
nsim_workers = nworkers-len(in_place)
comms = libE_specs['comms']
nodes_per_worker = 2
node_file = 'nodelist_in_place_workers_' + str(comms) + '_wrks_' + str(nworkers)
if is_master:
    if os.path.exists(node_file):
        os.remove(node_file)
    with open(node_file, 'w') as f:
        for i in range(1, (nsim_workers)*nodes_per_worker + 1):
            f.write('node-' + str(i) + '\n')
        f.flush()
        os.fsync(f)
if comms == 'mpi':
    libE_specs['comm'].Barrier()


# Mock up system
customizer = {'mpi_runner': 'mpich',    # Select runner: mpich, openmpi, aprun, srun, jsrun
              'runner_name': 'mpirun',  # Runner name: Replaces run command if not None
              'cores_on_node': (16, 64),   # Tuple (physical cores, logical cores)
              'node_file': node_file}      # Name of file containing a node-list

# Create executor and register sim to it.
exctr = MPIExecutor(in_place_workers=in_place, central_mode=True, auto_resources=True, custom_info=customizer)
exctr.register_calc(full_path=sim_app, calc_type='sim')


if nworkers < 2:
    sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

n = 2
sim_specs = {'sim_f': runline_check,
             'in': ['x'],
             'out': [('f', float)],
             }

gen_specs = {'gen_f': gen_f,
             'in': [],
             'out': [('x', float, (n,))],
             'user': {'gen_batch_size': 20,
                      'lb': np.array([-3, -2]),
                      'ub': np.array([3, 2])}
             }

alloc_specs = {'alloc_f': alloc_f, 'out': [('given_back', bool)]}
persis_info = add_unique_random_streams({}, nworkers + 1)
exit_criteria = {'sim_max': (nsim_workers)*rounds}

# Each worker has 2 nodes. Basic test list for portable options
test_list_base = [{'testid': 'base1', 'nprocs': 2, 'nnodes': 1, 'ppn': 2, 'e_args': '--xarg 1'},  # Under use
                  {'testid': 'base2'},  # Give no config and no extra_args
                  ]

exp_mpich = \
    ['mpirun -hosts node-1 -np 2 --ppn 2 --xarg 1 /path/to/fakeapp.x --testid base1',
     'mpirun -hosts node-1,node-2 -np 32 --ppn 16 /path/to/fakeapp.x --testid base2',
     ]

test_list = test_list_base
exp_list = exp_mpich
sim_specs['user'] = {'tests': test_list, 'expect': exp_list}


# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs, libE_specs)

# All asserts are in sim func
