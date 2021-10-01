# """
# Runs libEnsemble testing the MPI Runners command creation with uneven workers per node.
#
# This test must be run on an even number of workers >= 4 and <= 32 (e.g. odd no. of procs when using mpi4py).
#
# Execute via one of the following commands (e.g. 6 workers - one is zero resource):
#    mpiexec -np 7 python3 test_mpi_runners_zrw_subnode_uneven.py
#    python3 test_mpi_runners_zrw_subnode_uneven.py --nworkers 6 --comms local
#    python3 test_mpi_runners_zrw_subnode_uneven.py --nworkers 6 --comms tcp
# """

import sys
import numpy as np

from libensemble.libE import libE
from libensemble.sim_funcs.run_line_check import runline_check_by_worker as sim_f
from libensemble.gen_funcs.persistent_uniform_sampling import persistent_uniform as gen_f
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.tools import parse_args, add_unique_random_streams
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.tests.regression_tests.common import create_node_file
from libensemble import logger

# logger.set_level('DEBUG')  # For testing the test
logger.set_level('INFO')

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 5 7

nworkers, is_manager, libE_specs, _ = parse_args()
rounds = 1
sim_app = '/path/to/fakeapp.x'
comms = libE_specs['comms']

libE_specs['zero_resource_workers'] = [1]
libE_specs['dedicated_mode'] = True
libE_specs['enforce_worker_core_bounds'] = True

# To allow visual checking - log file not used in test
log_file = 'ensemble_mpi_runners_zrw_subnode_uneven_comms_' + str(comms) + '_wrks_' + str(nworkers) + '.log'
logger.set_filename(log_file)

# For varying size test - relate node count to nworkers
in_place = libE_specs['zero_resource_workers']
n_gens = len(in_place)
nsim_workers = nworkers - n_gens

if (nsim_workers % 2 == 0):
    sys.exit("This test must be run with an odd number of sim workers >= 3 and <= 31. There are {} sim workers.".format(
        nsim_workers))

comms = libE_specs['comms']
node_file = 'nodelist_mpi_runners_zrw_subnode_uneven_comms_' + str(comms) + '_wrks_' + str(nworkers)
nnodes = 2

# Mock up system
custom_resources = {
    'cores_on_node': (16, 64),  # Tuple (physical cores, logical cores)
    'node_file': node_file}  # Name of file containing a node-list
libE_specs['resource_info'] = custom_resources

if is_manager:
    create_node_file(num_nodes=nnodes, name=node_file)

if comms == 'mpi':
    libE_specs['mpi_comm'].Barrier()

# Create executor and register sim to it.
exctr = MPIExecutor(custom_info={'mpi_runner': 'srun'})
exctr.register_app(full_path=sim_app, calc_type='sim')

n = 2
sim_specs = {
    'sim_f': sim_f,
    'in': ['x'],
    'out': [('f', float)], }

gen_specs = {
    'gen_f': gen_f,
    'in': [],
    'out': [('x', float, (n, ))],
    'user': {
        'initial_batch_size': 20,
        'lb': np.array([-3, -2]),
        'ub': np.array([3, 2])}}

alloc_specs = {'alloc_f': alloc_f, 'out': []}
persis_info = add_unique_random_streams({}, nworkers + 1)
exit_criteria = {'sim_max': (nsim_workers) * rounds}

test_list_base = [
    {
        'testid': 'base1'},  # Give no config and no extra_args
]

# Example: On 5 workers, runlines should be ...
# [w1]: Gen only
# [w2]: srun -w node-1 --ntasks 5 --nodes 1 --ntasks-per-node 5 /path/to/fakeapp.x --testid base1
# [w3]: srun -w node-1 --ntasks 5 --nodes 1 --ntasks-per-node 5 /path/to/fakeapp.x --testid base1
# [w4]: srun -w node-1 --ntasks 5 --nodes 1 --ntasks-per-node 5 /path/to/fakeapp.x --testid base1
# [w5]: srun -w node-2 --ntasks 8 --nodes 1 --ntasks-per-node 8 /path/to/fakeapp.x --testid base1
# [w6]: srun -w node-2 --ntasks 8 --nodes 1 --ntasks-per-node 8 /path/to/fakeapp.x --testid base1

srun_p1 = 'srun -w '
srun_p2 = ' --ntasks '
srun_p3 = ' --nodes 1 --ntasks-per-node '
srun_p4 = ' /path/to/fakeapp.x --testid base1'

exp_tasks = []
exp_srun = []

# Hard coding an example for 2 nodes to avoid replicating general logic in libEnsemble.
low_wpn = nsim_workers // nnodes
high_wpn = nsim_workers // nnodes + 1

for i in range(nsim_workers):
    if i < (nsim_workers // nnodes + 1):
        nodename = 'node-1'
        ntasks = 16 // high_wpn
    else:
        nodename = 'node-2'
        ntasks = 16 // low_wpn
    exp_tasks.append(ntasks)
    exp_srun.append(srun_p1 + str(nodename) + srun_p2 + str(ntasks) + srun_p3 + str(ntasks) + srun_p4)

test_list = test_list_base
exp_list = exp_srun
sim_specs['user'] = {'tests': test_list, 'expect': exp_list, 'persis_gens': n_gens}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

# All asserts are in sim func
