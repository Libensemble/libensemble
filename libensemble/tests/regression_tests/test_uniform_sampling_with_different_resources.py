# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_6-hump_camel_with_different_nodes_uniform_sample.py
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi
# TESTSUITE_NPROCS: 2 4

# Disable matching probes to work on all fabrics
import mpi4py
mpi4py.rc.recv_mprobe = False

import sys
from mpi4py import MPI
import numpy as np
import pkg_resources
import argparse

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel_with_different_ranks_and_nodes as sim_f
from libensemble.gen_funcs.sampling import uniform_random_sample_with_different_nodes_and_ranks as gen_f
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble.executors.mpi_executor import MPIExecutor

nworkers, is_master, libE_specs, _ = parse_args()

libE_specs['sim_dirs_make'] = True
libE_specs['ensemble_dir_path'] = './ensemble_diff_nodes_w' + str(nworkers)

if libE_specs['comms'] != 'mpi':
    # Can't do this one with processes either?  Wants a machine file.
    sys.exit("This test only runs with MPI -- aborting...")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mfile', action="store", dest='machinefile',
                    help='A machine file containing ordered list of nodes required for each libE rank')
args = parser.parse_args()

try:
    libE_machinefile = open(args.machinefile).read().splitlines()
except (TypeError, NameError):
    if is_master:
        print("WARNING: No machine file provided - defaulting to local node")
    libE_machinefile = [MPI.Get_processor_name()]*MPI.COMM_WORLD.Get_size()

sim_app = pkg_resources.resource_filename('libensemble.sim_funcs', 'helloworld.py')
exctr = MPIExecutor()
exctr.register_calc(full_path=sim_app, calc_type='sim')

n = 2
sim_specs = {'sim_f': sim_f,
             'in': ['x', 'num_nodes', 'ranks_per_node'],
             'out': [('f', float)],
             'user': {'nodelist': libE_machinefile}
             }

gen_specs = {'gen_f': gen_f,
             'in': ['sim_id'],
             'out': [('priority', float),
                     ('num_nodes', int),
                     ('ranks_per_node', int),
                     ('x', float, n),
                     ('x_on_cube', float, n)],
             'user': {'initial_batch_size': 5,
                      'max_ranks_per_node': 8,
                      'give_all_with_same_priority': True,
                      'max_num_nodes': nworkers,  # Used in uniform_random_sample_with_different_nodes_and_ranks,
                      'lb': np.array([-3, -2]),
                      'ub': np.array([3, 2])}
             }

alloc_specs = {'alloc_f': give_sim_work_first,
               'out': [('allocated', bool)],
               'user': {'batch_mode': False,
                        'num_active_gens': 1}}

persis_info = add_unique_random_streams({}, nworkers + 1)

exit_criteria = {'sim_max': 40, 'elapsed_wallclock_time': 300}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            libE_specs=libE_specs, alloc_specs=alloc_specs)

if is_master:
    assert flag == 0

    save_libE_output(H, persis_info, __file__, nworkers)
