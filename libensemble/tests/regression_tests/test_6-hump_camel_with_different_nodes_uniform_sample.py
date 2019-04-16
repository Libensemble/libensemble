# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via the following command:
#    mpiexec -np 4 python3 test_6-hump_camel_with_different_nodes_uniform_sample.py
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """
from mpi4py import MPI # for libE communicator
import numpy as np
import argparse

# Import libEnsemble main, sim_specs, gen_specs, and persis_info
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel_with_different_ranks_and_nodes as sim_f
from libensemble.gen_funcs.uniform_sampling import uniform_random_sample_with_different_nodes_and_ranks as gen_f
from libensemble.tests.regression_tests.common import parse_args, save_libE_output, give_each_worker_own_stream

# Parse args for test code
nworkers, is_master, libE_specs, _ = parse_args()
if libE_specs['comms'] != 'mpi':
    # Can't do this one with processes either?  Wants a machine file.
    quit()

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--mfile', action="store", dest='machinefile',
    help='A machine file containing ordered list of nodes required for each libE rank'
)
args = parser.parse_args()

try:
    libE_machinefile = open(args.machinefile).read().splitlines()
except:
    if is_master:
        print("WARNING: No machine file provided - defaulting to local node")
    libE_machinefile = [MPI.Get_processor_name()]*MPI.COMM_WORLD.Get_size()

n = 2
sim_specs = {
    'sim_f': sim_f,
    'in': ['x', 'num_nodes', 'ranks_per_node'],
    'out': [('f', float)],
    'nodelist': libE_machinefile,}

gen_specs = {
    'gen_f': gen_f,
    'in': ['sim_id'],
    'out': [
        ('priority', float),
        ('num_nodes', int),
        ('ranks_per_node', int),
        ('x', float, n),
        ('x_on_cube', float, n),],
    'initial_batch_size': 5,
    'max_ranks_per_node': 8,
    'num_active_gens': 1,
    'batch_mode': False,
    'give_all_with_same_priority': True,
    'lb': np.array([-3, -2]),
    'ub': np.array([3, 2]),
    'max_num_nodes': nworkers # Used in uniform_random_sample_with_different_nodes_and_ranks,
}

persis_info = give_each_worker_own_stream({}, nworkers+1)

exit_criteria = {'sim_max': 10, 'elapsed_wallclock_time': 300}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            libE_specs=libE_specs)

if is_master:
    assert flag == 0

    save_libE_output(H, __file__, nworkers)
