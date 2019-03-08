# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via the following command:
#    mpiexec -np 4 python3 test_6-hump_camel_with_different_nodes_uniform_sample.py
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import numpy as np

from libensemble.tests.regression_tests.support import save_libE_output

# Import libEnsemble main, sim_specs, gen_specs, and persis_info
from libensemble.libE import libE
from libensemble.tests.regression_tests.support import six_hump_camel_with_different_ranks_and_nodes_sim_specs as sim_specs
from libensemble.tests.regression_tests.support import uniform_random_sample_with_different_nodes_and_ranks_gen_specs as gen_specs
from libensemble.tests.regression_tests.support import persis_info_0 as persis_info


import argparse
#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m','--mfile',action="store",dest='machinefile',
                    help='A machine file containing ordered list of nodes required for each libE rank')
args = parser.parse_args()

try:
    libE_machinefile = open(args.machinefile).read().splitlines()
except:
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("WARNING: No machine file provided - defaulting to local node")
    libE_machinefile = [MPI.Get_processor_name()]*MPI.COMM_WORLD.Get_size()

n=2
sim_specs['nodelist'] = libE_machinefile
gen_specs['out'] += [('x',float,n), ('x_on_cube',float,n),]
gen_specs['lb'] = np.array([-3,-2])
gen_specs['ub'] = np.array([ 3, 2])

# Tell libEnsemble when to stop
exit_criteria = {'sim_max': 10, 'elapsed_wallclock_time': 300}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info)

if MPI.COMM_WORLD.Get_rank() == 0:
    assert flag == 0

    save_libE_output(H,__file__)
