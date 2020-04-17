# """
# Runs libEnsemble with uniform random sampling and writes results into sim dirs.
#   tests  per-calculation sim_dir capabilities
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_worker_exceptions.py
#    python3 test_worker_exceptions.py --nworkers 3 --comms local
#    python3 test_worker_exceptions.py --nworkers 3 --comms tcp
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4

import numpy as np
import os

from libensemble.libE import libE
from libensemble.tests.regression_tests.support import write_func as sim_f
from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f
from libensemble.tools import parse_args, add_unique_random_streams

nworkers, is_master, libE_specs, _ = parse_args()

sim_input_dir = './sim_input_dir'
dir_to_copy = sim_input_dir + '/copy_this'
o_ensemble = './ensemble_inputdir_w' + str(nworkers) + '_' + libE_specs.get('comms')

for dir in [sim_input_dir, dir_to_copy]:
    if is_master and not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)

libE_specs['sim_input_dir'] = sim_input_dir
libE_specs['sim_dir_path'] = o_ensemble

sim_specs = {'sim_f': sim_f, 'in': ['x'], 'out': [('f', float)]}

gen_specs = {'gen_f': gen_f,
             'out': [('x', float, (1,))],
             'user': {'gen_batch_size': 20,
                      'lb': np.array([-3]),
                      'ub': np.array([3]),
                      }
             }

persis_info = add_unique_random_streams({}, nworkers + 1)

exit_criteria = {'sim_max': 21}

H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                            persis_info, libE_specs=libE_specs)

if is_master:
    assert os.path.isdir(o_ensemble), 'Ensemble directory {} not created.'.format(o_ensemble)
    dir_sum = sum(['worker' in i for i in os.listdir(o_ensemble)])
    assert dir_sum == exit_criteria['sim_max'], \
        'Number of sim directories ({}) does not match sim_max ({}).'\
        .format(dir_sum, exit_criteria['sim_max'])

    input_copied = []

    for base, files, _ in os.walk(o_ensemble):
        basedir = base.split('/')[-1]
        if basedir.startswith('sim'):
            input_copied.append(all([os.path.basename(j) in files for j in
                                    os.listdir(sim_input_dir)]))

    assert all(input_copied), \
        'Exact input files not copied to each calculation directory'
