# """
# Runs libEnsemble testing the executor functionality.
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_executor_hworld.py
#    python3 test_executor_hworld.py --nworkers 3 --comms local
#    python3 test_executor_hworld.py --nworkers 3 --comms tcp
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

import os
import time
import numpy as np
import multiprocessing

from libensemble.libE import libE
from libensemble.sim_funcs.distrib_ml_eval_model import distrib_ml_eval_model as sim_f
from libensemble.gen_funcs.distrib_ml_build_model import distrib_ml_build_model as gen_f

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 3 4

# JLN TODO: gen_f uses ML example, uses exctr to launch example that downloads data, generates keras model, writes to file
# JLN TODO: sim_f loads model, evaluates with with test data or orders predictions

nworkers, is_manager, libE_specs, _ = parse_args()

cores_per_task = 1
logical_cores = multiprocessing.cpu_count()
cores_all_tasks = nworkers*cores_per_task

if cores_all_tasks > logical_cores:
    use_auto_resources = False
    mess_resources = 'Oversubscribing - auto_resources set to False'
else:
    use_auto_resources = True
    mess_resources = 'Auto_resources set to True'

if is_manager:
    print('\nCores req: {} Cores avail: {}\n  {}\n'.format(cores_all_tasks, logical_cores, mess_resources))

from libensemble.executors.mpi_executor import MPIExecutor
exctr = MPIExecutor(auto_resources=use_auto_resources)

gen_app = '/Users/John-Luke/Desktop/libensemble/sdl_ai_workshop/01_distributedDeepLearning/Horovod/tensorflow2_keras_mnist.py'

exctr.register_calc(full_path=gen_app, app_name='ml_keras_mnist')  # Named app

sim_specs = {'sim_f': sim_f,
             'in': ['model_file'],
             'out': [('prediction', int)]
            }

gen_specs = {'gen_f': gen_f,
             'in': [],
             'out': [('model_file', "<U50"), ('cstat', int)],
             'user':{'num_procs': 4,
                     'app_args': "--device cpu --epochs 32",
                     'dry_run': True,
                     'time_limit': 1800}  # seconds
            }

libE_specs['gen_dirs_make'] = True
libE_specs['ensemble_dir_path'] = './' + time.asctime().replace(' ', '_')

exit_criteria = {'gen_max': 4}
persis_info = {}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            libE_specs=libE_specs)
