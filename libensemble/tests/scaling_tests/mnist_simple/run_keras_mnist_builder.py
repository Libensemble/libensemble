import os
import time

from libensemble.libE import libE
from libensemble.tools import parse_args, add_unique_random_streams, save_libE_output
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first

from keras_mnist_eval_model import keras_mnist_eval_model as sim_f
from keras_mnist_build_model import keras_mnist_build_model as gen_f

nworkers, is_manager, libE_specs, _ = parse_args()

dry_run = False
num_procs_for_app = 2
num_active_gens = nworkers
num_models_to_evaluate = 4

exctr = MPIExecutor(auto_resources=True)

gen_app = os.path.abspath('./tensorflow2_keras_mnist.py')

exctr.register_calc(full_path=gen_app, app_name='ml_keras_mnist')  # Named app

libE_specs['gen_dirs_make'] = True
libE_specs['ensemble_dir_path'] = './mnist_ensemble_' + time.asctime().replace(' ', '_')

sim_specs = {'sim_f': sim_f,  # evaluates resulting trained models, may bump epochs for future gens
             'in': ['model_file'],
             'out': [('loss', float, (1,)), ('accuracy', float, (1,)), ('epochs_to_try', int, (1,))],
             'user': {'eval_steps': 256}
             }

gen_specs = {'gen_f': gen_f,  # produces trained models, starting with 1 epoch
             'in': ['epochs_to_try'],  # Initially set by ['user']['init_epochs'], bumped by sim_f if accuracy low
             'out': [('model_file', "<U70", (1,)), ('cstat', int, (1,))],
             'user': {'num_procs': num_procs_for_app,
                      'init_epochs': 1,
                      'app_args': "--device cpu",
                      'dry_run': dry_run,
                      'time_limit': 1800}  # seconds
             }

alloc_specs = {'alloc_f': give_sim_work_first,
               'out': [('allocated', bool)],
               'user': {'batch_mode': False, 'num_active_gens': num_active_gens}}

persis_info = add_unique_random_streams({}, nworkers + 1)  # JLN: I *really* don't think I need this!

exit_criteria = {'sim_max': num_models_to_evaluate}

H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs=alloc_specs, libE_specs=libE_specs)

if is_manager:
    print('Routine complete!')
    save_libE_output(H, persis_info, __file__, nworkers)
    print(H.dtype.names)
    print(H)
