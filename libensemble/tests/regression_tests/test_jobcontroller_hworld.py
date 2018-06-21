from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import sys, os             # for adding to path
import numpy as np

# Import libEnsemble modules
from libensemble.libE import libE
from libensemble.sim_funcs.job_control_hworld import job_control_hworld
from libensemble.gen_funcs.uniform_sampling import uniform_random_sample
from libensemble.register import Register
from libensemble.controller import JobController, BalsamJobController

def build_simfunc():
    import subprocess
    
    #Build simfunc
    #buildstring='mpif90 -o my_simjob.x my_simjob.f90' # On cray need to use ftn
    buildstring='mpicc -o my_simjob.x ../unit_tests/simdir/my_simjob.c'
    #subprocess.run(buildstring.split(),check=True) #Python3.5+
    subprocess.check_call(buildstring.split())

script_name = os.path.splitext(os.path.basename(__file__))[0]

NCORES = 1
sim_app = './my_simjob.x'
if not os.path.isfile(sim_app):
    build_simfunc()

registry = Register()
jobctrl = JobController(registry = registry, auto_resources = False)
registry.register_calc(full_path=sim_app, calc_type='sim')
    
#State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': job_control_hworld, # This is the function whose output is being minimized
             'in': ['x'], # These keys will be given to the above function
             'out': [('f',float), # This is the output from the function being minimized
                    ],
             'cores': NCORES,
             'save_every_k': 400
             }

# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': uniform_random_sample,
             'in': ['sim_id'],
             'out': [('x',float,2),
                    ],
             'lb': np.array([-3,-2]),
             'ub': np.array([ 3, 2]),
             'gen_batch_size': 10,
             'batch_mode': True,
             'num_inst':1,
             'save_every_k': 10
             }
# Tell libEnsemble when to stop
exit_criteria = {'sim_max': 10}

np.random.seed(1)

# Perform the run
H, gen_info, flag = libE(sim_specs, gen_specs, exit_criteria)

if MPI.COMM_WORLD.Get_rank() == 0:
    assert flag == 0
    print("\n\n\nRun completed.")
    
