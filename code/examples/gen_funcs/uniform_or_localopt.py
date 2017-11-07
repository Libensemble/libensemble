from __future__ import division
from __future__ import absolute_import

import numpy as np
from mpi4py import MPI
import sys

from message_numbers import FINISHED_PERSISTENT_GEN_TAG
from message_numbers import STOP_TAG
from message_numbers import PERSIS_STOP
from message_numbers import PERSIS_GEN_TAG
from message_numbers import EVAL_GEN_TAG

import nlopt
def try_and_run_nlopt(H, gen_specs, libE_info):
    """ Set up objective and runs nlopt
    """

    def nlopt_obj_fun(x, grad, H, gen_specs, libE_info):
        # import ipdb; ipdb.set_trace(context=21) 
        if np.array_equiv(x, H['x']):
            grad[:] = H['grad']
            return np.float(H['f'])

        # Send back x to the manager
        O = np.zeros(1, dtype=gen_specs['out'])
        O = add_to_O(O,x,0,gen_specs['ub'],gen_specs['lb'],local=True,active=True)
        D = {'calc_out':O}
        libE_info['comm'].send(obj=D,dest=0,tag=PERSIS_GEN_TAG)


        # Receive information from the manager (or a STOP_TAG) 
        status = MPI.Status()
        E = libE_info['comm'].recv(buf=None,source=0,tag=MPI.ANY_TAG,status=status)
        tag = status.Get_tag()
        if tag in [STOP_TAG, PERSIS_STOP]:
            nlopt.forced_stop.message = 'tag=' + str(tag)
            raise nlopt.forced_stop

        if gen_specs['localopt_method'] in ['LD_MMA']:
            grad[:] = E['grad']
            f = E['f']

        return f

    # import ipdb; ipdb.set_trace(context=21) 
    x0 = H['x'].flatten()

    n = len(gen_specs['ub'])

    opt = nlopt.opt(getattr(nlopt,gen_specs['localopt_method']), n)

    # lb = np.zeros(n)
    # ub = np.ones(n)

    lb = gen_specs['lb']
    ub = gen_specs['ub']

    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)

    # Care must be taken here because a too-large initial step causes nlopt to move the starting point!
    dist_to_bound = min(min(ub-x0),min(x0-lb))

    if 'dist_to_bound_multiple' in gen_specs:
        opt.set_initial_step(dist_to_bound*gen_specs['dist_to_bound_multiple'])
    else:
        opt.set_initial_step(dist_to_bound)

    opt.set_maxeval(100*n) # evaluate one more point
    opt.set_min_objective(lambda x, grad: nlopt_obj_fun(x, grad, H, gen_specs, libE_info))
    opt.set_xtol_rel(gen_specs['xtol_rel'])
    
    try:
        x_opt = opt.optimize(x0)
        tag_out = FINISHED_PERSISTENT_GEN_TAG
    except Exception as e:
        x_opt = []
        tag_out = int(e.message.split('=')[-1])

    exit_code = opt.last_optimize_result()

    return x_opt, exit_code, tag_out

def uniform_or_localopt(H,gen_info,gen_specs,libE_info):
    ub = gen_specs['ub']
    lb = gen_specs['lb']

    if 'persistent' in libE_info and libE_info['persistent']:
        x_opt, exit_code, tag_out = try_and_run_nlopt(H, gen_specs,libE_info)

        if exit_code > 0 and exit_code < 5:
            O = np.zeros(1, dtype=gen_specs['out'])
            O = add_to_O(O,x_opt,0,ub,lb,local=True, active=True, minimum=True)
        else:
            O = []
        return O, gen_info, tag_out
    else:

        n = len(lb)
        b = gen_specs['gen_batch_size']

        O = np.zeros(b, dtype=gen_specs['out'])
        for i in range(0,b):
            # x = np.random.uniform(lb,ub,(1,n))
            x = gen_info['rand_stream'].uniform(lb,ub,(1,n))
            O = add_to_O(O,x,i,ub,lb)

        return O, gen_info

def add_to_O(O,x,i,ub,lb,local=False,active=False, minimum=False):
    O['x'][i] = x
    O['x_on_cube'][i] = (x-lb)/(ub-lb)
    O['dist_to_unit_bounds'][i] = np.inf
    O['dist_to_better_l'][i] = np.inf
    O['dist_to_better_s'][i] = np.inf
    O['ind_of_better_l'][i] = -1
    O['ind_of_better_s'][i] = -1
    if local:
        O['local_pt'] = True
    if active:
        O['num_active_runs'] = 1
    if minimum:
        O['local_min'] = True

    return O
