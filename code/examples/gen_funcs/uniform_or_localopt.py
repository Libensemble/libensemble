from __future__ import division
from __future__ import absolute_import

import numpy as np
from mpi4py import MPI

def set_up_and_run_nlopt(Run_H, gen_specs):
    """ Set up objective and runs nlopt

    Declares the appropriate syntax for our special objective function to read
    through Run_H, sets the parameters and starting points for the run.
    """

    def nlopt_obj_fun(x, grad, Run_H):
        jj
        f = 
        out = look_in_history(x, Run_H)

        if gen_specs['localopt_method'] in ['LD_MMA']:
            grad[:] = out[1]
            out = out[0]

        return f

    n = len(gen_specs['ub'])

    opt = nlopt.opt(getattr(nlopt,gen_specs['localopt_method']), n)

    lb = np.zeros(n)
    ub = np.ones(n)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    x0 = Run_H['x_on_cube'][0]

    # Care must be taken here because a too-large initial step causes nlopt to move the starting point!
    dist_to_bound = min(min(ub-x0),min(x0-lb))

    if 'dist_to_bound_multiple' in gen_specs:
        opt.set_initial_step(dist_to_bound*gen_specs['dist_to_bound_multiple'])
    else:
        opt.set_initial_step(dist_to_bound)

    opt.set_maxeval(len(Run_H)+1) # evaluate one more point
    opt.set_min_objective(lambda x, grad: nlopt_obj_fun(x, grad, Run_H))
    opt.set_xtol_rel(gen_specs['xtol_rel'])
    
    x_opt = opt.optimize(x0)
    exit_code = opt.last_optimize_result()

    if exit_code == 5: # NLOPT code for exhausting budget of evaluations, so not at a minimum
        exit_code = 0

    return x_opt, exit_code

def uniform_or_localopt(H,gen_info,gen_specs,libE_info):

    if 'persistent' in libE_info['persistent'] and libE_info['persistent']:

        while 1:
            D = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
            if status.Get_tag() == STOP_TAG: break
                    
            x = x - g_in['Hess_inv']*g_in['grad']

            O = np.zeros(1, dtype=gen_out)
            O['x'] = x
            O['priority'] = 1



    else:
        ub = gen_specs['ub']
        lb = gen_specs['lb']

        n = len(lb)
        b = gen_specs['gen_batch_size']

        O = np.zeros(b, dtype=gen_specs['out'])
        for i in range(0,b):
            # x = np.random.uniform(lb,ub,(1,n))
            # x = gen_info['rand_stream'].uniform(lb,ub,(1,n))
            x_on_cube = gen_info['rand_stream'].uniform(0,1,(1,n))

            O['x_on_cube'][i] = x_on_cube
            O['x'][i] = x_on_cube*(ub-lb)+lb
            O['dist_to_unit_bounds'][i] = np.inf
            O['dist_to_better_l'][i] = np.inf
            O['dist_to_better_s'][i] = np.inf
            O['ind_of_better_l'][i] = -1
            O['ind_of_better_s'][i] = -1

    return O, gen_info
