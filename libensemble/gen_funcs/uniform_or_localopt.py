from __future__ import division
from __future__ import absolute_import

import numpy as np
from mpi4py import MPI
import sys

from libensemble.message_numbers import UNSET_TAG, STOP_TAG, PERSIS_STOP, EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG

import nlopt

def uniform_or_localopt(H,gen_info,gen_specs,libE_info):
    """
    This generator
        - Returns "gen_batch_size" uniformly sampled points when called in
          nonpersistent mode. 
        - Performs a persistent nlopt local optimization run when called in
          persistent mode.
    """
    ub = gen_specs['ub']
    lb = gen_specs['lb']

    if 'persistent' in libE_info and libE_info['persistent']:
        x_opt, gen_info_updates, tag_out = try_and_run_nlopt(H, gen_specs,libE_info)
        O = []
        return O, gen_info_updates, tag_out
    else:
        n = len(lb)
        b = gen_specs['gen_batch_size']

        O = np.zeros(b, dtype=gen_specs['out'])
        for i in range(0,b):
            # x = np.random.uniform(lb,ub,(1,n))
            x = gen_info['rand_stream'].uniform(lb,ub,(1,n))
            O = add_to_O(O,x,i,ub,lb)

        gen_info_updates = gen_info # We want to send this back so it is over written.
        return O, gen_info_updates


def try_and_run_nlopt(H, gen_specs, libE_info):
    """ 
    Set up objective and runs nlopt performing communication with the manager in
    order receive function values for points of interest.
    """
    def nlopt_obj_fun(x, grad, H, gen_specs, comm):
        if np.array_equiv(x, H['x']):
            if gen_specs['localopt_method'] in ['LD_MMA']:
                grad[:] = H['grad']
            return np.float(H['f'])

        # Send back x to the manager
        O = np.zeros(1, dtype=gen_specs['out'])
        O = add_to_O(O,x,0,gen_specs['ub'],gen_specs['lb'],local=True,active=True)
                
        D = {'calc_out':O,
             'libE_info': {'persistent':True},
             'calc_status': UNSET_TAG,
             'calc_type': EVAL_GEN_TAG
            }
        #----------------------------------------------------------------------------------        
        comm.send(obj=D,dest=0,tag=EVAL_GEN_TAG)

        ## Receive information from the manager (or a STOP_TAG) 
        #status = MPI.Status()

        #libE_info = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)       

        #tag = status.Get_tag()
        #if tag in [STOP_TAG, PERSIS_STOP]:
            #nlopt.forced_stop.message = 'tag=' + str(tag)
            #raise nlopt.forced_stop

        #_ = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
        #calc_in = comm.recv(buf=None,source=0,tag=MPI.ANY_TAG,status=status)

        #----------------------------------------------------------------------------------


        # Receive information from the manager (or a STOP_TAG) 
        status = MPI.Status()     
        
        comm.probe(source=0, tag=MPI.ANY_TAG, status=status)          
        tag = status.Get_tag()
        if tag in [STOP_TAG, PERSIS_STOP]:            
            #sh - What is this doing...
            nlopt.forced_stop.message = 'tag=' + str(tag)
            raise nlopt.forced_stop           
            #man_signal = comm.recv(source=0, tag=STOP_TAG, status=status)
            #if man_signal == MAN_SIGNAL_FINISH: #shutdown the worker
                #break
        else:
            Work = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
        
        libE_info = Work['libE_info']
        calc_in = comm.recv(buf=None, source=0)
        #----------------------------------------------------------------------------------

        if gen_specs['localopt_method'] in ['LD_MMA']:
            grad[:] = calc_in['grad']

        f = float(calc_in['f'])

        return f

    x0 = H['x'].flatten()

    n = len(gen_specs['ub'])

    opt = nlopt.opt(getattr(nlopt,gen_specs['localopt_method']), n)

    # lb = np.zeros(n)
    # ub = np.ones(n)

    lb = gen_specs['lb']
    ub = gen_specs['ub']

    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)

    # Care must be taken with NLopt because a too-large initial step causes nlopt to move the starting point!
    dist_to_bound = min(min(ub-x0),min(x0-lb))

    if 'dist_to_bound_multiple' in gen_specs:
        opt.set_initial_step(dist_to_bound*gen_specs['dist_to_bound_multiple'])
    else:
        opt.set_initial_step(dist_to_bound)

    if 'localopt_maxeval' in gen_specs:
        opt.set_maxeval(gen_specs['localopt_maxeval'])
    else: 
        opt.set_maxeval(100*n) # evaluate one more point

    opt.set_min_objective(lambda x, grad: nlopt_obj_fun(x, grad, H, gen_specs, libE_info['comm']))
    opt.set_xtol_rel(gen_specs['xtol_rel'])
    
    # Try to peform a local optimization run. 
    #import pdb;pdb.set_trace()
    try:
        x_opt = opt.optimize(x0)
        exit_code = opt.last_optimize_result()

        if exit_code > 0 and exit_code < 5:
            gen_info_updates = {'done': True,'x_opt':x_opt} # Only send this back so new information added to gen_info since this persistent instance started (e.g., 'run_order'), is not overwritten
        else:
            gen_info_updates = {'done': True} # Only send this back so new information added to gen_info since this persistent instance started (e.g., 'run_order'), is not overwritten

        tag_out = FINISHED_PERSISTENT_GEN_TAG
    except Exception as e:
        # This exception is raised when the manager sends a PERSIS_STOP or
        # STOP_TAG signal
        x_opt = []
        gen_info_updates = {}
        tag_out = int(e.message.split('=')[-1])
        
    return x_opt, gen_info_updates, tag_out 


def add_to_O(O,x,i,ub,lb,local=False,active=False):
    """
    Builds or inserts points into the numpy structured array O that will be sent
    back to the manager.
    """
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

    return O
