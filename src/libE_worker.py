"""
libEnsemble worker routines
====================================================
"""
from __future__ import division
from __future__ import absolute_import


from mpi4py import MPI
import numpy as np
import time, sys, os, shutil 

from message_numbers import EVAL_TAG # manager tells worker to evaluate the point 
from message_numbers import STOP_TAG # manager tells worker run is over

def worker_main(c, allocation_specs, sim_specs, failure_processing):
    """ 
    Evaluate obj_func at point(s) passed to it by the manager 

    Parameters
    ----------
    comm: mpi4py communicator used to communicate with manager
    sim_specs: information on running the simulation 

    If obj_dir is not given, then it it assumed that no files are written/read
    out that can conflict with concurrent evaluations of sim_f. 

    If obj_dir is given, then it will be copied (using shuitl.copytree with
    symlinks=False) to a worker specific directory, and the worker will change
    into that directory to do all of it's work. 
    """
    comm = c['comm']
    comm_color = c['color']
    n = sim_specs['sim_f_params']['n']
    m = sim_specs['sim_f_params']['m']
    obj_func = sim_specs['sim_f']


    if 'obj_dir' in sim_specs['sim_f_params'].keys():
        obj_dir = sim_specs['sim_f_params']['obj_dir']
    else:
        obj_dir = None

    if 'sim_data' in sim_specs['sim_f_params'].keys():
        sim_data = sim_specs['sim_f_params']['sim_data']
    else:
        sim_data = None


    # size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    status = MPI.Status()

    if rank in allocation_specs['lead_worker_ranks']:
        # Create and move into the worker directory if necessary
        if obj_dir is not None:
            # worker_dir = '/scratch/' + obj_dir + '_' + str(comm_color) + "_" + str(rank) 
            worker_dir = obj_dir.split('/')[-1] + '_' + str(comm_color) + "_" + str(rank) 
            if os.path.exists(worker_dir):
                print("DELETING existing Worker directory.")
                sys.stdout.flush()
                shutil.rmtree(worker_dir)
            saved_dir = os.getcwd()
            shutil.copytree(obj_dir, worker_dir)
            os.chdir(worker_dir)

        while 1:
            data_in = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG,
                    status=status)
            if status.Get_tag() == STOP_TAG: break

            x_true = np.atleast_2d(data_in['x_true'])
            batch_size = x_true.shape[0]

            data_out = np.zeros(batch_size ,dtype=[('x_true','float',n),
                                    ('pt_id','int'), ('f_vec','float',m),
                                    ('grad','float',(m,n)),
                                    ('lead_rank','int'),
                                    ('worker_start_time','float'), ('worker_end_time','float')
                                    ])

            for x, i in zip(x_true, range(0,batch_size)):
                # print("Worker: %d; Starting to work on %r" % (rank,x))
                data_out['x_true'][i] = x

                data_out['worker_start_time'][i] = time.time()
                if sim_data is None:
                    obj_out = obj_func(x_true) 
                else:
                    obj_out = obj_func(x_true, sim_data) 
                data_out['worker_end_time'][i] = time.time()

                if isinstance(obj_out, tuple) and len(obj_out) == 2:
                    data_out['f_vec'][i] = obj_out[0]
                    data_out['grad'][i] = obj_out[1]
                elif isinstance(obj_out, np.float):
                    data_out['f_vec'][i] = obj_out
                elif isinstance(obj_out, np.ndarray):
                    if len(obj_out) == 1:
                        data_out['f_vec'][i] = obj_out[0]
                    else:
                        data_out['f_vec'][i] = obj_out
                else:
                    sys.exit("Objective output must be a tuple (f,gradient), a"\
                            "np.ndarray (f_vec) or a np.float (f)")

            data_out['lead_rank'] = rank
            data_out['pt_id'] = data_in['pt_id']

            comm.send(obj=data_out, dest=0) 
            # print("Worker: %d; Finished work on %r; finished sending" % (rank,[x,f_vec,f,0,0,rank]))

    elif comm.Get_rank() in worker_nonlead_ranks: 
        print("I am a non-lead worker rank %d on %s" % (rank,name))

    # Clean up
    if obj_dir is not None:
        os.chdir(saved_dir)
        shutil.rmtree(worker_dir)
