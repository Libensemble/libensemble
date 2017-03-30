"""
libEnsemble worker routines
====================================================
"""
from __future__ import division
from __future__ import absolute_import


from mpi4py import MPI
import numpy as np
import time, sys, os, shutil 

from message_numbers import STOP_TAG # manager tells worker to stop

def worker_main(c):
    """ 
    Evaluate calculations given to it by the manager 

    Parameters
    ----------
    comm: mpi4py communicator used to communicate with manager

    The data sent from the manager should have the following:

    D['form_subcomm']: List of integer ranks in comm that should form subcomm
        (empty list means no subcomm needed)

    D['calc_dir']: String for directory to be copied where work should be
        performed (empty string means no directory needed) 
    """
    comm = c['comm']
    comm_color = c['color']


    # size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    status = MPI.Status()

    while 1:
        D = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
        # print(D)
        sys.stdout.flush()

        if status.Get_tag() == STOP_TAG: break

        if len(D['form_subcomm']):
            sys.exit("Haven't implemented this yet")

        if len(D['calc_dir']):
            saved_dir = os.getcwd()
            # worker_dir = '/scratch/' + obj_dir + '_' + str(comm_color) + "_" + str(rank) 
            worker_dir = D['calc_dir'] + '_' + str(comm_color) + "_" + str(rank) 
            # assert ~os.path.isdir(worker_dir), "Worker directory already exists."
            if not os.path.exists(worker_dir):
                shutil.copytree(obj_dir, worker_dir)
            os.chdir(worker_dir)

        O = D['calc_f'](D['calc_in'],D['calc_out'],D['calc_params'])

        data_out = {'calc_out':O, 'calc_type': D['calc_type']}
        
        comm.send(obj=data_out, dest=0) 
        # print("Worker: %d; Finished work on %r; finished sending" % (rank,[x,f_vec,f,0,0,rank]))

    # Clean up
    if obj_dir is not None:
        os.chdir(saved_dir)
        shutil.rmtree(worker_dir)
