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
    rank = comm.Get_rank()
    status = MPI.Status()

    while 1:
        D = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)

        if status.Get_tag() == STOP_TAG: break

        assert(len(D['form_subcomm'])==0), "Haven't implemented form_subcomm yet"

        if 'sim_dir' in D['calc_params']:
            saved_dir = os.getcwd()
            worker_dir = D['calc_params']['sim_dir'] + '_' + str(comm_color) + "_" + str(rank) 

            if 'sim_dir_prefix' in D['calc_params']:
                worker_dir = os.path.join(os.path.expanduser(D['calc_params']['sim_dir_prefix']), os.path.split(os.path.abspath(os.path.expanduser(worker_dir)))[1])

            # assert ~os.path.isdir(worker_dir), "Worker directory already exists."
            if not os.path.exists(worker_dir):
                shutil.copytree(D['calc_params']['sim_dir'], worker_dir)
            os.chdir(worker_dir)

        O = D['calc_f'](D['calc_in'],D['calc_out'],D['calc_params'],D['calc_info'])

        if 'sim_dir' in D['calc_params']:
            os.chdir(saved_dir)

        data_out = {'calc_out':O, 'calc_info': D['calc_info']}
        
        comm.send(obj=data_out, dest=0) 

    # Clean up
    if 'saved_dir' in locals():
        shutil.rmtree(worker_dir)
