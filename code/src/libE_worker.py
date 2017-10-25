"""
libEnsemble worker routines
====================================================
"""
from __future__ import division
from __future__ import absolute_import


from mpi4py import MPI
import numpy as np
import os, shutil 

from message_numbers import STOP_TAG # manager tells worker to stop
from message_numbers import EVAL_SIM_TAG 
from message_numbers import EVAL_GEN_TAG 

def worker_main(c, sim_specs, gen_specs):
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

    dtypes = {}

    dtypes[EVAL_SIM_TAG] = comm.recv(buf=None, source=0)
    dtypes[EVAL_GEN_TAG] = comm.recv(buf=None, source=0)

    
    locations = {}

    # Make the directory for the worker to do their sim work in
    if 'sim_dir' in sim_specs:
        worker_dir = sim_specs['sim_dir'] + '_' + str(comm_color) + "_" + str(rank) 

        if 'sim_dir_prefix' in sim_specs:
            worker_dir = os.path.join(os.path.expanduser(sim_specs['sim_dir_prefix']), os.path.split(os.path.abspath(os.path.expanduser(worker_dir)))[1])

        # assert ~os.path.isdir(worker_dir), "Worker directory already exists."
        if not os.path.exists(worker_dir):
            shutil.copytree(sim_specs['sim_dir'], worker_dir)
        locations[EVAL_SIM_TAG] = worker_dir 

    while 1:
        calc_in_len = np.empty(1,dtype=int)
        comm.Recv(calc_in_len,source=0,tag=MPI.ANY_TAG,status=status)

        calc_tag = status.Get_tag()
        if calc_tag == STOP_TAG: break

        calc_in = np.zeros(calc_in_len,dtype=dtypes[calc_tag])

        if calc_in_len > 0: 
            for i in calc_in.dtype.names: 
                d = comm.recv(buf=None, source=0)
                data = np.empty(calc_in[i].shape, dtype=d)
                comm.Recv(data,source=0)
                calc_in[i] = data

        calc_info = comm.recv(buf=None, source=0)

        assert 'form_subcomm' not in calc_info or len(calc_info['form_subcomm'])==0, "Haven't implemented form_subcomm yet"


        if calc_tag in locations:
            saved_dir = os.getcwd()
            os.chdir(locations[calc_tag])

        if calc_tag == EVAL_SIM_TAG: 
            O = sim_specs['sim_f'][0](calc_in,[],sim_specs,calc_info)
        else: 
            O = gen_specs['gen_f'](calc_in,[],gen_specs,calc_info)

        if calc_tag in locations:
            os.chdir(saved_dir)

        data_out = {'calc_out':O, 'calc_info': calc_info}
        
        comm.send(obj=data_out, dest=0) 

    # Clean up
    if 'saved_dir' in locals():
        shutil.rmtree(worker_dir)
