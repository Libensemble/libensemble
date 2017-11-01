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
    comm, status, dtypes, locations = initialize_worker(c, sim_specs, gen_specs)

    while 1:
        # Receive libE_info from manager and check if STOP_TAG. 
        libE_info = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
        calc_tag = status.Get_tag()
        if calc_tag == STOP_TAG: break

        calc_in, calc_info = receive_calc(comm, calc_in_len, calc_tag, dtypes)

        data_out, tag_out = perform_calc(calc_in, calc_info, calc_tag, locations, sim_specs, gen_specs, comm) 
                            
        if tag_out == STOP_TAG: break
        if tag_out == FINISHED_PERSISTENT_GEN_TAG: 
            _ = comm.recv(buf=None, source=0,tag=MPI.ANY_TAG, status=status) # Need to receive an advance signal from manager
            if status.Get_tag() == STOP_TAG: break

        comm.send(obj=data_out, dest=0, tag=tag_out) 


        if calc_tag in locations:
            saved_dir = os.getcwd()
            os.chdir(locations[calc_tag])


        if calc_tag in locations:
            os.chdir(saved_dir)

        data_out = {'calc_out':H, 'gen_info':gen_info, 'libE_info': libE_info}
        
        comm.send(obj=data_out, dest=0, tag=calc_tag) 

    # Clean up
    for loc in locations.values():
        shutil.rmtree(loc)

def perform_calc(calc_in, calc_info, calc_tag, locations, sim_specs, gen_specs, comm):
    if calc_tag in locations:
        saved_dir = os.getcwd()
        os.chdir(locations[calc_tag])

    if calc_tag == EVAL_SIM_TAG: 
        H, gen_info = sim_specs['sim_f'][0](calc_in,gen_info,sim_specs,libE_info)
    elif calc_tag == EVAL_GEN_TAG: 
        if 'persistent' in calc_info and calc_info['persistent']:
            libE_info['comm'] = comm

        O = gen_specs['gen_f'](calc_in,gen_specs['out'],gen_specs['params'],libE_info)


    if calc_tag in locations:
        os.chdir(saved_dir)

    data_out = {'calc_out':O, 'calc_info': calc_info}

    return data_out, tag_out

def receive_calc(comm, calc_in_len, calc_tag, dtypes):
    gen_info = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
    calc_in = np.zeros(len(libE_info['H_rows']),dtype=dtypes[calc_tag])

    if len(calc_in) > 0: 
        calc_in = comm.recv(buf=None, source=0)
        # for i in calc_in.dtype.names: 
        #     # d = comm.recv(buf=None, source=0)
        #     # data = np.empty(calc_in[i].shape, dtype=d)
        #     data = np.empty(calc_in[i].shape, dtype=calc_in[i].dtype)
        #     comm.Recv(data,source=0)
        #     calc_in[i] = data

    return calc_in, gen_info

def initialize_worker(c, sim_specs, gen_specs):
    """ Receive sim and gen dtypes, copy sim_dir """

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

        assert ~os.path.isdir(worker_dir), "Worker directory already exists."
        # if not os.path.exists(worker_dir):
        shutil.copytree(sim_specs['sim_dir'], worker_dir)
        locations[EVAL_SIM_TAG] = worker_dir 

    return comm, status, dtypes, locations 
