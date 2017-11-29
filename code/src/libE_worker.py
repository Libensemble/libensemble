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
    c: dict containing fields 'comm' and 'color' for the communicator. 

    sim_specs: dict with parameters/information for simulation calculations

    gen_specs: dict with parameters/information for generation calculations

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

        assert ~os.path.isdir(worker_dir), "Worker directory already exists."
        # if not os.path.exists(worker_dir):
        shutil.copytree(sim_specs['sim_dir'], worker_dir)

        locations[EVAL_SIM_TAG] = worker_dir 

    while 1:
        libE_info = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
        calc_tag = status.Get_tag()
        if calc_tag == STOP_TAG: break

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

        if calc_tag in locations:
            saved_dir = os.getcwd()
            os.chdir(locations[calc_tag])

        if calc_tag == EVAL_SIM_TAG: 
            H, gen_info = sim_specs['sim_f'][0](calc_in,gen_info,sim_specs,libE_info)
        else: 
            H, gen_info = gen_specs['gen_f'](calc_in,gen_info,gen_specs,libE_info)

        if calc_tag in locations:
            os.chdir(saved_dir)

        data_out = {'calc_out':H, 'gen_info':gen_info, 'libE_info': libE_info}
        
        comm.send(obj=data_out, dest=0, tag=calc_tag) 

    # Clean up
    if 'saved_dir' in locals():
        shutil.rmtree(worker_dir)
