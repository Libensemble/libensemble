"""
libEnsemble worker routines
====================================================
"""
from __future__ import division
from __future__ import absolute_import


from mpi4py import MPI
import numpy as np
import os, shutil 

from message_numbers import *

def worker_main(c, sim_specs, gen_specs):
    """ 
    Evaluate calculations given to it by the manager 

    Parameters
    ----------
    c: dict containing fields 'comm' and 'color' for the communicator. 

    sim_specs: dict with parameters/information for simulation calculations

    gen_specs: dict with parameters/information for generation calculations

    """
    comm, status, dtypes, locations = initialize_worker(c, sim_specs, gen_specs)

    while 1:
        # Receive libE_info from manager and check if STOP_TAG. 
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

        data_out, tag_out = perform_calc(calc_in, gen_info, libE_info, calc_tag, locations, sim_specs, gen_specs, comm) 
                            
        if tag_out == STOP_TAG: break

        comm.send(obj=data_out, dest=0, tag=tag_out) 

    # Clean up
    for loc in locations.values():
        shutil.rmtree(loc)

def perform_calc(calc_in, gen_info, libE_info, calc_tag, locations, sim_specs, gen_specs, comm):
    if calc_tag in locations:
        saved_dir = os.getcwd()
        os.chdir(locations[calc_tag])

    if 'persistent' in libE_info and libE_info['persistent']:
        libE_info['comm'] = comm

    if calc_tag == EVAL_SIM_TAG: 
        out = sim_specs['sim_f'][0](calc_in,gen_info,sim_specs,libE_info)
    elif calc_tag == EVAL_GEN_TAG: 
        out = gen_specs['gen_f'](calc_in,gen_info,gen_specs,libE_info)

    if isinstance(out,np.ndarray): 
        H = out
    elif isinstance(out, tuple):
        assert len(out) >= 2, "Calculation output must be at least two elements when a tuple"
        H = out[0]
        gen_info = out[1]
        if len(out) >= 3:
            calc_tag = out[2]
    else:
        sys.exit("Calculation output must be a tuple. Worker exiting")

    if calc_tag in locations:
        os.chdir(saved_dir)

    data_out = {'calc_out':H, 'gen_info':gen_info, 'libE_info': libE_info}

    return data_out, calc_tag

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
