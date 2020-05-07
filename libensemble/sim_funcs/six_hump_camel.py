"""
This module contains various versions that evaluate the six hump camel function.
"""
__all__ = ['six_hump_camel_with_different_ranks_and_nodes', 'six_hump_camel', 'six_hump_camel_simple']

# import subprocess
import os
import numpy as np
import time
from libensemble.executors.executor import Executor
from libensemble.message_numbers import UNSET_TAG, WORKER_DONE, TASK_FAILED


def six_hump_camel_with_different_ranks_and_nodes(H, persis_info, sim_specs, libE_info):
    """
    Evaluates the six hump camel for a collection of points given in ``H['x']`` but also
    performs a system call with a given number of nodes and ranks per node
    using a machinefile (to show one way of evaluating a compiled simulation).

    .. seealso::
        `test_uniform_sampling_with_different_resources.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_uniform_sampling_with_different_resources.py>`_ # noqa
    """

    from mpi4py import MPI
    batch = len(H['x'])
    H_o = np.zeros(batch, dtype=sim_specs['out'])

    exctr = Executor.executor  # Get Executor

    task_states = []
    for i, x in enumerate(H['x']):

        if 'blocking' in libE_info:
            ranks_involved = [MPI.COMM_WORLD.Get_rank()] + list(libE_info['blocking'])
        else:
            ranks_involved = [MPI.COMM_WORLD.Get_rank()]

        machinefilename = 'machinefile_for_sim_id=' + str(libE_info['H_rows'][i]) + \
                          '_resource_set='+'_'.join([str(r) for r in ranks_involved])

        with open(machinefilename, 'w') as f:
            for rank in ranks_involved:
                b = sim_specs['user']['nodelist'][rank] + '\n'
                f.write(b*H['ranks_per_node'][i])

        out_name = 'helloworld_sim_id=' + str(libE_info['H_rows'][i]) + \
                   '_resource_set='+'_'.join([str(r) for r in ranks_involved])

        outfile = out_name + ".out"
        errfile = out_name + ".err"
        for iofile in outfile, errfile:
            try:
                os.remove(iofile)
            except FileNotFoundError:
                pass

        # Run directly -------------------------------------------------------
        # call_str = ["mpiexec", "-machinefile", machinefilename,
        #             "python", os.path.join(os.path.dirname(__file__), "helloworld.py")]
        # p = subprocess.call(call_str, stdout=open(outfile, 'w'), stderr=open(errfile, 'w'), shell=False)
        # if p == 0:
            # task_states.append('FINISHED')
        # else:
            # task_states.append('FAILED')

        # Run with Executor --------------------------------------------------
        task = exctr.submit(calc_type='sim', machinefile=machinefilename,
                            stdout=outfile, stderr=errfile, hyperthreads=True)
        while(not task.finished):
            time.sleep(0.2)
            task.poll()
        task_states.append(task.state)

        H_o['f'][i] = six_hump_camel_func(x)

        # v = np.random.uniform(0, 10)
        # print('About to sleep for :' + str(v))
        # time.sleep(v)

    calc_status = UNSET_TAG  # Returns to worker
    if all(t == 'FINISHED' for t in task_states):
        calc_status = WORKER_DONE
    elif any(t == 'FAILED' for t in task_states):
        calc_status = TASK_FAILED

    return H_o, persis_info, calc_status


def six_hump_camel(H, persis_info, sim_specs, _):
    """
    Evaluates the six hump camel function for a collection of points given in ``H['x']``.
    Additionally evaluates the gradient if ``'grad'`` is a field in
    ``sim_specs['out']`` and pauses for ``sim_specs['user']['pause_time']]`` if
    defined.

    .. seealso::
        `test_old_aposmm_with_gradients.py  <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_old_aposmm_with_gradients.py>`_ # noqa
    """

    batch = len(H['x'])
    H_o = np.zeros(batch, dtype=sim_specs['out'])

    for i, x in enumerate(H['x']):
        H_o['f'][i] = six_hump_camel_func(x)

        if 'grad' in H_o.dtype.names:
            H_o['grad'][i] = six_hump_camel_grad(x)

        if 'user' in sim_specs and 'pause_time' in sim_specs['user']:
            time.sleep(sim_specs['user']['pause_time'])

    return H_o, persis_info


def six_hump_camel_simple(x, persis_info, sim_specs, _):
    """
    Evaluates the six hump camel function for a single point ``x``.

    .. seealso::
        `test_fast_alloc.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_fast_alloc.py>`_ # noqa
    """

    H_o = np.zeros(1, dtype=sim_specs['out'])

    H_o['f'] = six_hump_camel_func(x[0][0])

    if 'pause_time' in sim_specs['user']:
        time.sleep(sim_specs['user']['pause_time'])

    return H_o, persis_info


def six_hump_camel_func(x):
    """
    Definition of the six-hump camel
    """
    x1 = x[0]
    x2 = x[1]
    term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2
    term2 = x1*x2
    term3 = (-4+4*x2**2) * x2**2

    return term1 + term2 + term3


def six_hump_camel_grad(x):
    """
    Definition of the six-hump camel gradient
    """

    x1 = x[0]
    x2 = x[1]
    grad = np.zeros(2)

    grad[0] = 2.0*(x1**5 - 4.2*x1**3 + 4.0*x1 + 0.5*x2)
    grad[1] = x1 + 16*x2**3 - 8*x2

    return grad
